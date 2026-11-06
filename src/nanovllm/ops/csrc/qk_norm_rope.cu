#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cassert>

namespace {

// One warp owns a head; each lane holds corresponding split-half RoPE pairs.
template <typename scalar_t>
__global__ void qk_norm_rope_kernel(
    const scalar_t* qkv, const int64_t* positions, const scalar_t* q_weight,
    const scalar_t* k_weight, const float* cache, scalar_t* q, scalar_t* k,
    int64_t tokens, int64_t row_stride, int q_heads, int kv_heads,
    int dim, int64_t max_position, float eps, bool round_normalized) {
    const int lane = threadIdx.x % 32;
    const int64_t head_index = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    const int heads = q_heads + kv_heads;
    if (head_index >= tokens * heads) return;
    const int64_t token = head_index / heads;
    const int head = head_index % heads;
    const bool is_q = head < q_heads;
    const int output_head = is_q ? head : head - q_heads;
    const int output_heads = is_q ? q_heads : kv_heads;
    const scalar_t* input = qkv + token * row_stride + head * dim;
    const scalar_t* weight = is_q ? q_weight : k_weight;
    scalar_t* output = (is_q ? q : k) + (token * output_heads + output_head) * dim;
    const int64_t position = positions[token];
    assert(position >= 0 && position < max_position);
    const float* trig = cache + position * dim;
    const int half = dim / 2;
    float x1[4], x2[4];
    float square_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = lane + i * 32;
        x1[i] = col < half ? float(input[col]) : 0.f;
        x2[i] = col < half ? float(input[col + half]) : 0.f;
        square_sum += x1[i] * x1[i] + x2[i] * x2[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        square_sum += __shfl_xor_sync(0xffffffff, square_sum, offset);
    const float inv_rms = rsqrtf(square_sum / dim + eps);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = lane + i * 32;
        if (col < half) {
            // Inductor may elide the normalize downcast; preserve the norm output cast.
            float n1 = x1[i] * inv_rms, n2 = x2[i] * inv_rms;
            if (round_normalized) {
                n1 = float(scalar_t(n1));
                n2 = float(scalar_t(n2));
            }
            const float a = float(scalar_t(n1 * float(weight[col])));
            const float b = float(scalar_t(n2 * float(weight[col + half])));
            const float cos = trig[col], sin = trig[col + half];
            output[col] = scalar_t(a * cos - b * sin);
            output[col + half] = scalar_t(b * cos + a * sin);
        }
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> fused_qk_norm_rope_cuda(
    const at::Tensor& qkv, const at::Tensor& positions,
    const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& cos_sin_cache, int64_t num_q_heads,
    int64_t num_kv_heads, double eps, bool round_normalized) {
    TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA");
    TORCH_CHECK(qkv.scalar_type() == at::kFloat || qkv.scalar_type() == at::kHalf
                || qkv.scalar_type() == at::kBFloat16, "qkv must be float32, float16 or bfloat16");
    const c10::cuda::CUDAGuard guard(qkv.device());
    for (const auto& tensor : {positions, q_weight, k_weight, cos_sin_cache})
        TORCH_CHECK(tensor.device() == qkv.device(), "all inputs must be on the same CUDA device");
    TORCH_CHECK(qkv.dim() == 2 && qkv.stride(1) == 1,
                "qkv must be a packed 2D tensor with contiguous columns");
    TORCH_CHECK(q_weight.dim() == 1 && k_weight.sizes() == q_weight.sizes()
                && q_weight.is_contiguous() && k_weight.is_contiguous(),
                "Q/K weights must be contiguous vectors of the same shape");
    const int64_t dim = q_weight.numel();
    TORCH_CHECK(dim > 0 && dim <= 256 && dim % 2 == 0,
                "head_dim must be even and in [2, 256]");
    TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && eps > 0,
                "head counts and eps must be positive");
    TORCH_CHECK(qkv.size(1) == (num_q_heads + 2 * num_kv_heads) * dim,
                "packed QKV width does not match head counts");
    TORCH_CHECK(positions.scalar_type() == at::kLong && positions.dim() == 1
                && positions.size(0) == qkv.size(0) && positions.is_contiguous(),
                "positions must be contiguous int64 with one entry per token");
    TORCH_CHECK(q_weight.scalar_type() == qkv.scalar_type()
                && k_weight.scalar_type() == qkv.scalar_type(),
                "Q/K weights must have the same dtype as qkv");
    TORCH_CHECK(cos_sin_cache.scalar_type() == at::kFloat
                && cos_sin_cache.dim() == 3 && cos_sin_cache.size(1) == 1
                && cos_sin_cache.size(2) == dim && cos_sin_cache.is_contiguous(),
                "cos_sin_cache must be contiguous float32 [max_position, 1, head_dim]");
    auto q = at::empty({qkv.size(0), num_q_heads, dim}, qkv.options());
    auto k = at::empty({qkv.size(0), num_kv_heads, dim}, qkv.options());
    if (qkv.size(0) == 0) return {q, k};
    const int64_t warps = qkv.size(0) * (num_q_heads + num_kv_heads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qkv.scalar_type(),
        "fused_qk_norm_rope", [&] {
            qk_norm_rope_kernel<scalar_t><<<(warps + 3) / 4, 128, 0,
                at::cuda::getCurrentCUDAStream()>>>(
                qkv.data_ptr<scalar_t>(), positions.data_ptr<int64_t>(),
                q_weight.data_ptr<scalar_t>(), k_weight.data_ptr<scalar_t>(),
                cos_sin_cache.data_ptr<float>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                qkv.size(0), qkv.stride(0), num_q_heads, num_kv_heads, dim,
                cos_sin_cache.size(0), float(eps), round_normalized);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q, k};
}
