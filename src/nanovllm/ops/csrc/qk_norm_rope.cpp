#include <ATen/ATen.h>
#include <torch/library.h>

std::tuple<at::Tensor, at::Tensor> fused_qk_norm_rope_cuda(
    const at::Tensor& qkv, const at::Tensor& positions,
    const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& cos_sin_cache, int64_t num_q_heads,
    int64_t num_kv_heads, double eps, bool round_normalized);

std::tuple<at::Tensor, at::Tensor> fused_qk_norm_rope_meta(
    const at::Tensor& qkv, const at::Tensor& positions,
    const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& cos_sin_cache, int64_t num_q_heads,
    int64_t num_kv_heads, double eps, bool round_normalized) {
    const auto dim = q_weight.sym_numel();
    return {at::empty_symint({qkv.sym_size(0), num_q_heads, dim}, qkv.options()),
            at::empty_symint({qkv.sym_size(0), num_kv_heads, dim}, qkv.options())};
}

TORCH_LIBRARY(nanovllm, m) {
    m.def("fused_qk_norm_rope(Tensor qkv, Tensor positions, Tensor q_weight, "
          "Tensor k_weight, Tensor cos_sin_cache, int num_q_heads, "
          "int num_kv_heads, float eps, bool round_normalized=True) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(nanovllm, CUDA, m) {
    m.impl("fused_qk_norm_rope", &fused_qk_norm_rope_cuda);
}

TORCH_LIBRARY_IMPL(nanovllm, Meta, m) {
    m.impl("fused_qk_norm_rope", &fused_qk_norm_rope_meta);
}
