# nano-vLLM 统一调度与异步执行

## 本地整合：缓存策略与独立开关

当前 `src/nanovllm` 已整合 `nano-vllm-queue-aware-cache` 中的等待队列前瞻、
uncached-first 和侵入式空闲块链表，并保留本项目的混合 batch、增量 KV 分配、
异步执行及 full/piecewise CUDA Graph。原始上游、`baseline/` 和独立缓存项目保留作对照。
另提供 CUDA Q/K RMSNorm + RoPE 融合算子，通过 PyTorch dispatcher 注册，支持独立开关。

| 参数 | 默认值 | 开启/关闭的行为 |
|---|---|---|
| `prefix_cache_lookahead` | `0` | `0` 关闭前瞻；正数限制检查 waiting 队列的条目数 |
| `uncached_first` | `False` | 优先回收无有效缓存的空闲块 |
| `intrusive_free_queue` | `False` | 使用侵入式双向链表；关闭时使用 deque |
| `fused_qk_norm_rope` | `False` | 使用 CUDA 融合 Q/K RMSNorm 与 RoPE；关闭时保留原模型路径 |
| `mixed_batching` | `True` | 允许一个 batch 同时包含 Prefill 和 Decode；关闭时每批只包含一类 |
| `incremental_kv_allocation` | `True` | 只分配本轮所需 KV；关闭时至少分配完整已知序列所需 KV |
| `async_scheduling` | `True` | 允许有界异步提交；关闭时每批提交后立即回收 |
| `max_inflight_batches` | `2` | 异步深度，可选 1/2；同步模式强制为 1 |
| `cuda_graph_mode` | `'auto'` | `'eager'` 关闭 Graph；也可选择 `'full'`、`'piecewise'` |

新增缓存选项默认关闭，保持整合前的默认分配行为。每项可以独立选择；队列容器不会改变缓存选择策略。
这些参数在创建实例时确定，不支持运行中热切换。

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/home/smh/huggingface/Qwen3-0.6B",
    prefix_cache_lookahead=16,
    uncached_first=True,
    intrusive_free_queue=True,
    fused_qk_norm_rope=True,
    mixed_batching=True,
    incremental_kv_allocation=True,
    async_scheduling=True,
    cuda_graph_mode="auto",
)
outputs = llm.generate(["Explain KV cache."], SamplingParams(max_tokens=32))
```

进行逐项消融时，只改变对应字段。关闭上述可选优化可使用：

```python
llm = LLM(
    model_path,
    prefix_cache_lookahead=0,
    uncached_first=False,
    intrusive_free_queue=False,
    fused_qk_norm_rope=False,
    mixed_batching=False,
    incremental_kv_allocation=False,
    async_scheduling=False,
    cuda_graph_mode="eager",
)
```

关闭混合 batch 只限制批次组成，仍保留当前统一进度和快照架构，不会恢复上游的 Prefill 优先策略。
关闭增量分配不会预留全部未来生成 token，准入检查 `admission_full_sequence` 与实际分配仍是不同概念。
要对比原始 nano-vLLM，继续使用后文的独立 upstream 基线。

前瞻仅复用 `cached_prefix()` 返回的已确认完整块，在一次分配调用中惰性构建一次 retained 集合；
当前请求先引用自身命中的前缀，再选择新块。全部候选被软保留时仍回退队头，且始终只选择引用计数为零的块。
EOS、取消、抢占和前缀发布继续遵守现有在途保护。链表只将已知块摘除变成 O(1)，候选扫描仍可能为 O(F)。

```bash
PYTHONPATH=src python -m pytest tests -q
python scripts/smoke_v1.py --model /path/to/Qwen3-0.6B --mode auto \
  --async-scheduling --prefix-cache-lookahead 16 --uncached-first --intrusive-free-queue
```

`smoke_v1.py` 还支持 `--no-mixed-batching` 和 `--no-incremental-kv-allocation`。
增加 `--fused-qk-norm-rope` 可与以上选项同时启用 CUDA 融合。
`scripts/check_integrated.py --model /path/to/Qwen3-0.6B --output /path/to/checks.json`
在四个独立进程中检查关闭开关、缓存同步 eager、缓存异步 auto，以及异步分开 batch/整段分配。
它比较相同冷请求的四行完整 logits，并检查重复前缀压力负载的有限数值、固定生成历史和资源排空；
后者的缓存命中和执行形状可能不同，不要求跨策略逐位一致。诊断读取 logits 会同步 GPU，不用于计时。
后文历史性能数据和源码哈希对应整合前版本，不作为整合后的性能结论；原打包清单也保留原版本记录。

本地整合验证（2026-11-04）：231 项测试、19 项 subtest 通过，包含 64 组策略/批次/分配/在途深度组合。
RTX 4070 Ti SUPER / Qwen3-0.6B BF16 的四组独立进程 GPU 检查通过：
相同冷请求四个输出位置的完整 logits 逐位一致；重复前缀压力负载数值有限、固定续写和资源排空正确。
压力用例确认了新策略的前缀命中，不据此推断吞吐提升。
另一次不读取 logits 的全开 smoke 实际执行了混合 forward、full/piecewise Graph 和 GPU token 反馈。
[GPU 检查](results/integrated-validation-20261104/gpu_checks.json) ·
[回归日志](results/integrated-validation-20261104/pytest.log)

### CUDA Q/K RMSNorm + RoPE

实现位于 `src/nanovllm/ops/csrc/`，不是 Triton kernel：
`TORCH_LIBRARY` 定义 schema，`TORCH_LIBRARY_IMPL` 注册 CUDA 和 Meta 实现，
Python 通过 `torch.ops.nanovllm.fused_qk_norm_rope` 调用。
一个 warp 处理一个 token 的一个 Q/K head，FP32 归约与旋转，Q/K 一次 kernel 输出；
V 保留原始 packed QKV 的 view。普通 forward、full Graph 和 piecewise Graph 均接入。

支持 FP16/BF16/FP32，偶数 `head_dim` 2–256，连续列且可带行间距的 `[T, (Hq+2Hkv)*D]` QKV，
同 dtype 的连续 Q/K 权重、连续 int64 positions，以及 FP32 `[max_position, 1, D]` cos/sin cache。
位置必须在 `[0, max_position)`。使用 split-half RoPE；当前仅用于无 QKV bias、具有 Q/K RMSNorm 的路径。
这是推理算子，不提供 backward。

归一化和旋转使用 FP32，保留 RMSNorm 输出到低精度的舍入，再执行 RoPE。
原 `@torch.compile` RMSNorm 是否保留归一化后、乘权重前的舍入，取决于
`torch._inductor.config.emulate_precision_casts`；初始化时读取该配置并传入算子以匹配原执行语义。
直接调用算子时 `round_normalized=True` 默认匹配 Python eager 的两个舍入节点，
传 `False` 匹配当前 Inductor 默认的中间转换消除。归约顺序不同，低精度输出不承诺逐位相等。

首次开启时在模型预热和 Graph capture **之前** JIT 编译并加载扩展，之后复用 PyTorch 扩展缓存。
需要与 PyTorch 兼容的 CUDA Toolkit（含 nvcc）、C++ 编译器和 ninja；仅安装 PyTorch CUDA wheel 不够。
本机已验证的环境是 WSL Ubuntu 22.04、CUDA Toolkit 12.8、PyTorch 2.11.0+cu128、SM89。

```bash
source /home/smh/ai_infra/nano-vllm/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install ninja
PYTHONPATH=src python -m pytest tests/test_cuda_qk_norm_rope.py -q
python scripts/smoke_v1.py --model /path/to/Qwen3-0.6B --mode auto \
  --async-scheduling --fused-qk-norm-rope \
  --prefix-cache-lookahead 16 --uncached-first --intrusive-free-queue
python scripts/check_integrated.py --model /path/to/Qwen3-0.6B \
  --include-cuda-fusion --output /path/to/checks.json
```

最后一项在原四组对照之外增加融合同步 eager、融合异步 auto；
融合模型的四行完整 BF16 logits 使用预设 `rtol=0.01, atol=0.125` 校验，
原四组继续要求逐位一致。此处不据正确性测试推断性能收益。

本次本机验证（2026-11-04）：268 项测试、19 项 subtest 通过。
六组独立进程的真实模型检查均通过；包括融合同步 eager 和融合异步 auto，
固定生成历史下四个输出位置的完整 logits 实际均逐位一致（最大绝对误差 0）。
融合异步用例执行了 11 次 full Graph、20 次 piecewise Graph，捕获无错误、资源排空正确。
逐位一致仅是该 BF16 用例的实测结果，不作为所有形状和精度的保证。
不读取诊断 logits 的全开 smoke 也通过：4 次 mixed forward、7 次 full Graph、
4 次 piecewise Graph、14 个 GPU 反馈 token，最大在途深度为 2，捕获无错误。
[模型检查](results/cuda-fusion-validation-20261104/model-checks.json) ·
[完整测试日志](results/cuda-fusion-validation-20261104/all-tests.log) ·
[全开 smoke](results/cuda-fusion-validation-20261104/smoke-auto.log)

### RTX 4070 Ti SUPER 锁频复测

2026-11-04，Qwen3-0.6B BF16，SM 锁定 2400 MHz。独立缓存方案相对原始 upstream
吞吐 +11.53%、Prefill -23.58%；预算 512、4 条 Decode + 4 条 3000-token Prompt 下，
统一调度/异步 auto 吞吐 +65.13%，已有请求最大 token 间隔均值 -95.82%。
CUDA 融合的 32-token 前处理耗时 5.398 → 1.794 μs；十对独立进程的六种本地负载
端到端延迟降低 1.52%–4.02%。Profiler 确认该 32-token 用例为 4 → 1 个 CUDA kernel。

整合后收益不直接相加：8192-token budget 的重复前缀压力负载中，整合版开启缓存
吞吐 +8.05%，缓存与融合一起开启 +9.72%，但其绝对吞吐低于独立缓存版本。
扩大数值测试后，部分 logits 不再逐位一致：相对原始上游最大相对 L2 为 2.72%，
隔离融合差异时为 2.77%；满足原项目预设的 3% 门槛，不代表生成历史完全一致。
完整性能矩阵、进程配对统计、数值边界、源码哈希和重跑方法见
[锁频复测报告](results/resume-reproduction-20261104/locked2400/report.md)。

---

在已有的混合 batch 单次 forward 基础上，参考 vLLM V1 的调度思路，实现统一 token 进度、增量 KV 分配、分段 CUDA Graph 和有界异步执行。

以下介绍整合前的调度项目及历史实验。`baseline/nanovllm` 冻结上一版实现，原 `nano-vllm-mixed-scheduling` 项目保持不变。对照基线已经支持混合 batch 单次 forward，不能把本轮收益解释为“两次 forward 变一次”。当时模型层与 Attention 沿用 nano-vLLM 算子，未启用自定义融合扩展。

## 问题与实现

| 项目 | 冻结基线 | 本版 |
|---|---|---|
| 请求队列 | 未完成 Prompt 留在 waiting，Decode 优先 | 首次准入即进入 running；先遍历 running，再准入 waiting |
| 计算预算 | Decode 固定 1，剩余预算分给 Prefill | 按已知 token、待返回输出和已安排进度统一计算需要处理的 token 数 |
| KV 分配 | 首次预留整段已知序列 | 准入检查与实际分配分开；只分配到本轮目标位置 |
| 混合执行 | 一次 forward | 一次 forward；保持每个请求独立的 Q 边界、KV 长度和 block table |
| CUDA Graph | 混合 batch eager，纯 Decode full Graph | 支持 eager、piecewise、full、auto；依据查询形状和实际捕获结果分发 |
| 调度与执行 | 每轮回收后才能安排下一轮 | 单 GPU 最多两批在途；下一轮输入可直接使用 GPU 上的上一轮采样结果 |

Scheduler 为每轮生成不可变快照，分别记录已安排、已确认和在途 token。只有确认完成的完整 KV block 才能发布为可复用前缀；抢占、EOS 和取消都不能释放仍被在途 batch 引用的 block。

```text
CPU：安排 N → 提交 N → 安排 N+1 → 提交 N+1 → 回收 N → …
GPU：计算 N → 采样 N → 将采样结果填入 N+1 输入 → 计算 N+1 → …
```

GPU 计算仍遵守同一流上的依赖顺序；异步执行重叠的是 CPU 调度、输入准备和 GPU 工作，不表示同一请求的两次 forward 可以独立并行。中间 Prefill chunk 只推进 KV，最终 chunk 才接收输出 token。

| 改动文件 | 职责 |
|---|---|
| [scheduler.py](src/nanovllm/engine/scheduler.py)、[sequence.py](src/nanovllm/engine/sequence.py) | running/waiting 准入、统一进度、预算、抢占和输出占位 |
| [block_manager.py](src/nanovllm/engine/block_manager.py) | 增量分配、前缀复用、引用计数和安全释放 |
| [schedule_output.py](src/nanovllm/engine/schedule_output.py)、[batch_input.py](src/nanovllm/engine/batch_input.py) | 不可变批次快照、Q/K 边界、位置、slot 和 GPU token 依赖 |
| [llm_engine.py](src/nanovllm/engine/llm_engine.py)、[execution.py](src/nanovllm/engine/execution.py) | 有界提交队列、完成事件、FIFO 回收、取消与状态清理 |
| [model_runner.py](src/nanovllm/engine/model_runner.py)、[graph_runner.py](src/nanovllm/engine/graph_runner.py) | GPU 反馈、异步结果拷贝、Graph 捕获和运行分发 |
| [config.py](src/nanovllm/config.py) | 调度、在途深度、准入和 Graph 配置 |

## Graph 选择与数值问题

`auto` 优先为每条请求 Q 长度均为 1、且存在可用 batch 捕获尺寸的输入使用 full Graph。其余 batch 若精确命中分段捕获的 token 尺寸，则使用 piecewise Graph；否则回退 eager。

这里的 full Graph 延续 nano-vLLM 的边界：捕获模型主干（包含 Attention），logits 投影和采样仍在图外执行。

分段 Graph 针对当前 Qwen3 模型手工按 Attention 边界切分：线性层、Norm、RoPE、MLP 等进入 Graph，Attention 使用当前请求元数据在段间执行。它不是 vLLM 通用编译系统的移植。

初次 BF16 对照曾出现最高 3.334% 的 logits 相对 L2，超过预设 3% 门槛。CPU 重放和隔离 GPU 对照定位到两类形状差异，随后采用以下默认规则：

- uniform 查询的 block table 元数据宽度与 full Graph 保持一致。补的是整数索引表，不额外预留 KV block；实际有效 KV 长度不变。
- piecewise 默认只复用精确 token 桶，避免 padding 改变矩阵计算形状。`cudagraph_allow_padding=True` 可显式启用补齐，但本轮性能配置不启用。

原失败结果与隔离实验保存在 [诊断结果](results/20261015/)，没有放宽门槛或覆盖失败记录。最终配置在预算 512、1024 下，以教师强制后续 token 对照同步 eager、同步 auto、异步 auto，每组 32 行完整 logits 均与冻结基线逐位一致；该结论只覆盖已测输入和配置。

额外通过真实 GPU 边界检查：255/257-token Prompt 跨 block 的同步/异步 7 行 logits 逐位一致；两次相同前缀复用的 2 行 logits 逐位一致。冷缓存与前缀命中会改变计算形状，该对照最大相对 L2 为 1.333%，不宣称逐位一致。2-block 调度池触发了实际抢占与重算；EOS、取消均覆盖两个真实在途 batch，排空后检查 KV 引用、输出占位和 GPU 反馈引用归零。[边界检查数据](results/20261016-final/boundaries.json)

最终回归测试 **93 项通过**，另有 16 项 subtest 通过。模型数值检查共 192 行对照，六种预算/执行配置均逐位一致。[预算 512 异步数值对照](results/20261016-final/comparison_v1_async_auto_b512.json) · [预算 1024 异步数值对照](results/20261016-final/comparison_v1_async_auto_b1024.json)

## 性能结果

2026-10-16，RTX 5080 / Qwen3-0.6B / BF16 / PyTorch 2.11.0+cu128 / FlashAttention 2.8.3.post1。两版均固定 64 个 KV block（每块 256 token）、最多 16 条请求。共 24 个独立性能进程、360 条正式观测；以下为各自正反序配对中六次观测的中位数，箭头表示冻结基线 → **新版异步 auto**。

混合负载先让 4 条请求进入 Decode，再注入 1 条或 4 条 3000-token Prompt；容量场景注入 9 条。纯 Prefill 使用 300、500、1000-token Prompt，每条只生成 1 个输出 token。同步 eager 和同步 auto 作为独立对照，所有结果保留。

| 预算 | 负载 | 输出吞吐 token/s | 吞吐变化 | 新请求 TTFT ms | 已有请求 TPOT ms | 已有请求 max gap 均值 ms |
|---|---|---:|---:|---:|---:|---:|
| 512 | 纯 Decode | 1397.83 → 1586.31 | +13.48% | — | 2.86 → 2.52 | 3.56 → 3.06 |
| 512 | 纯 Prefill | 42.64 → 84.36 | +97.87% | 40.47 → 21.20 | — | — |
| 512 | 1 条长 Prompt | 888.52 → 1133.59 | +27.58% | 104.32 → 67.13 | 4.89 → 3.83 | 20.42 → 21.78 |
| 512 | 4 条长 Prompt | 484.44 → 690.14 | +42.46% | 275.70 → 179.51 | 11.07 → 7.77 | 26.50 → 16.22 |
| 512 | 等待需求超过 KV 容量 | 322.45 → 456.66 | +41.62% | 547.47 → 372.71 | 18.14 → 12.65 | 29.83 → 14.42 |
| 1024 | 纯 Decode | 1395.42 → 1591.67 | +14.06% | — | 2.87 → 2.51 | 3.61 → 3.08 |
| 1024 | 纯 Prefill | 83.86 → 98.63 | +17.62% | 23.21 → 23.49 | — | — |
| 1024 | 1 条长 Prompt | 1034.44 → 1145.33 | +10.72% | 61.11 → 57.25 | 4.20 → 3.79 | 22.58 → 21.30 |
| 1024 | 4 条长 Prompt | 656.85 → 711.59 | +8.33% | 163.27 → 151.31 | 8.16 → 7.53 | 24.58 → 23.16 |
| 1024 | 等待需求超过 KV 容量 | 478.17 → 510.27 | +6.71% | 339.40 → 320.33 | 14.77 → 13.84 | 27.49 → 23.77 |

收益有边界：预算 512 的单长 Prompt 场景，max gap 均值增加 **6.69%**；预算 1024 的纯 Prefill 场景，TTFT 增加 **1.20%**。同步 auto 的纯 Decode 吞吐分别下降 **3.30% / 4.70%**，不能将结构改进等同于所有配置都更快。

<details>
<summary>查看同步对照：相对各自配对基线的输出吞吐变化</summary>

| 预算 | 负载 | 同步 eager | 同步 auto | 异步 auto |
|---|---|---:|---:|---:|
| 512 | 纯 Decode | −81.45% | −3.30% | +13.48% |
| 512 | 纯 Prefill | +9.04% | +47.17% | +97.87% |
| 512 | 1 条长 Prompt | −68.80% | +15.25% | +27.58% |
| 512 | 4 条长 Prompt | −37.86% | +26.06% | +42.46% |
| 512 | 等待需求超过 KV 容量 | −12.88% | +31.01% | +41.62% |
| 1024 | 纯 Decode | −81.57% | −4.70% | +14.06% |
| 1024 | 纯 Prefill | −3.59% | +3.20% | +17.62% |
| 1024 | 1 条长 Prompt | −74.75% | −0.23% | +10.72% |
| 1024 | 4 条长 Prompt | −54.16% | +2.06% | +8.33% |
| 1024 | 等待需求超过 KV 容量 | −24.15% | +0.87% | +6.71% |

同步 eager 同时关闭了 Graph，该列不能单独解释为调度策略的收益。各候选使用自己的配对基线，也不能直接相减得到某项优化的精确贡献。

</details>

全部 180 对观测的输出数量、Prefill/Decode 计算 token 数和 forward 次数一致。异步组实际执行 1632 次 full Graph replay、720 次分段 forward（20880 次 segment replay），GPU 直接反馈 11352 个输入 token，最大在途批次数为 2。

增量 KV 分配也有占用差异：预算 512 的四长 Prompt 和容量场景，在用 block 峰值由 52 降到同步模式的 47、异步模式的 49；预算 1024 的对应异步峰值仍为 52 / 64。异步会同时持有在途资源，不保证每种负载都减少峰值。在这些性能负载中未观察到额外 Prefill 重算；真正的抢占重算由独立边界测试验证。

`max gap 均值` 先对每条已有请求取最大相邻输出间隔，再求请求均值，包含跨越测量起点的间隔。TPOT 从测量开始前的首个输出起计算。表中结果仅代表本机这些负载的观测，不宣称统计显著或线上收益。

[完整测量汇总](results/20261016-final/v1_summary.json) · [原始指标与执行证据](results/20261016-final/performance.md) · [独立数据审计](results/20261016-final/audit.json) · [环境与源码哈希](results/20261016-final/v1_environment.json)

## 复现

在 Linux / WSL CUDA 环境准备 Qwen3-0.6B，使用现有兼容的 PyTorch、Triton、FlashAttention 环境：

```bash
pip install -e .
python -m pytest tests -q
python scripts/check_v1_boundaries.py --model /path/to/Qwen3-0.6B \
  --include-lifecycle --output results/boundaries.json
python scripts/compare_v1.py --action all \
  --model /path/to/Qwen3-0.6B \
  --budgets 512 1024 --kv-blocks 64 --reps 3 \
  --output-dir results/reproduction
```

脚本先验证模型数值，通过后才测性能。每个候选按 baseline → candidate → candidate → baseline 四个独立进程执行；每类 workload 在各进程内先预热，再测三轮。保存逐请求输出时间、原始指标、环境、源码哈希和实际 Graph replay 次数。`--resume` 仅允许源码与参数完全一致的结果续跑。

运行新实现时，可通过 `LLM(model, max_num_batched_tokens=1024, cuda_graph_mode="auto", async_scheduling=True)` 启用默认方案。设置 `async_scheduling=False` 使用单批次同步回收；`cuda_graph_mode="eager"` 关闭 Graph。两类开关可以独立选择。

基准覆盖纯 Decode、纯 Prefill、单长 Prompt 混合、四长 Prompt 混合和 KV 容量压力。TTFT、TPOT、token gap 均来自 `engine.step()` 接收结果的时间，不包含 HTTP 和网络延迟；吞吐只统计测量窗口内生成的输出 token。`kv_peak_blocks` 是峰值在用 block 数，KV tensor 本身仍在初始化时预分配，不能把它直接称为进程显存降幅。

## 范围

- 已实现和测量的异步路径为单 GPU，最多两个在途 batch；多卡异步配置会明确报错。多卡同步未做真实设备验证。
- 默认准入仍检查整段已知序列能否容纳，但实际只分配当前 chunk 所需 block；该检查不承诺提前预留未来所有生成 token。
- 不含 KV offload、滑动窗口、推测解码和完整 vLLM 功能；单请求超过配置上下文或 KV 总容量时拒绝。
- 完整 Graph 只覆盖支持的 uniform 查询形状；捕获不可用或尺寸未命中时使用 eager。

保留上游 MIT 许可证，见 [LICENSE](LICENSE)。
