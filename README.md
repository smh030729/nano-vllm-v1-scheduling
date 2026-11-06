# Nano-vLLM 单卡推理引擎

基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的离线推理实现。一个 Python 包包含 Qwen3 模型、KV 分页管理、调度器、CUDA Graph 执行器与 CUDA 融合算子。三类优化可以在创建引擎时分别配置。见 [MIT 许可证](LICENSE)。

## 安装与生成

已验证环境为 Linux/WSL、Python 3.10、PyTorch 2.11.0+cu128、CUDA Toolkit 12.8、FlashAttention 2.8.3.post1、RTX 4070 Ti SUPER。准备兼容的 PyTorch、flash-attn 和完整的本地 Qwen3-0.6B 权重。首次启用融合时，PyTorch 用 nvcc、C++ 编译器和 ninja 构建扩展。以下命令在本仓库根目录运行：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install .
export MODEL=/path/to/Qwen3-0.6B
```

```python
import os
from nanovllm import LLM, SamplingParams

llm = LLM(os.environ["MODEL"], max_num_seqs=4, max_model_len=512,
          max_num_batched_tokens=128, num_kvcache_blocks=16,
          async_scheduling=True, cuda_graph_mode="auto",
          prefix_cache_lookahead=16, uncached_first=True,
          intrusive_free_queue=True, fused_qk_norm_rope=True)
try:
    result = llm.generate(["Explain the KV cache in one sentence."],
                          SamplingParams(max_tokens=32))
    print(result[0]["text"])
finally:
    llm.exit()
```

独立性检查使用仅含源码、打包配置、README 与许可证的临时目录构建 wheel，装入独立导入目录，复用预装的 CUDA/Python 依赖。从安装包重新编译融合扩展，三项全开，两个真实请求各生成 8 个 token，观测最大在途深度为 2。运行引擎需要模型权重；历史对照实验的其他源码目录只在指定对照组中使用。

## 三项改动

| 问题 | 处理方式 | 实现入口与开关 |
|---|---|---|
| 长 Prompt 连续占用 Prefill，生成请求停顿；同步回收让 GPU 等待下一批输入 | 按剩余 token 和本轮预算混排 Prefill/Decode，按进度分配 KV。用不可变批次快照区分已安排、在途及已确认状态，最多两批在途。前一批采样 token 直接在 GPU 上填入下一批。auto 按形状选 full/piecewise Graph 或 eager | [调度器](src/nanovllm/engine/scheduler.py)、[提交与回收](src/nanovllm/engine/llm_engine.py)、[GPU 输入与执行](src/nanovllm/engine/model_runner.py)、[Graph 分发](src/nanovllm/engine/graph_runner.py)。mixed_batching、incremental_kv_allocation、async_scheduling 默认开启，max_inflight_batches=2 |
| 内存紧张时空闲块可被覆写，后续请求失去可复用的文档前缀 | 有界扫描等待队列，软保留已确认的完整前缀块。优先取没有有效缓存的空闲块，必要时回退以保持请求推进。侵入式链表将**已知块摘除**降至 O(1)，候选扫描仍可能线性 | [KV 块管理](src/nanovllm/engine/block_manager.py)、[请求队列](src/nanovllm/engine/scheduler.py)。prefix_cache_lookahead 默认 0，uncached_first 与 intrusive_free_queue 默认关闭 |
| Q/K 分别执行 RMSNorm 和 RoPE，产生多次 kernel 启动与中间读写 | 一个 warp 处理一个 token 的一个 Q/K head，直接读取带行间距的 packed QKV，在 FP32 下完成归约和旋转，V 延用原 view。注册 PyTorch CUDA/Meta 实现，接入 Prefill、Decode 和 Graph | [CUDA kernel](src/nanovllm/ops/csrc/qk_norm_rope.cu)、[dispatcher 注册](src/nanovllm/ops/csrc/qk_norm_rope.cpp)、[Qwen3 接入](src/nanovllm/models/qwen3.py)。fused_qk_norm_rope 默认关闭 |

## Benchmark

单张 RTX 4070 Ti SUPER，Graphics/SM 锁定 2400 MHz，Qwen3-0.6B BF16。统计本地推理窗口的**输出 token** 与接收时间，模型加载、编译、Graph capture 和网络均在计时范围外。以下三组实验分别改变不同的因素。

### 调度与异步执行

原始上游 → 统一调度 + 异步 auto Graph。固定 64 个 KV 块、每块 256 token。已有 4 条请求进入 Decode 后注入 3000-token 长 Prompt，容量压力场景注入 9 条。最大间隔先按已有请求各取一次最大相邻输出间隔，再对请求求均值。ABBA 独立进程预热后取中位数。

| Budget | 场景 | 吞吐 token/s | 变化 | 已有请求 TPOT ms | 最大 token 间隔均值 ms |
|---:|---|---:|---:|---:|---:|
| 512 | 纯 Decode | 1199.85 → 1301.63 | +8.48% | 3.33 → 3.07 | 3.96 → 3.68 |
| 512 | 纯 Prefill | 33.74 → 75.01 | +122.29% | — | — |
| 512 | 1 条长 Prompt | 654.79 → 917.29 | +40.09% | 6.63 → 4.73 | 133.14 → 22.30 |
| 512 | 4 条长 Prompt | 363.36 → 600.01 | +65.13% | 14.78 → 8.94 | 507.53 → 21.21 |
| 512 | KV 容量压力 | 238.02 → 389.65 | +63.70% | 29.75 → 14.66 | 661.50 → 17.65 |
| 1024 | 纯 Decode | 1142.50 → 1303.72 | +14.11% | 3.51 → 3.07 | 4.17 → 3.80 |
| 1024 | 纯 Prefill | 69.89 → 87.07 | +24.58% | — | — |
| 1024 | 1 条长 Prompt | 784.11 → 906.96 | +15.67% | 5.55 → 4.79 | 76.46 → 24.64 |
| 1024 | 4 条长 Prompt | 534.03 → 608.41 | +13.93% | 10.04 → 8.81 | 285.90 → 25.48 |
| 1024 | KV 容量压力 | 389.81 → 445.29 | +14.23% | 18.12 → 15.86 | 351.54 → 27.63 |

在 budget=512、4 条长 Prompt 的场景，输出吞吐提升 **65.13%**，最大间隔均值下降 **95.82%**。若以已支持混合 batch 的旧版为对照，同一场景的吞吐从 422.24 提升至 599.95 token/s（+42.09%）。两套基线衡量不同的问题。

### KV 前缀缓存

8 轮 × 8 请求；每个 Prompt 含 4096-token 文档与 64-token 独立后缀，输出 32 token。共 68 个 KV 块，预算 8192。表中“仅开启缓存”以同一整合引擎关闭缓存策略为基线。

| 对照 | 输出吞吐 token/s | 吞吐变化 | Prefill token 数 | Prefill 变化 | P95 TTFT ms | TTFT 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 原始 → 独立缓存方案 | 346.33 → 386.26 | +11.53% | 140032 → 107008 | -23.58% | 649.51 → 625.16 | -3.75% |
| 整合版：仅开启缓存 | 299.29 → 323.40 | +8.05% | 176128 → 152320 | -13.52% | 735.48 → 736.44 | +0.13% |
| 整合版：缓存与融合同时开启 | 299.29 → 328.38 | +9.72% | 176128 → 152320 | -13.52% | 735.48 → 724.12 | -1.54% |
| 独立前缀控制：仅开启缓存 | 232.26 → 232.23 | -0.01% | 266240 → 266240 | +0.00% | 912.21 → 912.08 | -0.01% |
| 独立前缀控制：缓存与融合同时开启 | 232.26 → 236.05 | +1.63% | 266240 → 266240 | +0.00% | 912.21 → 896.02 | -1.77% |

整合版缓存开关的收益为吞吐 **+8.05%**、Prefill 计算 **−13.52%**。前缀互不相同时计算量持平。独立缓存实现有不同的准入和分配时机，性能单列。

### CUDA Q/K RMSNorm + RoPE

16 个 Q heads、8 个 KV heads、head_dim=128、BF16。算子使用 CUDA Graph replay 与 CUDA Event 计时；32-token 用例另用 profiler 确认启动次数从 4 次减至 1 次。

| Token 数 | 未融合 μs | CUDA 融合 μs | 延迟降低 |
|---:|---:|---:|---:|
| 1 | 4.028 | 1.601 | 60.25% |
| 4 | 4.743 | 1.654 | 65.12% |
| 16 | 5.093 | 1.703 | 66.56% |
| 32 | 5.398 | 1.794 | 66.77% |
| 128 | 8.318 | 2.841 | 65.84% |
| 512 | 19.119 | 7.687 | 59.80% |

端到端对照在相同的异步 auto 引擎上只切换融合开关，关闭队列感知缓存。每个场景使用十对独立进程，表中延迟为整批请求完成时间。

| Batch / 输入 / 输出 | 未融合 ms | 融合 ms | 延迟降低 | 配对降低中位数 | 配对 bootstrap 95% 区间 |
|---|---:|---:|---:|---:|---:|
| 1 / 32 / 128 | 365.429 | 352.139 | 3.64% | 3.95% | [3.21%, 4.19%] |
| 4 / 32 / 128 | 397.555 | 388.195 | 2.35% | 2.48% | [2.03%, 3.10%] |
| 16 / 32 / 128 | 430.941 | 421.702 | 2.14% | 2.56% | [0.93%, 2.93%] |
| 1 / 1024 / 32 | 108.168 | 106.529 | 1.52% | 1.85% | [1.41%, 2.68%] |
| 4 / 1024 / 32 | 209.667 | 201.234 | 4.02% | 3.95% | [3.75%, 4.40%] |
| 1 / 3000 / 1 | 68.730 | 67.456 | 1.85% | 1.94% | [1.63%, 2.28%] |

## 正确性与复现

固定后续 token 比较 BF16 完整 logits。仅改变融合开关，最大相对 L2 为 2.77%，满足实验预设的 3% 门槛；随机采样输出可能分歧。生命周期测试覆盖 partial prefill、抢占、EOS、取消和在途 KV 引用回收。历史数值与执行边界详见[完整复测报告](results/resume-reproduction-20261104/locked2400/report.md)、[原始汇总](results/resume-reproduction-20261104/locked2400/formal/summary.json)及[调度设计与验证记录](DESIGN.md)。

以下命令仅使用本目录及本地模型：

```bash
PYTHONPATH=src python scripts/check_integrated.py --model "$MODEL" --include-cuda-fusion --output /tmp/nano-checks.json
python scripts/benchmark_v1.py --model "$MODEL" --source-root src --mode v1_async_auto --action benchmark --budget 512 --output /tmp/nano-scheduler.json
for arm in cache_off cache_on; do
  python scripts/reproduce_resume.py --model "$MODEL" --stage cache --arm "$arm" --output "/tmp/nano-$arm.json"
done
for arm in fusion_off fusion_on; do
  python scripts/reproduce_resume.py --model "$MODEL" --stage fusion --arm "$arm" --output "/tmp/nano-$arm.json"
done
python scripts/reproduce_resume.py --model "$MODEL" --stage micro --output /tmp/nano-micro.json
```

历史全矩阵脚本另需原始 upstream 和独立 cache 两份对照源码。脚本按相对路径从本目录的 `../nano-vllm-inference-optimization/src/nano-vllm-upstream` 和 `../../nano-vllm-queue-aware-cache/src` 加载；冻结的 mixed baseline 在本仓库的 `baseline/`。对照数据及频率记录见[环境清单](results/resume-reproduction-20261104/locked2400/formal/environment.json)。
