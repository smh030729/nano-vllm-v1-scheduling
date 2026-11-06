import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    cuda_graph_mode: str = 'auto'
    cudagraph_token_sizes: tuple[int, ...] = (64, 128, 256, 512, 1024)
    cudagraph_allow_padding: bool = False
    max_cudagraph_tokens: int = 1024
    async_scheduling: bool = True
    max_inflight_batches: int = 2
    mixed_batching: bool = True
    incremental_kv_allocation: bool = True
    prefix_cache_lookahead: int = 0
    uncached_first: bool = False
    intrusive_free_queue: bool = False
    fused_qk_norm_rope: bool = False
    admission_full_sequence: bool = True
    long_prefill_token_threshold: int = 0
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        if self.cuda_graph_mode not in ('auto', 'eager', 'piecewise', 'full'):
            raise ValueError('invalid cuda_graph_mode')
        if not 1 <= self.max_inflight_batches <= 2:
            raise ValueError('max_inflight_batches must be 1 or 2')
        if not self.async_scheduling:
            self.max_inflight_batches = 1
        if self.async_scheduling and self.tensor_parallel_size != 1:
            raise ValueError('asynchronous execution currently requires tensor_parallel_size=1')
        if self.enforce_eager:
            self.cuda_graph_mode = 'eager'
        if self.max_cudagraph_tokens < 1 or self.long_prefill_token_threshold < 0:
            raise ValueError('invalid token threshold')
        if self.prefix_cache_lookahead < 0:
            raise ValueError('prefix_cache_lookahead must be nonnegative')
        self.cudagraph_token_sizes = tuple(sorted(set(
            n for n in self.cudagraph_token_sizes
            if 0 < n <= min(self.max_cudagraph_tokens, self.max_num_batched_tokens))))
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
