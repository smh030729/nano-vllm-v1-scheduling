import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.batch_input import build_batch_metadata
from nanovllm.engine.execution import ExecutionTicket
from nanovllm.engine.graph_runner import GraphDispatcher, PiecewiseGraphRunner
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self._warming_up = True
        self._last_sampled = {}
        self.logits_observer = None
        self.token_override = None
        self.runtime_stats = dict(forward_count=0, eager_count=0, full_graph_replays=0,
            piecewise_graph_replays=0, piecewise_segment_replays=0, mixed_forward_count=0,
            gpu_feedback_tokens=0)

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        if config.fused_qk_norm_rope:
            from torch._inductor import config as inductor_config
            from nanovllm.ops import load_fused_qk_norm_rope
            load_fused_qk_norm_rope()
            for layer in self.model.model.layers:
                layer.self_attn.fused_qk_norm_rope = True
                layer.self_attn.qk_norm_round_normalized = inductor_config.emulate_precision_casts
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        self.graphs = {}
        self.graph_pool = None
        self.full_capture_errors = []
        if config.cuda_graph_mode in ('auto', 'full'):
            try:
                self.capture_cudagraph()
            except RuntimeError as error:
                message = str(error).lower()
                if ('illegal memory' in message or 'device-side assert' in message
                        or not any(word in message for word in ('capture', 'graph', 'out of memory'))):
                    raise
                reset_context()
                torch.cuda.synchronize()
                self.full_capture_errors.append(str(error))
                self.graphs.clear()
                self.graph_pool = None
        self.piecewise = PiecewiseGraphRunner(self.model, config.cudagraph_token_sizes)
        if config.cuda_graph_mode in ('auto', 'piecewise'):
            self.piecewise.capture()
        self.graph_dispatcher = GraphDispatcher(config.cuda_graph_mode,
            min(config.max_cudagraph_tokens, config.max_num_batched_tokens),
            self.piecewise.buckets, allow_padding=config.cudagraph_allow_padding)
        self.copy_stream = torch.cuda.Stream()
        self._warming_up = False
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        torch.cuda.synchronize()
        self._last_sampled.clear()
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        self.graphs.clear()
        self.piecewise = None
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        # Only rank zero owns sampled-token feedback. Broadcasting this local
        # cleanup would race the next SHM command because it has no collective.
        if self.world_size > 1 and self.rank == 0 and method_name != 'forget':
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        available_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = available_blocks
        elif config.num_kvcache_blocks > available_blocks:
            raise ValueError(f'requested {config.num_kvcache_blocks} KV blocks exceed memory budget {available_blocks}')
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            if seq.is_prefill:
                input_ids.extend(seq[start:end])
            else:
                assert seqlen_q == 1 and end == len(seq)
                input_ids.append(seq.last_token)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if getattr(self, '_warming_up', False) or is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].fill_(1)
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def forget(self, keys):
        # A preempted request can finish before its new generation is submitted.
        seq_ids = {key[0] for key in keys}
        for key in tuple(self._last_sampled):
            if key[0] in seq_ids:
                del self._last_sampled[key]

    @torch.inference_mode()
    def submit(self, scheduled):
        items = scheduled.items
        for item in items:
            for key in tuple(self._last_sampled):
                if key[0] == item.seq_id and key[1] != item.generation:
                    del self._last_sampled[key]
        data = build_batch_metadata(items, self.block_size,
            decode_table_width=(self.config.max_model_len + self.block_size - 1) // self.block_size)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        host_buffers = []
        def upload(value, dtype):
            host = torch.tensor(value, dtype=dtype, device='cpu', pin_memory=True)
            host_buffers.append(host)
            return host.to(device=self.rank, non_blocking=True)

        input_ids = upload(data['input_ids'], torch.int64)
        positions = upload(data['positions'], torch.int64)
        if data['feedback']:
            indices, values = [], []
            for index, key, position in data['feedback']:
                previous = self._last_sampled.get(key)
                if previous is None or previous[0] != position:
                    raise RuntimeError(f'missing GPU token dependency for request {key} at {position}')
                indices.append(index)
                values.append(previous[1].reshape(1))
            input_ids.index_copy_(0, upload(indices, torch.int64), torch.cat(values))
            self.runtime_stats['gpu_feedback_tokens'] += len(indices)

        uniform = all(n == 1 for n in data['query_lens'])
        slot_mapping = upload(data['slot_mapping'], torch.int32)
        block_tables = upload(data['block_tables'], torch.int32)
        if uniform:
            set_context(False, slot_mapping=slot_mapping, block_tables=block_tables,
                        context_lens=upload(data['context_lens'], torch.int32))
        else:
            set_context(True, cu_seqlens_q=upload(data['cu_seqlens_q'], torch.int32),
                cu_seqlens_k=upload(data['cu_seqlens_k'], torch.int32),
                max_seqlen_q=max(data['query_lens']), max_seqlen_k=max(data['context_lens']),
                slot_mapping=slot_mapping, block_tables=block_tables)
        context = get_context()
        full_available = uniform and bool(self.graphs) and len(items) <= max(self.graphs)
        mode = self.graph_dispatcher.select(len(data['input_ids']), data['query_lens'], full_available)
        try:
            if mode == 'full':
                logits = self.run_model(input_ids, positions, False)
                self.runtime_stats['full_graph_replays'] += 1
            elif mode == 'piecewise':
                logits = self.model.compute_logits(self.piecewise.run(input_ids, positions))
                self.runtime_stats['piecewise_graph_replays'] += 1
                self.runtime_stats['piecewise_segment_replays'] += len(self.model.model.layers) + 1
            else:
                logits = self.model.compute_logits(self.model(input_ids, positions))
                self.runtime_stats['eager_count'] += 1
            self.runtime_stats['forward_count'] += 1
            self.runtime_stats['mixed_forward_count'] += bool(scheduled.num_prefill_tokens and scheduled.num_decode_tokens)
            if self.rank != 0:
                return None
            if self.logits_observer is not None:
                self.logits_observer(items, logits)
            temperatures = upload([item.temperature for item in items], torch.float32)
            sampled = self.sampler(logits, temperatures)
            if self.token_override is not None:
                sampled = self.token_override(items, sampled)
                if sampled.shape != (len(items),) or sampled.device != input_ids.device:
                    raise ValueError('token_override must return one GPU token per scheduled request')
            for row, item in enumerate(items):
                if item.produces_output:
                    self._last_sampled[(item.seq_id, item.generation)] = (
                        item.start_pos + item.num_scheduled_tokens, sampled[row])
            end_event.record()
            cpu_tokens = torch.empty(sampled.shape, dtype=sampled.dtype, device='cpu', pin_memory=True)
            completion = torch.cuda.Event()
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(end_event)
                cpu_tokens.copy_(sampled, non_blocking=True)
                sampled.record_stream(self.copy_stream)
                completion.record()
            return ExecutionTicket(scheduled.batch_id, cpu_tokens, completion, start_event, end_event,
                (sampled, input_ids, positions, context, tuple(host_buffers)))
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        context_lens = torch.ones(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = sorted({n for n in [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) + [max_bs] if n <= max_bs})
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
