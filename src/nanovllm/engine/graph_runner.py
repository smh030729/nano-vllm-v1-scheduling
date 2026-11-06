from bisect import bisect_left
from types import SimpleNamespace

import torch


def _capture_sizes(token_sizes):
    sizes = tuple(sorted(set(token_sizes)))
    if any(not isinstance(size, int) or size <= 0 for size in sizes):
        raise ValueError('capture sizes must be positive integers')
    return sizes


def _is_recoverable_capture_error(error):
    message = str(error).lower()
    if 'illegal memory' in message or 'device-side assert' in message:
        return False
    return any(word in message for word in ('captur', 'graph', 'out of memory'))


class GraphDispatcher:
    MODES = {'auto', 'eager', 'piecewise', 'full'}

    def __init__(self, mode='auto', max_tokens=512,
                 token_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512), allow_padding=False):
        if mode not in self.MODES:
            raise ValueError(f'unsupported graph mode: {mode}')
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError('max_tokens must be a positive integer')
        self.mode = mode
        self.max_tokens = max_tokens
        self.allow_padding = allow_padding
        self.token_sizes = tuple(size for size in _capture_sizes(token_sizes) if size <= max_tokens)

    def token_bucket(self, num_tokens):
        if num_tokens <= 0 or num_tokens > self.max_tokens:
            return None
        index = bisect_left(self.token_sizes, num_tokens)
        return self.token_sizes[index] if index < len(self.token_sizes) else None

    def select(self, num_tokens, query_lens, full_available):
        if num_tokens <= 0 or not query_lens or any(length <= 0 for length in query_lens):
            raise ValueError('a batch must contain positive query lengths')
        if sum(query_lens) != num_tokens:
            raise ValueError('query lengths do not match the packed token count')
        if self.mode == 'eager' or num_tokens > self.max_tokens:
            return 'eager'
        uniform = all(length == 1 for length in query_lens)
        if self.mode in ('auto', 'full') and uniform and full_available:
            return 'full'
        if self.mode == 'full':
            return 'eager'
        bucket = self.token_bucket(num_tokens)
        return 'piecewise' if bucket is not None and (bucket == num_tokens or self.allow_padding) else 'eager'


def copy_padded_inputs(input_ids, positions, static_input_ids, static_positions):
    if (input_ids.ndim != 1 or positions.ndim != 1
            or static_input_ids.ndim != 1 or static_positions.ndim != 1
            or input_ids.numel() != positions.numel()
            or static_input_ids.numel() != static_positions.numel()
            or not 0 < input_ids.numel() <= static_input_ids.numel()):
        raise ValueError('input and position vectors must fit the same capture bucket')
    num_tokens = input_ids.numel()
    static_input_ids[:num_tokens].copy_(input_ids)
    static_positions[:num_tokens].copy_(positions)
    static_input_ids[num_tokens:].zero_()
    static_positions[num_tokens:].zero_()
    return num_tokens


class PiecewiseGraphRunner:
    """按 Attention 边界捕获 Qwen3；同一实例的所有调用必须使用同一执行流。"""

    def __init__(self, model, token_sizes):
        self.model = model
        self.backbone = model.model
        self.layers = self.backbone.layers
        self.token_sizes = _capture_sizes(token_sizes)
        if not self.layers:
            raise ValueError('piecewise execution requires at least one decoder layer')
        self.buckets = {}
        self.pool = None
        self.capture_errors = {}
        self.stats = {'captures': 0, 'forward_replays': 0, 'segment_replays': 0}

    def _create_buffers(self, size):
        parameter = next(self.model.parameters())
        device, dtype = parameter.device, parameter.dtype
        attention = self.layers[0].self_attn
        hidden_size = self.backbone.embed_tokens.weight.shape[1]
        return SimpleNamespace(
            input_ids=torch.zeros(size, dtype=torch.int64, device=device),
            positions=torch.zeros(size, dtype=torch.int64, device=device),
            q=torch.zeros(size, attention.num_heads, attention.head_dim, dtype=dtype, device=device),
            k=torch.zeros(size, attention.num_kv_heads, attention.head_dim, dtype=dtype, device=device),
            v=torch.zeros(size, attention.num_kv_heads, attention.head_dim, dtype=dtype, device=device),
            attention_out=torch.zeros(size, attention.num_heads, attention.head_dim, dtype=dtype, device=device),
            residual=torch.zeros(size, hidden_size, dtype=dtype, device=device),
            hidden=torch.zeros(size, hidden_size, dtype=dtype, device=device),
            graphs=[],
        )

    @staticmethod
    def _write_qkv(attention, hidden, state):
        qkv = attention.qkv_proj(hidden)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
        q = q.view(-1, attention.num_heads, attention.head_dim)
        k = k.view(-1, attention.num_kv_heads, attention.head_dim)
        v = v.view(-1, attention.num_kv_heads, attention.head_dim)
        if getattr(attention, 'fused_qk_norm_rope', False) and not attention.qkv_bias:
            q, k = torch.ops.nanovllm.fused_qk_norm_rope(
                qkv, state.positions, attention.q_norm.weight, attention.k_norm.weight,
                attention.rotary_emb.cos_sin_cache, attention.num_heads,
                attention.num_kv_heads, attention.q_norm.eps, attention.qk_norm_round_normalized)
        else:
            if not attention.qkv_bias:
                q, k = attention.q_norm(q), attention.k_norm(k)
            q, k = attention.rotary_emb(state.positions, q, k)
        state.q.copy_(q)
        state.k.copy_(k)
        state.v.copy_(v)

    @staticmethod
    def _finish_layer(layer, state):
        hidden = layer.self_attn.o_proj(state.attention_out.flatten(1))
        hidden, residual = layer.post_attention_layernorm(hidden, state.residual)
        return layer.mlp(hidden), residual

    def _make_segments(self, state):
        def first():
            residual = self.backbone.embed_tokens(state.input_ids)
            hidden = self.layers[0].input_layernorm(residual)
            self._write_qkv(self.layers[0].self_attn, hidden, state)
            state.residual.copy_(residual)

        segments = [first]
        for index in range(1, len(self.layers)):
            def middle(index=index):
                hidden, residual = self._finish_layer(self.layers[index - 1], state)
                hidden, residual = self.layers[index].input_layernorm(hidden, residual)
                self._write_qkv(self.layers[index].self_attn, hidden, state)
                state.residual.copy_(residual)
            segments.append(middle)

        def last():
            hidden, residual = self._finish_layer(self.layers[-1], state)
            hidden, _ = self.backbone.norm(hidden, residual)
            state.hidden.copy_(hidden)

        segments.append(last)
        return segments

    @torch.inference_mode()
    def capture(self):
        if not self.token_sizes:
            return
        if next(self.model.parameters()).device.type != 'cuda':
            raise ValueError('CUDA Graph capture requires a CUDA model')
        if getattr(self.backbone.embed_tokens, 'tp_size', 1) != 1:
            self.capture_errors['backend'] = 'piecewise capture currently supports tensor_parallel_size=1'
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for size in reversed(self.token_sizes):
            if size in self.buckets:
                continue
            state = None
            try:
                state = self._create_buffers(size)
                segments = self._make_segments(state)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        for segment in segments:
                            segment()
                stream.synchronize()
                for segment in segments:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                        segment()
                    if self.pool is None:
                        self.pool = graph.pool()
                    state.graphs.append(graph)
                stream.synchronize()
                self.buckets[size] = state
                self.stats['captures'] += len(state.graphs)
            except RuntimeError as error:
                if not _is_recoverable_capture_error(error):
                    raise
                self.capture_errors[size] = f'{type(error).__name__}: {error}'
                if state is not None:
                    state.graphs.clear()
        torch.cuda.current_stream().wait_stream(stream)

    def token_bucket(self, num_tokens):
        return next((size for size in sorted(self.buckets) if size >= num_tokens), None) if num_tokens > 0 else None

    def supports(self, num_tokens):
        return self.token_bucket(num_tokens) is not None

    @torch.inference_mode()
    def run(self, input_ids, positions):
        num_tokens = input_ids.numel()
        bucket = self.token_bucket(num_tokens)
        if bucket is None:
            raise ValueError(f'no captured piecewise bucket can hold {num_tokens} tokens')
        state = self.buckets[bucket]
        copy_padded_inputs(input_ids, positions, state.input_ids, state.positions)
        state.attention_out[num_tokens:].zero_()
        state.graphs[0].replay()
        for index, layer in enumerate(self.layers):
            output = layer.self_attn.attn(state.q[:num_tokens], state.k[:num_tokens], state.v[:num_tokens])
            state.attention_out[:num_tokens].copy_(output.reshape_as(state.attention_out[:num_tokens]))
            state.graphs[index + 1].replay()
        self.stats['forward_replays'] += 1
        self.stats['segment_replays'] += len(state.graphs)
        return state.hidden[:num_tokens]
