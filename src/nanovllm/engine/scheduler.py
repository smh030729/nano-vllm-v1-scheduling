from collections import deque
from itertools import islice

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.schedule_output import ScheduledSequence, ScheduleOutput


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_inflight_batches = getattr(config, 'max_inflight_batches', 1)
        self.admission_full_sequence = getattr(config, 'admission_full_sequence', True)
        self.max_model_len = getattr(config, 'max_model_len', 4096)
        self.long_prefill_token_threshold = getattr(config, 'long_prefill_token_threshold', 0)
        self.mixed_batching = getattr(config, 'mixed_batching', True)
        self.incremental_kv_allocation = getattr(config, 'incremental_kv_allocation', True)
        self.prefix_cache_lookahead = getattr(config, 'prefix_cache_lookahead', 0)
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, self.block_size,
            uncached_first=getattr(config, 'uncached_first', False),
            intrusive_free_queue=getattr(config, 'intrusive_free_queue', False))
        self.waiting = deque()
        self.running = deque()
        self.requests = {}
        self._inflight = {}
        self._next_batch_id = 0
        if min(self.max_num_seqs, self.max_num_batched_tokens, self.max_inflight_batches) <= 0:
            raise ValueError('budgets and in-flight batch capacity must be positive')
        if self.long_prefill_token_threshold < 0:
            raise ValueError('long_prefill_token_threshold must be nonnegative')
        if self.prefix_cache_lookahead < 0:
            raise ValueError('prefix_cache_lookahead must be nonnegative')

    def is_finished(self):
        return not self.waiting and not self.running and not self._inflight

    @property
    def num_inflight_batches(self):
        return len(self._inflight)

    def add(self, seq: Sequence):
        if seq.seq_id in self.requests:
            raise ValueError('request already exists')
        if len(seq) >= self.max_model_len:
            raise ValueError('prompt must leave room for an output within max_model_len')
        if seq.num_blocks > len(self.block_manager.blocks):
            raise ValueError('prompt exceeds the total KV cache capacity')
        if seq.max_tokens <= 0:
            raise ValueError('max_tokens must be positive')
        seq.max_tokens = min(seq.max_tokens, self.max_model_len - seq.num_prompt_tokens)
        self.requests[seq.seq_id] = seq
        self.waiting.append(seq)

    def _empty(self):
        return ScheduleOutput(-1, ())

    def _eviction_lookahead(self):
        return islice(self.waiting, self.prefix_cache_lookahead) if self.prefix_cache_lookahead else None

    def _allocation_end(self, seq, end):
        return end if self.incremental_kv_allocation else max(end, len(seq))

    def _needed_tokens(self, seq):
        if seq.num_completion_tokens + seq.num_output_placeholders >= seq.max_tokens:
            return 0
        return len(seq) + seq.num_output_placeholders - seq.num_computed_tokens

    def _cap_prefill_tokens(self, seq, count):
        if seq.is_prefill and self.long_prefill_token_threshold:
            return min(count, self.long_prefill_token_threshold)
        return count

    def _append_snapshot(self, seq, count, items):
        start = seq.num_computed_tokens
        end = start + count
        target = len(seq) + seq.num_output_placeholders
        produces_output = end == target
        inputs = tuple(seq.token_ids[pos] if pos < len(seq) else -1 for pos in range(start, end))
        item = ScheduledSequence(seq.seq_id, seq.generation, start, count, inputs,
                                 tuple(seq.block_table), seq.is_prefill, produces_output,
                                 seq.temperature, seq.num_prompt_tokens)
        seq.num_computed_tokens = end
        seq.num_in_flight_tokens += count
        if produces_output:
            seq.num_output_placeholders += 1
            seq.is_prefill = False
        items.append(item)

    def schedule(self):
        if len(self._inflight) >= self.max_inflight_batches:
            return self._empty()
        remaining = self.max_num_batched_tokens
        items = []
        preempted = False
        for seq in tuple(self.running):
            if remaining == 0 or len(items) == self.max_num_seqs:
                break
            if seq.status != SequenceStatus.RUNNING:
                continue
            if items and not self.mixed_batching and seq.is_prefill != items[0].is_prefill:
                continue
            count = self._cap_prefill_tokens(seq, min(self._needed_tokens(seq), remaining))
            if count <= 0:
                continue
            end = seq.num_computed_tokens + count
            allocation_end = self._allocation_end(seq, end)
            while not self.block_manager.can_allocate_slots(seq, allocation_end):
                order = list(self.running)
                index = order.index(seq)
                victim = next((s for s in reversed(order[index+1:])
                               if not s.num_in_flight_tokens), None)
                if victim is not None:
                    self.preempt(victim)
                    preempted = True
                    continue
                if not seq.num_in_flight_tokens:
                    if (allocation_end + self.block_size - 1) // self.block_size > len(self.block_manager.blocks):
                        raise RuntimeError('request context has exhausted the total KV cache capacity')
                    self.preempt(seq)
                    preempted = True
                break
            else:
                self.block_manager.allocate_slots(seq, allocation_end, waiting_seqs=self._eviction_lookahead())
                self._append_snapshot(seq, count, items)
                remaining -= count

        while (remaining and self.waiting and len(self.running) < self.max_num_seqs and not preempted
               and (self.mixed_batching or not items or items[0].is_prefill)):
            seq = self.waiting[0]
            cached = self.block_manager.cached_prefix(seq)
            start = len(cached) * self.block_size
            count = self._cap_prefill_tokens(seq, min(len(seq) - start, remaining))
            end = start + count
            allocation_end = self._allocation_end(seq, end)
            full_tokens = len(seq) if self.admission_full_sequence else None
            if not self.block_manager.can_allocate_slots(seq, allocation_end, cached, full_tokens):
                break
            self.block_manager.allocate_slots(seq, allocation_end, cached, self._eviction_lookahead())
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            self._append_snapshot(seq, count, items)
            remaining -= count

        if not items:
            return self._empty()
        output = ScheduleOutput(self._next_batch_id, tuple(items))
        self._next_batch_id += 1
        self._inflight[output.batch_id] = output
        return output

    def preempt(self, seq):
        if seq.num_in_flight_tokens or seq.num_output_placeholders:
            raise RuntimeError('cannot preempt a request with in-flight work')
        if seq.status != SequenceStatus.RUNNING:
            raise ValueError('only running requests may be preempted')
        self.running.remove(seq)
        self.block_manager.deallocate(seq)
        seq.generation += 1
        seq.is_prefill = True
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

    def _finish(self, seq):
        seq.status = SequenceStatus.FINISHED
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        if not seq.num_in_flight_tokens:
            self.block_manager.deallocate(seq)
            del self.requests[seq.seq_id]

    def abort(self, seq_id):
        seq = self.requests.get(seq_id)
        if seq is None or seq.is_finished:
            return False
        self._finish(seq)
        return True

    def postprocess(self, scheduled: ScheduleOutput, token_ids: list[int]):
        if not self._inflight or next(iter(self._inflight)) != scheduled.batch_id:
            raise ValueError('batch results must be retired once in submission order')
        if self._inflight[scheduled.batch_id] != scheduled:
            raise ValueError('result does not match the submitted batch snapshot')
        if len(scheduled.items) != len(token_ids):
            raise ValueError('one sampled token is required per scheduled item')
        accepted = []
        for item, token_id in zip(scheduled.items, token_ids):
            seq = self.requests[item.seq_id]
            if seq.generation != item.generation:
                raise RuntimeError('stale generation returned while request state was reused')
            seq.num_in_flight_tokens -= item.num_scheduled_tokens
            if item.produces_output:
                seq.num_output_placeholders -= 1
            if seq.is_finished:
                if not seq.num_in_flight_tokens:
                    self.block_manager.deallocate(seq)
                    del self.requests[seq.seq_id]
                continue
            if seq.num_cached_tokens != item.start_pos:
                raise RuntimeError('confirmed KV progress does not match the returned input span')
            self.block_manager.hash_blocks(seq, item.start_pos, item.end_pos)
            seq.num_cached_tokens = item.end_pos
            if item.produces_output:
                seq.append_token(int(token_id))
                accepted.append((seq.seq_id, int(token_id)))
                if ((not seq.ignore_eos and token_id == self.eos)
                        or seq.num_completion_tokens >= seq.max_tokens
                        or len(seq) >= self.max_model_len):
                    self._finish(seq)
        del self._inflight[scheduled.batch_id]
        return accepted
