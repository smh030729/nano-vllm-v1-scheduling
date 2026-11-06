import sys
import unittest
import pickle
import random
from collections import Counter, deque
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class SchedulerTests(unittest.TestCase):
    def make(self, budget=4, seqs=8, blocks=32, inflight=2, full=True, model_len=128, threshold=0):
        Sequence.block_size = 4
        return Scheduler(SimpleNamespace(max_num_seqs=seqs, max_num_batched_tokens=budget,
            eos=99999, num_kvcache_blocks=blocks, kvcache_block_size=4,
            max_inflight_batches=inflight, admission_full_sequence=full, max_model_len=model_len,
            long_prefill_token_threshold=threshold))

    def seq(self, n, offset=10, max_tokens=4, ignore=True):
        return Sequence(list(range(offset, offset+n)),
                        SamplingParams(max_tokens=max_tokens, ignore_eos=ignore))

    def test_partial_prefill_enters_running_at_admission(self):
        sch = self.make()
        seq = self.seq(12)
        sch.add(seq)
        sch.schedule()
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertEqual(list(sch.running), [seq])
        self.assertFalse(sch.waiting)
        self.assertEqual(len(seq.block_table), 1)

    def finish(self, sch, batch, token=123):
        accepted = sch.postprocess(batch, [token] * len(batch.items))
        self.invariants(sch)
        return accepted

    def invariants(self, sch):
        live = [*sch.running, *sch.waiting]
        self.assertEqual(len(live), len({s.seq_id for s in live}))
        refs = Counter(b for s in sch.requests.values() for b in s.block_table)
        bm = sch.block_manager
        self.assertEqual(set(bm.free_block_ids) | bm.used_block_ids, set(range(len(bm.blocks))))
        self.assertFalse(set(bm.free_block_ids) & bm.used_block_ids)
        self.assertEqual(len(bm.free_block_ids), len(set(bm.free_block_ids)))
        for block in bm.blocks:
            self.assertEqual(block.ref_count, refs[block.block_id])
        for seq in sch.requests.values():
            self.assertGreaterEqual(seq.num_output_placeholders, 0)
            self.assertGreaterEqual(seq.num_in_flight_tokens, 0)
            self.assertLessEqual(seq.num_completion_tokens + seq.num_output_placeholders, seq.max_tokens)
            pending = [item for batch in sch._inflight.values() for item in batch.items
                       if item.seq_id == seq.seq_id]
            self.assertEqual(seq.num_in_flight_tokens, sum(item.num_scheduled_tokens for item in pending))
            self.assertEqual(seq.num_output_placeholders, sum(item.produces_output for item in pending))
            if not seq.is_finished:
                self.assertLessEqual(seq.num_cached_tokens, seq.num_computed_tokens)
                self.assertEqual(seq.num_computed_tokens - seq.num_cached_tokens, seq.num_in_flight_tokens)
        self.assertTrue(all(s.status == SequenceStatus.RUNNING for s in sch.running))
        self.assertTrue(all(s.status == SequenceStatus.WAITING for s in sch.waiting))

    def test_two_prefill_chunks_use_independent_snapshots(self):
        sch = self.make()
        seq = self.seq(12)
        sch.add(seq)
        first, second = sch.schedule(), sch.schedule()
        self.assertEqual((first.items[0].start_pos, second.items[0].start_pos), (0, 4))
        self.assertEqual((seq.num_computed_tokens, seq.num_cached_tokens, seq.num_in_flight_tokens), (8, 0, 8))
        self.assertEqual(first.items[0].input_token_ids, (10, 11, 12, 13))
        self.assertEqual(len(first.items[0].block_table), 1)
        self.assertEqual(len(second.items[0].block_table), 2)
        self.assertFalse(sch.schedule().items)
        with self.assertRaises(FrozenInstanceError):
            first.items[0].start_pos = 10
        self.assertEqual(pickle.loads(pickle.dumps(first)), first)
        self.finish(sch, first)
        self.assertEqual((seq.num_cached_tokens, seq.num_in_flight_tokens), (4, 4))
        self.finish(sch, second)

    def test_running_order_does_not_reorder_prefill_after_decode(self):
        sch = self.make(budget=6)
        a, b = self.seq(10), self.seq(2, offset=100)
        sch.add(a)
        self.finish(sch, sch.schedule())
        sch.add(b)
        self.finish(sch, sch.schedule())
        self.assertEqual(list(sch.running), [a, b])
        c = self.seq(12, offset=200)
        sch.add(c)
        batch = sch.schedule()
        self.assertEqual([i.seq_id for i in batch.items], [a.seq_id, b.seq_id, c.seq_id])
        self.finish(sch, batch)
        d = self.seq(2, offset=300)
        sch.add(d)
        batch = sch.schedule()
        self.assertEqual([i.seq_id for i in batch.items][:3], [a.seq_id, b.seq_id, c.seq_id])

    def test_partial_prefill_can_precede_later_decode_in_running(self):
        sch = self.make(budget=4)
        a, b = self.seq(8), self.seq(2, offset=100)
        sch.add(b)
        self.finish(sch, sch.schedule())
        sch.add(a)
        self.finish(sch, sch.schedule())
        sch.running = deque([a, b])
        batch = sch.schedule()
        self.assertEqual(batch.items[0].seq_id, a.seq_id)
        self.assertTrue(batch.items[0].is_prefill)
        self.assertEqual(batch.num_prefill_tokens, 4)

    def test_output_placeholders_supply_unknown_decode_token(self):
        sch = self.make()
        seq = self.seq(2)
        sch.add(seq)
        first, second = sch.schedule(), sch.schedule()
        self.assertTrue(first.items[0].produces_output)
        self.assertEqual(first.items[0].output_index, 0)
        self.assertEqual(second.items[0].input_token_ids, (-1,))
        self.assertEqual(second.items[0].start_pos, 2)
        self.assertEqual(second.items[0].output_index, 1)
        self.assertEqual(seq.num_output_placeholders, 2)
        self.finish(sch, first, 77)
        self.assertEqual(seq.token_ids, [10, 11, 77])
        self.assertEqual(seq.num_output_placeholders, 1)
        self.finish(sch, second, 88)
        self.assertEqual(seq.completion_token_ids, [77, 88])

    def test_mid_chunk_sample_is_discarded(self):
        sch = self.make()
        seq = self.seq(6)
        sch.add(seq)
        self.assertEqual(self.finish(sch, sch.schedule()), [])
        self.assertEqual(seq.num_completion_tokens, 0)
        accepted = self.finish(sch, sch.schedule(), 77)
        self.assertEqual(accepted, [(seq.seq_id, 77)])

    def test_eos_retains_blocks_until_inflight_drains(self):
        sch = self.make()
        seq = self.seq(4, ignore=False)
        sch.add(seq)
        first, second = sch.schedule(), sch.schedule()
        blocks = tuple(seq.block_table)
        self.finish(sch, first, sch.eos)
        self.assertTrue(seq.is_finished)
        self.assertFalse(sch.is_finished())
        self.assertEqual(tuple(seq.block_table), blocks)
        self.assertEqual(self.finish(sch, second), [])
        self.assertFalse(seq.block_table)
        self.assertTrue(sch.is_finished())
        self.assertEqual(seq.completion_token_ids, [sch.eos])

    def test_max_tokens_prevents_extra_inflight_outputs(self):
        sch = self.make()
        seq = self.seq(2, max_tokens=1)
        sch.add(seq)
        first = sch.schedule()
        self.assertFalse(sch.schedule().items)
        self.finish(sch, first)
        self.assertEqual(seq.num_completion_tokens, 1)
        self.assertTrue(sch.is_finished())

    def test_out_of_order_duplicate_and_wrong_length_rejected(self):
        sch = self.make()
        sch.add(self.seq(12))
        first, second = sch.schedule(), sch.schedule()
        with self.assertRaises(ValueError): sch.postprocess(second, [1])
        with self.assertRaises(ValueError): sch.postprocess(first, [])
        self.finish(sch, first)
        with self.assertRaises(ValueError): sch.postprocess(first, [1])
        self.finish(sch, second)

    def test_prefix_only_publishes_confirmed_full_blocks(self):
        sch = self.make()
        a = self.seq(8, max_tokens=1)
        sch.add(a)
        first, second = sch.schedule(), sch.schedule()
        self.assertFalse(sch.block_manager.hash_to_block_id)
        self.finish(sch, first)
        self.assertEqual(len(sch.block_manager.hash_to_block_id), 1)
        self.finish(sch, second)
        b = self.seq(8, max_tokens=1)
        sch.add(b)
        batch = sch.schedule()
        self.assertEqual(batch.items[0].start_pos, 4)
        self.assertEqual(batch.num_prefill_tokens, 4)

    def test_shared_cached_prefix_reference_counts(self):
        sch = self.make(budget=8)
        a = self.seq(9, max_tokens=1)
        sch.add(a)
        while not sch.is_finished(): self.finish(sch, sch.schedule())
        b, c = self.seq(9), self.seq(9)
        sch.add(b)
        sch.add(c)
        batch = sch.schedule()
        self.assertEqual(b.block_table[:2], c.block_table[:2])
        self.assertNotEqual(b.block_table[2], c.block_table[2])
        for block in b.block_table[:2]: self.assertEqual(sch.block_manager.blocks[block].ref_count, 2)
        self.finish(sch, batch)

    def test_active_sequence_limit_includes_partial_prefill(self):
        sch = self.make(seqs=1)
        a, b = self.seq(10), self.seq(2, offset=100)
        sch.add(a)
        sch.add(b)
        self.finish(sch, sch.schedule())
        self.assertEqual(list(sch.running), [a])
        self.assertEqual(list(sch.waiting), [b])

    def test_abort_queued_and_inflight(self):
        sch = self.make(seqs=1)
        a, b = self.seq(8), self.seq(2, offset=100)
        sch.add(a)
        sch.add(b)
        first = sch.schedule()
        sch.abort(b.seq_id)
        sch.abort(a.seq_id)
        self.assertTrue(a.block_table)
        self.assertFalse(sch.is_finished())
        self.assertEqual(self.finish(sch, first), [])
        self.assertTrue(sch.is_finished())

    def test_completed_and_queued_aborted_requests_leave_registry(self):
        sch = self.make()
        completed = self.seq(2, max_tokens=1)
        queued = self.seq(2, offset=100)
        sch.add(completed)
        sch.add(queued)
        self.assertTrue(sch.abort(queued.seq_id))
        self.assertNotIn(queued.seq_id, sch.requests)
        self.finish(sch, sch.schedule())
        self.assertTrue(completed.is_finished)
        self.assertEqual(completed.num_completion_tokens, 1)
        self.assertFalse(sch.requests)

    def test_finished_request_remains_registered_until_last_inflight_result(self):
        sch = self.make()
        seq = self.seq(2, ignore=False)
        sch.add(seq)
        first, second = sch.schedule(), sch.schedule()
        self.finish(sch, first, sch.eos)
        self.assertIs(sch.requests[seq.seq_id], seq)
        self.assertTrue(seq.is_finished)
        self.assertEqual(self.finish(sch, second), [])
        self.assertNotIn(seq.seq_id, sch.requests)
        self.assertEqual(seq.completion_token_ids, [sch.eos])

    def test_full_admission_check_differs_from_actual_allocation(self):
        sch = self.make(blocks=3, full=True)
        a = self.seq(4)
        sch.add(a)
        self.finish(sch, sch.schedule())
        b = self.seq(8, offset=100)
        sch.add(b)
        batch = sch.schedule()
        self.assertEqual([i.seq_id for i in batch.items], [a.seq_id])
        self.assertEqual(list(sch.waiting), [b])
        self.assertFalse(b.block_table)

    def test_preemption_never_releases_inflight_blocks(self):
        sch = self.make(budget=4, blocks=2, full=False)
        a, b = self.seq(4), self.seq(4, offset=100)
        sch.add(a)
        sch.add(b)
        first = sch.schedule()
        second = sch.schedule()
        before = tuple(a.block_table)
        self.assertEqual(tuple(a.block_table), before)
        self.assertEqual(a.generation, 0)
        self.assertTrue(a.num_in_flight_tokens)
        self.finish(sch, first)
        if second.items: self.finish(sch, second)

    def test_running_tail_preemption_recomputes_known_context(self):
        sch = self.make(budget=8, blocks=3, full=False)
        a, b = self.seq(4, max_tokens=3), self.seq(4, offset=100, max_tokens=3)
        sch.add(a)
        sch.add(b)
        self.finish(sch, sch.schedule(), 77)
        first = sch.schedule()
        self.assertEqual([i.seq_id for i in first.items], [a.seq_id])
        self.assertEqual(b.status, SequenceStatus.WAITING)
        self.assertEqual(b.generation, 1)
        self.assertEqual(b.token_ids, [100, 101, 102, 103, 77])
        self.assertFalse(b.block_table)
        self.finish(sch, first)
        for _ in range(10):
            if sch.is_finished(): break
            batch = sch.schedule()
            self.assertTrue(batch.items)
            self.finish(sch, batch)
        self.assertTrue(sch.is_finished())
        self.assertEqual(b.num_completion_tokens, 3)

    def test_inflight_kv_exhaustion_returns_empty_without_preemption(self):
        sch = self.make(budget=4, blocks=1, full=False)
        seq = self.seq(4)
        sch.add(seq)
        first = sch.schedule()
        self.assertFalse(sch.schedule().items)
        self.assertEqual(seq.generation, 0)
        self.assertEqual(seq.num_in_flight_tokens, 4)
        self.assertEqual(len(seq.block_table), 1)
        self.finish(sch, first)

    def test_one_batch_mode_never_advances_before_retirement(self):
        sch = self.make(inflight=1)
        seq = self.seq(10)
        sch.add(seq)
        first = sch.schedule()
        self.assertFalse(sch.schedule().items)
        self.assertEqual(seq.num_computed_tokens, 4)
        self.finish(sch, first)
        self.assertEqual(sch.schedule().items[0].start_pos, 4)

    def test_mid_block_confirmation_hashes_only_at_full_boundary(self):
        sch = self.make(budget=3)
        seq = self.seq(9, max_tokens=1)
        sch.add(seq)
        first, second = sch.schedule(), sch.schedule()
        self.finish(sch, first)
        self.assertFalse(sch.block_manager.hash_to_block_id)
        self.finish(sch, second)
        self.assertEqual(len(sch.block_manager.hash_to_block_id), 1)
        self.finish(sch, sch.schedule())
        self.assertEqual(len(sch.block_manager.hash_to_block_id), 2)

    def test_model_and_pool_capacity_errors(self):
        sch = self.make(blocks=2)
        with self.assertRaises(ValueError): sch.add(self.seq(9))
        sch = self.make(model_len=8)
        with self.assertRaises(ValueError): sch.add(self.seq(8))
        seq = self.seq(6, max_tokens=10)
        sch.add(seq)
        while not sch.is_finished(): self.finish(sch, sch.schedule())
        self.assertEqual(seq.num_tokens, 8)

    def test_randomized_two_batch_progress_and_block_accounting(self):
        rng = random.Random(731)
        for _ in range(15):
            sch = self.make(budget=rng.randint(1, 8), seqs=rng.randint(1, 4), blocks=16, full=False)
            requests = [self.seq(rng.randint(1, 20), offset=100*i, max_tokens=3) for i in range(8)]
            for seq in requests: sch.add(seq)
            queue = deque()
            for step in range(1000):
                batch = sch.schedule()
                if batch.items: queue.append(batch)
                if queue and (not batch.items or rng.random() < 0.6):
                    self.finish(sch, queue.popleft())
                if sch.is_finished(): break
            self.assertTrue(sch.is_finished())
            self.assertTrue(all(s.num_completion_tokens == 3 for s in requests))

    def test_prefill_threshold_applies_on_admission_and_running(self):
        sch = self.make(budget=16, threshold=3)
        a, b = self.seq(10), self.seq(10, offset=100)
        sch.add(a)
        sch.add(b)
        first = sch.schedule()
        self.assertEqual([item.num_scheduled_tokens for item in first.items], [3, 3])
        self.assertEqual(first.num_scheduled_tokens, 6)
        second = sch.schedule()
        self.assertEqual([item.start_pos for item in second.items], [3, 3])
        self.assertEqual([item.num_scheduled_tokens for item in second.items], [3, 3])
        self.finish(sch, first)
        self.finish(sch, second)

    def test_prefill_threshold_applies_after_preemption(self):
        sch = self.make(budget=16, threshold=2)
        seq = self.seq(3)
        sch.add(seq)
        self.finish(sch, sch.schedule())
        self.finish(sch, sch.schedule())
        self.assertEqual(seq.num_completion_tokens, 1)
        sch.preempt(seq)
        resumed = sch.schedule()
        self.assertTrue(resumed.items[0].is_prefill)
        self.assertEqual(resumed.items[0].num_scheduled_tokens, 2)
        self.finish(sch, resumed)
        final_chunk = sch.schedule()
        self.assertEqual(final_chunk.items[0].num_scheduled_tokens, 2)
        self.assertEqual(final_chunk.items[0].output_index, 1)

    def test_zero_prefill_threshold_preserves_remaining_budget(self):
        sch = self.make(budget=8, threshold=0)
        seq = self.seq(12)
        sch.add(seq)
        self.assertEqual(sch.schedule().num_scheduled_tokens, 8)

    def test_negative_prefill_threshold_is_rejected(self):
        with self.assertRaises(ValueError): self.make(threshold=-1)

    def test_two_pending_outputs_respect_max_tokens_after_first_retirement(self):
        sch = self.make()
        seq = self.seq(2, max_tokens=2)
        sch.add(seq)
        first, second = sch.schedule(), sch.schedule()
        self.finish(sch, first)
        self.assertFalse(sch.schedule().items)
        self.assertEqual(seq.num_output_placeholders, 1)
        self.finish(sch, second)
        self.assertTrue(sch.is_finished())
        self.assertEqual(seq.num_completion_tokens, 2)

    def test_eos_candidate_from_incomplete_prefill_is_ignored(self):
        sch = self.make(budget=2)
        seq = self.seq(5, ignore=False)
        sch.add(seq)
        first, second = sch.schedule(), sch.schedule()
        self.finish(sch, first, sch.eos)
        self.finish(sch, second, sch.eos)
        self.assertFalse(seq.is_finished)
        self.assertEqual(seq.num_completion_tokens, 0)
        self.finish(sch, sch.schedule(), sch.eos)
        self.assertTrue(seq.is_finished)
        self.assertEqual(seq.num_completion_tokens, 1)

    def test_prefill_threshold_does_not_delay_a_later_running_decode(self):
        sch = self.make(budget=8, threshold=2)
        decode, prefill = self.seq(2), self.seq(10, offset=100)
        sch.add(decode)
        self.finish(sch, sch.schedule())
        sch.add(prefill)
        self.finish(sch, sch.schedule())
        sch.running = deque([prefill, decode])
        batch = sch.schedule()
        self.assertEqual([(item.seq_id, item.num_scheduled_tokens) for item in batch.items],
                         [(prefill.seq_id, 2), (decode.seq_id, 1)])

    def test_pressure_randomized_drain_abort_and_eos(self):
        rng = random.Random(159731)
        for trial in range(80):
            blocks = rng.randint(3, 12)
            sch = self.make(budget=rng.randint(1, 16), seqs=rng.randint(1, 5), blocks=blocks,
                            full=bool(trial % 2), model_len=blocks*4,
                            threshold=rng.choice((0, 1, 2, 3, 5)))
            requests = [self.seq(rng.randint(1, blocks*4-4), offset=(i % 4)*100,
                                 max_tokens=rng.randint(1, 4), ignore=bool(i % 2)) for i in range(10)]
            for seq in requests: sch.add(seq)
            queue = deque()
            no_work_without_inflight = 0
            for step in range(2000):
                batch = sch.schedule()
                if batch.items:
                    queue.append(batch)
                    no_work_without_inflight = 0
                elif not queue and not sch.is_finished():
                    no_work_without_inflight += 1
                    self.assertLessEqual(no_work_without_inflight, 1,
                                         f'stalled trial={trial}, step={step}')
                if queue and (not batch.items or rng.random() < 0.65):
                    batch = queue.popleft()
                    tokens = [sch.eos if rng.random() < 0.15 else 123 for _ in batch.items]
                    sch.postprocess(batch, tokens)
                    self.invariants(sch)
                if rng.random() < 0.04:
                    live = [s for s in sch.requests.values() if not s.is_finished]
                    if live: sch.abort(rng.choice(live).seq_id)
                if sch.is_finished(): break
            self.assertTrue(sch.is_finished(), f'trial {trial} failed to finish')
            for seq in requests:
                self.assertEqual(seq.num_output_placeholders, 0)
                self.assertEqual(seq.num_in_flight_tokens, 0)
                self.assertLessEqual(seq.num_completion_tokens, seq.max_tokens)
                self.assertFalse(seq.block_table)


if __name__ == '__main__':
    unittest.main(verbosity=2)
