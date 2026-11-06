from collections import Counter, deque
from itertools import product
import random
from types import SimpleNamespace

import pytest

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager, FreeBlockQueue
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    monkeypatch.setattr(Sequence, 'block_size', 4)


def request(size, value=11, max_tokens=3):
    return Sequence([value] * size, SamplingParams(max_tokens=max_tokens, ignore_eos=True))


def seed(manager, value, size=8):
    seq = request(size, value)
    prefix = manager.cached_prefix(seq)
    manager.allocate_slots(seq, len(seq), prefix)
    manager.hash_blocks(seq, len(prefix) * manager.block_size, len(seq))
    manager.deallocate(seq)


def scheduler(**kwargs):
    values = dict(max_num_seqs=4, max_num_batched_tokens=6, eos=99999,
                  num_kvcache_blocks=10, kvcache_block_size=4,
                  max_inflight_batches=2, max_model_len=40,
                  admission_full_sequence=True, long_prefill_token_threshold=3)
    values.update(kwargs)
    return Scheduler(SimpleNamespace(**values))


@pytest.mark.parametrize('intrusive', [False, True])
def test_retention_changes_victim_without_mutating_prefix_query(intrusive):
    manager = BlockManager(4, 4, intrusive_free_queue=intrusive)
    seed(manager, 11)
    seed(manager, 22)
    current, waiting = request(4, 33), request(12, 11)
    before = (tuple(manager.free_block_ids), [b.ref_count for b in manager.blocks],
              dict(manager.hash_to_block_id))
    protected = manager._get_retained_blocks(current, [current, waiting])
    assert manager.free_block_ids[0] in protected
    assert before == (tuple(manager.free_block_ids), [b.ref_count for b in manager.blocks],
                      dict(manager.hash_to_block_id))
    manager.allocate_slots(current, 4, waiting_seqs=[current, waiting])
    assert current.block_table[0] not in protected
    assert tuple(manager.cached_prefix(waiting)) == (0, 1)


@pytest.mark.parametrize('intrusive', [False, True])
def test_all_retained_falls_back_to_original_head(intrusive):
    manager = BlockManager(4, 4, intrusive_free_queue=intrusive)
    seed(manager, 11)
    seed(manager, 22)
    head = manager.free_block_ids[0]
    assert manager._allocate_block(set(manager.free_block_ids)) == head


@pytest.mark.parametrize('intrusive', [False, True])
def test_uncached_first_defers_lookahead_until_cached_victim(intrusive, monkeypatch):
    manager = BlockManager(4, 4, uncached_first=True, intrusive_free_queue=intrusive)
    seed(manager, 11, size=5)
    calls = []
    original = manager._get_retained_blocks
    def retained(current, waiting):
        calls.append(1)
        return original(current, waiting)
    monkeypatch.setattr(manager, '_get_retained_blocks', retained)
    current = request(9, 22)
    manager.allocate_slots(current, 9, waiting_seqs=[request(8, 11)])
    assert calls == []
    assert all(manager.blocks[i].hash == -1 for i in current.block_table)
    manager.allocate_slots(current, 13, waiting_seqs=[request(8, 11)])
    assert calls == [1]


@pytest.mark.parametrize('intrusive', [False, True])
def test_empty_retained_set_is_built_only_once(intrusive, monkeypatch):
    manager = BlockManager(4, 4, intrusive_free_queue=intrusive)
    seed(manager, 11)
    seed(manager, 22)
    calls = []
    original = manager._get_retained_blocks
    def retained(current, waiting):
        calls.append(1)
        return original(current, waiting)
    monkeypatch.setattr(manager, '_get_retained_blocks', retained)
    manager.allocate_slots(request(8, 33), 8, waiting_seqs=[request(8, 99)])
    assert calls == [1]


def test_scheduler_lookahead_is_bounded_and_only_protects_confirmed_blocks():
    sch = scheduler(prefix_cache_lookahead=2)
    seed(sch.block_manager, 11)
    seed(sch.block_manager, 22)
    current, near, far = request(4, 33), request(12, 11), request(12, 22)
    sch.waiting.extend([current, near, far])
    protected = sch.block_manager._get_retained_blocks(current, sch._eviction_lookahead())
    assert protected == set(sch.block_manager.cached_prefix(near))
    sch = scheduler(prefix_cache_lookahead=16)
    seq = request(12)
    sch.add(seq)
    sch.schedule()
    assert not sch.block_manager.hash_to_block_id
    assert sch.block_manager._get_retained_blocks(request(4, 22), [request(12)]) == set()


@pytest.mark.parametrize('mixed', [False, True])
def test_mixed_batching_switch(mixed):
    sch = scheduler(mixed_batching=mixed, max_num_batched_tokens=8,
                    long_prefill_token_threshold=0)
    decode, prefill = request(2), request(10, 22)
    sch.add(decode)
    sch.postprocess(sch.schedule(), [100])
    sch.add(prefill)
    batch = sch.schedule()
    assert batch.num_decode_tokens == 1
    assert bool(batch.num_prefill_tokens) == mixed


@pytest.mark.parametrize('incremental,expected_blocks', [(True, 1), (False, 3)])
def test_incremental_allocation_switch(incremental, expected_blocks):
    sch = scheduler(incremental_kv_allocation=incremental, max_num_batched_tokens=3)
    seq = request(10)
    sch.add(seq)
    batch = sch.schedule()
    assert batch.num_prefill_tokens == 3
    assert len(seq.block_table) == expected_blocks


def test_negative_lookahead_rejected_before_model_loading():
    with pytest.raises(ValueError, match='prefix_cache_lookahead'):
        Config('unused', prefix_cache_lookahead=-1)
    with pytest.raises(ValueError, match='prefix_cache_lookahead'):
        scheduler(prefix_cache_lookahead=-1)


@pytest.mark.parametrize('intrusive,uncached,lookahead,mixed,incremental,depth',
    list(product((False, True), (False, True), (0, 4), (False, True), (False, True), (1, 2))))
def test_switch_combinations_preserve_async_lifetimes(
        intrusive, uncached, lookahead, mixed, incremental, depth):
    sch = scheduler(intrusive_free_queue=intrusive, uncached_first=uncached,
                    prefix_cache_lookahead=lookahead, mixed_batching=mixed,
                    incremental_kv_allocation=incremental, max_inflight_batches=depth)
    assert isinstance(sch.block_manager.free_block_ids, FreeBlockQueue if intrusive else deque)
    seed(sch.block_manager, 11)
    rng = random.Random(513)
    for i in range(8):
        seq = request(rng.randint(1, 15), 11 + i % 3, max_tokens=4)
        seq.ignore_eos = False
        sch.add(seq)
    pending = deque()
    for _ in range(1000):
        batch = sch.schedule()
        if batch.items:
            if not mixed:
                assert len({item.is_prefill for item in batch.items}) == 1
            pending.append(batch)
        if pending and (not batch.items or rng.random() < .7):
            batch = pending.popleft()
            sch.postprocess(batch, [sch.eos if rng.random() < .1 else 100 for _ in batch.items])
        if rng.random() < .03:
            live = [s for s in sch.requests.values() if not s.is_finished]
            if live:
                sch.abort(rng.choice(live).seq_id)
        manager = sch.block_manager
        free = list(manager.free_block_ids)
        refs = Counter(b for seq in sch.requests.values() for b in seq.block_table)
        assert len(free) == len(set(free))
        assert set(free).isdisjoint(manager.used_block_ids)
        assert set(free) | manager.used_block_ids == set(range(len(manager.blocks)))
        assert [b.ref_count for b in manager.blocks] == [refs[b.block_id] for b in manager.blocks]
        if uncached:
            hashes = [manager.blocks[i].hash != -1 for i in free]
            assert hashes == sorted(hashes)
        for seq in sch.requests.values():
            items = [i for out in pending for i in out.items if i.seq_id == seq.seq_id]
            assert seq.num_in_flight_tokens == sum(i.num_scheduled_tokens for i in items)
            assert seq.num_output_placeholders == sum(i.produces_output for i in items)
        if sch.is_finished():
            break
    assert sch.is_finished()
    assert not pending and not sch.requests and not sch.block_manager.used_block_ids
