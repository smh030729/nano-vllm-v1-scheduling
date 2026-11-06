from collections import deque
import random

import pytest

from nanovllm.engine import block_manager


def queue_type():
    cls = getattr(block_manager, "FreeBlockQueue", None)
    assert cls is not None, "FreeBlockQueue 尚未实现"
    return cls


def test_intrusive_queue_interface_and_detached_block_links():
    block = block_manager.Block(0)
    missing = object()
    assert getattr(block, "prev_free_block", missing) is None
    assert getattr(block, "next_free_block", missing) is None
    assert queue_type() is not None


def make_queue(size):
    blocks = [block_manager.Block(i) for i in range(size)]
    return queue_type()(blocks), blocks


def assert_queue_state(queue, blocks, expected):
    expected = list(expected)
    assert len(expected) == len(set(expected))
    assert len(queue) == len(expected)
    assert bool(queue) is bool(expected)
    assert isinstance(queue._head, block_manager.Block)
    assert isinstance(queue._tail, block_manager.Block)
    assert queue._head is not queue._tail
    assert queue._head.block_id == queue._tail.block_id == -1

    linked = [queue._head, *(blocks[i] for i in expected), queue._tail]
    for i, block in enumerate(linked):
        assert block.prev_free_block is (linked[i - 1] if i else None)
        assert block.next_free_block is (linked[i + 1] if i + 1 < len(linked) else None)

    members = set(expected)
    for block in blocks:
        assert (block.block_id in queue) is (block.block_id in members)
        if block.block_id not in members:
            assert block.prev_free_block is None
            assert block.next_free_block is None

    assert list(queue) == expected
    assert list(reversed(queue)) == expected[::-1]
    if expected:
        assert queue[0] == expected[0]
        assert queue[-1] == expected[-1]
    else:
        with pytest.raises(IndexError):
            queue[0]
        with pytest.raises(IndexError):
            queue[-1]


def test_empty_queue_has_connected_sentinels_and_rejects_popleft():
    queue, blocks = make_queue(0)
    assert_queue_state(queue, blocks, [])
    with pytest.raises(IndexError):
        queue.popleft()
    assert_queue_state(queue, blocks, [])


@pytest.mark.parametrize("insert", ["append", "appendleft"])
def test_single_block_can_leave_and_reenter_queue(insert):
    queue, blocks = make_queue(1)
    assert_queue_state(queue, blocks, [0])
    assert queue.popleft() == 0
    assert_queue_state(queue, blocks, [])
    getattr(queue, insert)(0)
    assert_queue_state(queue, blocks, [0])
    queue.remove(0)
    assert_queue_state(queue, blocks, [])


@pytest.mark.parametrize("removed", [0, 2, 4])
def test_remove_head_middle_or_tail_reconnects_both_neighbors(removed):
    queue, blocks = make_queue(5)
    queue.remove(removed)
    expected = [i for i in range(5) if i != removed]
    assert_queue_state(queue, blocks, expected)
    queue.append(removed)
    assert_queue_state(queue, blocks, [*expected, removed])


def test_appendleft_and_append_preserve_requested_order():
    queue, blocks = make_queue(5)
    for block_id in [1, 3, 4]:
        queue.remove(block_id)
    assert_queue_state(queue, blocks, [0, 2])
    queue.appendleft(3)
    queue.append(1)
    queue.appendleft(4)
    assert_queue_state(queue, blocks, [4, 3, 0, 2, 1])
    for expected in [4, 3, 0, 2, 1]:
        assert queue.popleft() == expected
    assert_queue_state(queue, blocks, [])


@pytest.mark.parametrize("insert", ["append", "appendleft"])
@pytest.mark.parametrize("block_id", [0, 2, 4])
def test_duplicate_insert_is_rejected_without_mutation(insert, block_id):
    queue, blocks = make_queue(5)
    with pytest.raises(ValueError):
        getattr(queue, insert)(block_id)
    assert_queue_state(queue, blocks, range(5))


@pytest.mark.parametrize("insert", ["append", "appendleft"])
def test_referenced_block_cannot_reenter_free_queue(insert):
    queue, blocks = make_queue(3)
    queue.remove(1)
    blocks[1].ref_count = 1
    with pytest.raises(ValueError):
        getattr(queue, insert)(1)
    assert_queue_state(queue, blocks, [0, 2])
    blocks[1].ref_count = 0
    getattr(queue, insert)(1)
    assert_queue_state(queue, blocks, [0, 2, 1] if insert == "append" else [1, 0, 2])


def test_repeated_remove_is_rejected_without_mutation():
    queue, blocks = make_queue(3)
    queue.remove(1)
    with pytest.raises(ValueError):
        queue.remove(1)
    assert_queue_state(queue, blocks, [0, 2])


@pytest.mark.parametrize("operation", ["append", "appendleft", "remove"])
@pytest.mark.parametrize("invalid_id", [-1, 3, 999, 1.0, 1.5, "1", None])
def test_mutations_reject_invalid_block_ids(operation, invalid_id):
    queue, blocks = make_queue(3)
    with pytest.raises(ValueError):
        getattr(queue, operation)(invalid_id)
    assert_queue_state(queue, blocks, [0, 1, 2])


@pytest.mark.parametrize("invalid_id", [-1, 3, 999, 1.0, 1.5, "1", None])
def test_membership_returns_false_for_invalid_block_ids(invalid_id):
    queue, blocks = make_queue(3)
    assert invalid_id not in queue
    assert_queue_state(queue, blocks, [0, 1, 2])


@pytest.mark.parametrize("ids", [[1], [0, 0], [1, 0], [0, 2], [-1]])
def test_constructor_requires_consecutive_ids_without_linking_invalid_input(ids):
    blocks = [block_manager.Block(i) for i in ids]
    with pytest.raises(ValueError):
        queue_type()(blocks)
    for block in blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None


def test_constructor_rejects_referenced_blocks_before_linking_any_node():
    blocks = [block_manager.Block(i) for i in range(3)]
    blocks[-1].ref_count = 1
    with pytest.raises(ValueError):
        queue_type()(blocks)
    for block in blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None


@pytest.mark.parametrize("link", ["prev_free_block", "next_free_block"])
def test_constructor_rejects_existing_links_without_changing_them(link):
    blocks = [block_manager.Block(i) for i in range(3)]
    peer = block_manager.Block(-1)
    setattr(blocks[-1], link, peer)
    before = [(block.prev_free_block, block.next_free_block) for block in blocks]
    with pytest.raises(ValueError):
        queue_type()(blocks)
    assert [(block.prev_free_block, block.next_free_block) for block in blocks] == before


def test_constructing_second_queue_cannot_corrupt_existing_queue():
    queue, blocks = make_queue(3)
    with pytest.raises(ValueError):
        queue_type()(blocks)
    assert_queue_state(queue, blocks, [0, 1, 2])


def test_block_reset_does_not_silently_clear_free_queue_links():
    queue, blocks = make_queue(3)
    block = blocks[1]
    previous, following = block.prev_free_block, block.next_free_block
    block.reset()
    assert block.prev_free_block is previous
    assert block.next_free_block is following
    assert previous.next_free_block is block
    assert following.prev_free_block is block
    assert len(queue) == 3


def test_remove_does_not_iterate_over_queue(monkeypatch):
    queue, blocks = make_queue(128)

    def reject_iteration(self):
        raise AssertionError("remove 不应遍历队列")

    with monkeypatch.context() as patch:
        patch.setattr(type(queue), "__iter__", reject_iteration)
        patch.setattr(type(queue), "__reversed__", reject_iteration)
        queue.remove(63)
    assert_queue_state(queue, blocks, [i for i in range(128) if i != 63])


@pytest.mark.parametrize("seed", [0, 19, 2026])
def test_random_operations_match_deque_after_every_step(seed):
    rng = random.Random(seed)
    queue, blocks = make_queue(31)
    expected = deque(range(len(blocks)))
    assert_queue_state(queue, blocks, expected)
    for _ in range(1500):
        operation = rng.choice(["append", "appendleft", "popleft", "remove"])
        if operation in ("append", "appendleft"):
            detached = [i for i in range(len(blocks)) if i not in expected]
            if detached:
                block_id = rng.choice(detached)
                getattr(queue, operation)(block_id)
                getattr(expected, operation)(block_id)
        elif operation == "popleft":
            if expected:
                assert queue.popleft() == expected.popleft()
            else:
                with pytest.raises(IndexError):
                    queue.popleft()
        elif expected:
            block_id = rng.choice(list(expected))
            queue.remove(block_id)
            expected.remove(block_id)
        else:
            with pytest.raises(ValueError):
                queue.remove(rng.randrange(len(blocks)))
        assert_queue_state(queue, blocks, expected)
