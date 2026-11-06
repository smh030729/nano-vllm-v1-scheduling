from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.prev_free_block = None
        self.next_free_block = None

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = list(token_ids)

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class FreeBlockQueue:
    """Intrusive free list; external users address blocks by ID."""

    def __init__(self, blocks):
        for block_id, block in enumerate(blocks):
            if (block.block_id != block_id or block.ref_count
                    or block.prev_free_block is not None or block.next_free_block is not None):
                raise ValueError('initial blocks must be ordered, unreferenced and detached')
        self._blocks = blocks
        self._size = 0
        self._head = Block(-1)
        self._tail = Block(-1)
        self._head.next_free_block = self._tail
        self._tail.prev_free_block = self._head
        for block in blocks:
            self.append(block.block_id)

    def __len__(self):
        return self._size

    def __iter__(self):
        block = self._head.next_free_block
        while block is not self._tail:
            yield block.block_id
            block = block.next_free_block

    def __reversed__(self):
        block = self._tail.prev_free_block
        while block is not self._head:
            yield block.block_id
            block = block.prev_free_block

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError('queue index must be an integer')
        if not self._size:
            raise IndexError('empty free block queue')
        if index == 0:
            return self._head.next_free_block.block_id
        if index == -1:
            return self._tail.prev_free_block.block_id
        raise IndexError('only endpoint indexing is supported')

    def __contains__(self, block_id):
        return (isinstance(block_id, int) and 0 <= block_id < len(self._blocks)
                and self._blocks[block_id].prev_free_block is not None)

    def _get_block(self, block_id):
        if not isinstance(block_id, int) or not 0 <= block_id < len(self._blocks):
            raise ValueError('invalid physical block ID')
        return self._blocks[block_id]

    def remove(self, block_id):
        block = self._get_block(block_id)
        previous, following = block.prev_free_block, block.next_free_block
        if previous is None or following is None:
            raise ValueError('block is not in the free queue')
        previous.next_free_block = following
        following.prev_free_block = previous
        block.prev_free_block = block.next_free_block = None
        self._size -= 1

    def popleft(self):
        block_id = self[0]
        self.remove(block_id)
        return block_id

    def _insert(self, block_id, previous, following):
        block = self._get_block(block_id)
        if block.ref_count or block.prev_free_block is not None or block.next_free_block is not None:
            raise ValueError('only detached, unreferenced blocks may be inserted')
        block.prev_free_block, block.next_free_block = previous, following
        previous.next_free_block = following.prev_free_block = block
        self._size += 1

    def append(self, block_id):
        self._insert(block_id, self._tail.prev_free_block, self._tail)

    def appendleft(self, block_id):
        self._insert(block_id, self._head, self._head.next_free_block)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, *, uncached_first=False,
                 intrusive_free_queue=False):
        self.block_size = block_size
        self.uncached_first = uncached_first
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}
        self.free_block_ids = (FreeBlockQueue(self.blocks) if intrusive_free_queue
                               else deque(range(num_blocks)))
        self.used_block_ids = set()
        self.peak_used_blocks = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, 'little'))
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    def _allocate_block(self, retained=None):
        block_id = self.free_block_ids[0]
        if retained and block_id in retained:
            block_id = next((i for i in self.free_block_ids if i not in retained), block_id)
        if block_id == self.free_block_ids[0]:
            self.free_block_ids.popleft()
        else:
            self.free_block_ids.remove(block_id)
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        self.peak_used_blocks = max(self.peak_used_blocks, len(self.used_block_ids))
        return block_id

    def cached_prefix(self, seq: Sequence):
        block_ids = []
        prefix = -1
        # 至少保留一个输入位置，用于计算下一个 token 的分布。
        for i in range((len(seq)-1) // self.block_size):
            tokens = seq.block(i)
            prefix = self.compute_hash(tokens, prefix)
            block_id = self.hash_to_block_id.get(prefix)
            if block_id is None or self.blocks[block_id].token_ids != tokens:
                break
            block_ids.append(block_id)
        return tuple(block_ids)

    def _get_retained_blocks(self, current, waiting_seqs):
        retained = set()
        for seq in waiting_seqs:
            if seq is not current and not seq.block_table:
                retained.update(self.cached_prefix(seq))
        return retained

    def _allocate_blocks(self, seq, count, waiting_seqs):
        retained = None
        for _ in range(count):
            if (waiting_seqs is not None and retained is None
                    and self.blocks[self.free_block_ids[0]].hash != -1):
                retained = self._get_retained_blocks(seq, waiting_seqs)
            yield self._allocate_block(retained)

    def can_allocate_slots(self, seq: Sequence, end_pos: int, cached_blocks=(), full_sequence_tokens=None):
        if end_pos < 0:
            return False
        allocated = len(seq.block_table)
        if allocated and cached_blocks:
            raise ValueError('prefix blocks may only be attached on admission')
        cached_count = len(cached_blocks)
        required = (end_pos + self.block_size - 1) // self.block_size
        check_required = required
        if full_sequence_tokens is not None:
            check_required = max(required, (full_sequence_tokens + self.block_size - 1) // self.block_size)
        cached_free = sum(self.blocks[i].ref_count == 0 for i in cached_blocks)
        additional = max(0, check_required - allocated - cached_count)
        return cached_free + additional <= len(self.free_block_ids)

    def allocate_slots(self, seq: Sequence, end_pos: int, cached_blocks=(), waiting_seqs=None):
        if not self.can_allocate_slots(seq, end_pos, cached_blocks):
            raise RuntimeError('insufficient KV blocks for scheduled positions')
        if cached_blocks:
            for block_id in cached_blocks:
                block = self.blocks[block_id]
                if block.ref_count == 0:
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
                block.ref_count += 1
                seq.block_table.append(block_id)
            seq.num_cached_tokens = seq.num_computed_tokens = len(cached_blocks) * self.block_size
        required = (end_pos + self.block_size - 1) // self.block_size
        count = max(0, required - len(seq.block_table))
        seq.block_table.extend(self._allocate_blocks(seq, count, waiting_seqs))
        self.peak_used_blocks = max(self.peak_used_blocks, len(self.used_block_ids))

    def deallocate(self, seq: Sequence):
        if seq.num_in_flight_tokens:
            raise RuntimeError('cannot release KV blocks still referenced by in-flight work')
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            assert block.ref_count >= 0
            if block.ref_count == 0:
                self.used_block_ids.remove(block_id)
                if self.uncached_first and block.hash == -1:
                    self.free_block_ids.appendleft(block_id)
                else:
                    self.free_block_ids.append(block_id)
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        seq.num_computed_tokens = 0

    def hash_blocks(self, seq: Sequence, start_pos: int, end_pos: int):
        if end_pos > len(seq):
            raise ValueError('cannot publish a prefix whose token IDs are unknown')
        first = start_pos // self.block_size
        stop = end_pos // self.block_size
        if first == stop:
            return
        prefix = self.blocks[seq.block_table[first-1]].hash if first else -1
        if first and prefix == -1:
            raise RuntimeError('previous full prefix block has not been confirmed')
        for i in range(first, stop):
            block = self.blocks[seq.block_table[i]]
            tokens = seq.block(i)
            if len(tokens) != self.block_size:
                raise ValueError('only complete KV blocks may enter the prefix cache')
            prefix = self.compute_hash(tokens, prefix)
            if block.hash != -1 and (block.hash != prefix or block.token_ids != tokens):
                raise RuntimeError('attempted to overwrite a shared cached prefix')
            block.update(prefix, tokens)
            self.hash_to_block_id[prefix] = block.block_id
