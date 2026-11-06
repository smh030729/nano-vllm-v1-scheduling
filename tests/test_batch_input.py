import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def item(seq_id, start, tokens, blocks):
    return SimpleNamespace(seq_id=seq_id, generation=0, start_pos=start,
        num_scheduled_tokens=len(tokens), input_token_ids=tuple(tokens), block_table=tuple(blocks))


def test_metadata_uses_snapshot_positions_and_scattered_blocks():
    from nanovllm.engine.batch_input import build_batch_metadata
    data = build_batch_metadata([
        item(1, 4, [-1], [7, 2]), item(2, 3, [33, 34, 35, 36], [1, 8])], 4)
    assert data['input_ids'] == [0, 33, 34, 35, 36]
    assert data['positions'] == [4, 3, 4, 5, 6]
    assert data['slot_mapping'] == [8, 7, 32, 33, 34]
    assert data['cu_seqlens_q'] == [0, 1, 5]
    assert data['cu_seqlens_k'] == [0, 5, 12]
    assert data['feedback'] == [(0, (1, 0), 4)]


def test_short_block_table_is_rejected_before_gpu_submission():
    import pytest
    from nanovllm.engine.batch_input import build_batch_metadata
    with pytest.raises(ValueError, match='block'):
        build_batch_metadata([item(1, 3, [3, 4], [0])], 4)


def test_unknown_token_must_be_single_decode_input():
    import pytest
    from nanovllm.engine.batch_input import build_batch_metadata
    with pytest.raises(ValueError, match='unknown'):
        build_batch_metadata([item(1, 3, [-1, 4], [0, 1])], 4)


def test_uniform_queries_use_the_same_table_width_as_full_graph_without_more_kv():
    from nanovllm.engine.batch_input import build_batch_metadata
    data = build_batch_metadata([item(1, 3, [9], [7]), item(2, 4, [10], [1, 2])],
                                4, decode_table_width=4)
    assert data['block_tables'] == [[7, 0, 0, 0], [1, 2, 0, 0]]
    assert data['slot_mapping'] == [31, 8]
    assert data['context_lens'] == [4, 5]


def test_variable_queries_keep_dynamic_block_table_width():
    from nanovllm.engine.batch_input import build_batch_metadata
    data = build_batch_metadata([item(1, 2, [9, 10], [7])], 4, decode_table_width=4)
    assert data['block_tables'] == [[7]]


def test_decode_table_width_cannot_hide_live_blocks():
    import pytest
    from nanovllm.engine.batch_input import build_batch_metadata
    with pytest.raises(ValueError, match='width'):
        build_batch_metadata([item(1, 4, [9], [7, 8])], 4, decode_table_width=1)
