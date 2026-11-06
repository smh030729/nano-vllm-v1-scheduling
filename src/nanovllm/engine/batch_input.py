def build_batch_metadata(items, block_size, decode_table_width=None):
    data = dict(input_ids=[], positions=[], slot_mapping=[], cu_seqlens_q=[0],
                cu_seqlens_k=[0], context_lens=[], query_lens=[], feedback=[])
    if not items:
        raise ValueError('cannot execute an empty batch')
    for item in items:
        start = item.start_pos
        count = item.num_scheduled_tokens
        tokens = item.input_token_ids
        if count <= 0 or len(tokens) != count or start < 0:
            raise ValueError('invalid scheduled input span')
        end = start + count
        if len(item.block_table) * block_size < end:
            raise ValueError('block table does not cover the scheduled span')
        offset = len(data['input_ids'])
        for i, token in enumerate(tokens):
            if token < 0:
                if token != -1 or count != 1:
                    raise ValueError('unknown token must be a single pending decode input')
                data['feedback'].append((offset + i, (item.seq_id, item.generation), start + i))
            data['input_ids'].append(max(token, 0))
        data['positions'].extend(range(start, end))
        data['slot_mapping'].extend(item.block_table[pos // block_size] * block_size + pos % block_size
                                    for pos in range(start, end))
        data['query_lens'].append(count)
        data['context_lens'].append(end)
        data['cu_seqlens_q'].append(data['cu_seqlens_q'][-1] + count)
        data['cu_seqlens_k'].append(data['cu_seqlens_k'][-1] + end)
    width = max(len(item.block_table) for item in items)
    if decode_table_width is not None and all(length == 1 for length in data['query_lens']):
        if decode_table_width < width:
            raise ValueError('decode table width must cover all live blocks')
        width = decode_table_width
    data['block_tables'] = [list(item.block_table) + [0] * (width - len(item.block_table)) for item in items]
    return data
