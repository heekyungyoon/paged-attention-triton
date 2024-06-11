import pytest
import torch

from paged_attention.naive_attention import attention
from paged_attention.attention_kernel import paged_attention, paged_attention_torch


@pytest.mark.parametrize(
    "num_batches, num_heads, y_size, x_size, is_causal", [(4, 8, 128, 64, True), (4, 8, 256, 32, False)]
)
def test_forward(num_batches, num_heads, y_size, x_size, is_causal):
    query = torch.randn(num_batches, num_heads, y_size, x_size, device='cuda')
    key = torch.randn_like(query, device='cuda')
    value = torch.randn_like(query, device='cuda')

    assert torch.allclose(
        torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal),
        attention(query, key, value, is_causal),
        rtol=1e-2,
        atol=1e-3,
    )

@pytest.mark.parametrize(
    "num_batches, num_heads, y_size, x_size, is_causal", [(4, 8, 128, 64, False)]
)
def test_paged_attention(num_batches, num_heads, y_size, x_size, is_causal):
    import random

    num_blocks_in_cache = 8
    block_size = 2
    max_seq_len = y_size
    head_size = x_size
    num_seqs = num_batches
    num_query_heads = num_heads
    scale = float(1.0 / (head_size**0.5))

    query = torch.randn(num_batches, num_heads, y_size, x_size, device='cuda')
    # key = torch.randn_like(query, device='cuda')
    # value = torch.randn_like(query, device='cuda')
    output = torch.empty_like(query, device="cuda")
    output_torch = torch.empty_like(query, device="cuda")

    cache_shape = (num_blocks_in_cache, num_query_heads, head_size, block_size)

    key_cache = torch.empty(cache_shape, dtype=torch.float32, device="cuda")
    key_cache.uniform_(-scale, scale)
    assert key_cache.stride(0) == num_query_heads * head_size * block_size

    value_cache = torch.empty(cache_shape, dtype=torch.float32, device="cuda")
    value_cache.uniform_(-scale, scale)

    context_lens = torch.tensor(
        [random.randint(1, max_seq_len) for _ in range(num_seqs)], device="cuda"
    )
    context_lens[-1] = max_seq_len

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = [
        [
            random.randint(0, num_blocks_in_cache - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        for _ in range(num_seqs)
    ]
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # create tensor of all 0s of size 16
    debug_block_idxs = torch.zeros(
        max_num_blocks_per_seq, dtype=torch.int, device="cuda"
    )
    debug_key_cache_load = torch.zeros(
        max_num_blocks_per_seq, dtype=torch.float32, device="cuda"
    )
    debug_key_cache_load2 = torch.zeros(
        max_num_blocks_per_seq,
        head_size,
        block_size,
        dtype=torch.float32,
        device="cuda",
    )
    debug_block_idx_ptr2 = torch.zeros(1, dtype=torch.int, device="cuda")
    debug_key_cache_load3 = torch.zeros(head_size, dtype=torch.float32, device="cuda")
    debug_key_cache_load4 = torch.zeros(head_size, dtype=torch.float32, device="cuda")
    debug_key_cache_load5 = torch.zeros(head_size, dtype=torch.float32, device="cuda")
    debug_scores = torch.zeros(max_seq_len, dtype=torch.float32, device="cuda")
    debug_softmax = torch.zeros(max_seq_len, dtype=torch.float32, device="cuda")
    debug_output_ptr = torch.zeros(head_size, dtype=torch.float32, device="cuda")

    scratchpad_key = torch.zeros(
        (num_seqs, max_seq_len, num_query_heads, head_size),
        dtype=torch.float32,
        device="cuda",
    )
    scratchpad_value = torch.zeros_like(scratchpad_key)

    paged_attention[(num_seqs, num_query_heads)](
        debug_block_idxs_ptr=debug_block_idxs,
        debug_key_cache_load_ptr=debug_key_cache_load,
        debug_key_cache_load_ptr2=debug_key_cache_load2,
        debug_block_idx_ptr2=debug_block_idx_ptr2,
        debug_key_cache_load_ptr3=debug_key_cache_load3,
        debug_key_cache_load_ptr4=debug_key_cache_load4,
        debug_key_cache_load_ptr5=debug_key_cache_load5,
        debug_scores_ptr=debug_scores,
        debug_softmax_ptr=debug_softmax,
        debug_output_ptr=debug_output_ptr,

        scratchpad_key_ptr=scratchpad_key,
        scratchpad_value_ptr=scratchpad_value,
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        context_lens_ptr=context_lens,
        scale=scale,
        num_seqs=num_seqs,
        num_heads=num_query_heads,
        cache_block_stride=key_cache.stride(0),
        MAX_CONTEXT_LEN=max_seq_len,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        MAX_NUM_BLOCKS_PER_SEQ=max_num_blocks_per_seq,
    )

    output_torch = paged_attention_torch(query, key_cache, value_cache, block_tables, context_lens, scale, num_seqs,
                                         num_heads, key_cache.stride(0), max_seq_len, block_size, head_size)
    assert torch.allclose(
        output,
        output_torch,
        rtol=1e-2,
        atol=1e-3,
    )
