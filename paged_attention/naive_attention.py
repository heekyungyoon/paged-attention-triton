import math
import torch

import triton
import triton.language as tl


@triton.jit
def _forward(
    output_ptr: tl.tensor,
    query_ptr: tl.tensor,
    key_ptr: tl.tensor,
    value_ptr: tl.tensor,
    y_size: tl.int32,
    x_size: tl.int32,
    head_stride: tl.int32,
    y_stride: tl.int32,
    x_stride: tl.int32,
    mask_ptr: tl.tensor,
    mask_head_stride: tl.int32,
    mask_y_stride: tl.int32,
    mask_x_stride: tl.int32,
    is_causal: tl.constexpr,
    softmax_scale: tl.float32,
    dtype: tl.constexpr,
    y_block_size: tl.constexpr,
    x_block_size: tl.constexpr,
):
    log2e = tl.constexpr(1.4426950408889634)
    pid = tl.program_id(0)
    num_y_blocks = tl.cdiv(y_size, y_block_size)
    head = pid // num_y_blocks
    y_block = pid % num_y_blocks
    head_offset = head * head_stride
    y_offset = y_block * y_block_size

    output_block_ptr = tl.make_block_ptr(
        output_ptr + head_offset,
        shape=(y_size, x_size),
        strides=(y_stride, x_stride),
        offsets=(y_offset, 0),
        block_shape=(y_block_size, x_block_size),
        order=(1, 0),
    )
    query_block_ptr = tl.make_block_ptr(
        query_ptr + head_offset,
        shape=(y_size, x_size),
        strides=(y_stride, x_stride),
        offsets=(y_offset, 0),
        block_shape=(y_block_size, x_block_size),
        order=(1, 0),
    )
    key_block_ptr = tl.make_block_ptr(
        key_ptr + head_offset,
        shape=(x_size, y_size),
        strides=(x_stride, y_stride),
        offsets=(0, 0),
        block_shape=(x_block_size, y_block_size),
        order=(0, 1),
    )
    value_block_ptr = tl.make_block_ptr(
        value_ptr + head_offset,
        shape=(y_size, x_size),
        strides=(y_stride, x_stride),
        offsets=(0, 0),
        block_shape=(y_block_size, x_block_size),
        order=(1, 0),
    )

    if mask_ptr is not None:
        mask_block_ptr = tl.make_block_ptr(
            mask_ptr + head * mask_head_stride,
            shape=(y_size, y_size),
            strides=(mask_y_stride, mask_x_stride),
            offsets=(y_offset, 0),
            block_shape=(y_block_size, y_block_size),
            order=(1, 0),
        )

    query = tl.load(query_block_ptr)
    score_scale = (softmax_scale * log2e).to(dtype)
    query *= score_scale
    max = tl.full((y_block_size,), float("-inf"), tl.float32)
    sum = tl.zeros((y_block_size,), tl.float32)
    output = tl.zeros((y_block_size, x_block_size), dtype)
    m_offsets = tl.arange(0, y_block_size) + y_offset

    if is_causal:
        n_size = y_offset + y_block_size
    else:
        n_size = y_size

    for n_offset in range(0, n_size, y_block_size):
        score = tl.zeros((y_block_size, y_block_size), dtype)

        if is_causal:
            n_offsets = tl.arange(0, y_block_size) + n_offset
            condition = m_offsets[:, None] >= n_offsets[None, :]
            score = tl.where(condition, score, float("-inf"))
        elif mask_ptr is not None:
            mask = tl.load(mask_block_ptr)
            mask *= log2e
            score += mask

        key = tl.load(key_block_ptr)
        score += tl.dot(query, key)
        peak = tl.maximum(max, tl.max(score, 1))
        alpha = tl.math.exp2(max - peak)
        beta = tl.math.exp2(score - peak[:, None])
        sum = sum * alpha + tl.sum(beta, 1)
        max = peak
        output *= alpha[:, None].to(dtype)
        value = tl.load(value_block_ptr)
        output += tl.dot(beta.to(dtype), value)
        key_block_ptr = tl.advance(key_block_ptr, (0, y_block_size))
        value_block_ptr = tl.advance(value_block_ptr, (y_block_size, 0))

        if mask_ptr is not None:
            mask_block_ptr = tl.advance(mask_block_ptr, (0, y_block_size))

    output /= sum[:, None].to(dtype)

    tl.store(output_block_ptr, output.to(dtype))


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale=None):
        sm_scale = 1 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)

        num_batches, num_heads, y_size, x_size = q.shape
        mask = None
        def grid(meta):
            num_m_blocks = triton.cdiv(y_size, meta["y_block_size"])
            return (num_batches * num_heads * num_m_blocks,)
        _forward[grid](
            o,
            q,
            k,
            v,
            y_size,
            x_size,
            q.stride(1),
            q.stride(2),
            q.stride(3),
            mask,
            mask.stride(1) if mask is not None else 0,
            mask.stride(2) if mask is not None else 0,
            mask.stride(3) if mask is not None else 0,
            causal,
            sm_scale,
            tl.float32,
            x_block_size=triton.next_power_of_2(x_size),
            y_block_size=32,
        )

        return o


attention = Attention.apply