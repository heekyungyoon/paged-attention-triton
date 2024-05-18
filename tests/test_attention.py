import pytest
import torch

from paged_attention.naive_attention import attention


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
