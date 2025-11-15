import pytest
import torch
from swiglu_triton_impl import swiglu_triton, swiglu_torch


@pytest.mark.parametrize(
    "batch,hidden",
    [
        (32, 1024),
        (15, 500),
        (77, 2171)
    ]
)
def test_swiglu(
    batch,
    hidden,
    seed=0,
    dtype=torch.float16,
    device="cuda:0"
    ):
    torch.manual_seed(seed)
    a = torch.randn(batch, hidden, dtype=dtype, device=device)
    b = torch.randn(batch, hidden, dtype=dtype, device=device)
    c_ref = swiglu_torch(a, b)
    c = swiglu_triton(a, b)

    torch.testing.assert_close(c, c_ref)
