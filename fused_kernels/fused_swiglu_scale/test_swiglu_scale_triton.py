import pytest
import torch
from swiglu_scale_triton_impl import swiglu_scale_triton, swiglu_scale_torch


@pytest.mark.parametrize(
    "batch,hidden",
    [
        (32, 1024),
        (15, 500),
        (77, 2171)
    ]
)
def test_swiglu_scale(
    batch,
    hidden,
    seed=0,
    dtype=torch.float16,
    device="cuda:0"
    ):
    torch.manual_seed(seed)
    a = torch.randn(batch, hidden, dtype=dtype, device=device)
    b = torch.randn(batch, hidden, dtype=dtype, device=device)
    s = torch.abs(torch.randn(1, hidden, dtype=dtype, device=device))  # scale can only be postive
    c_ref = swiglu_scale_torch(a, b, s)
    c = swiglu_scale_triton(a, b, s)

    torch.testing.assert_close(c, c_ref)
