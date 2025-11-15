"""
Fuse
    c = swiglu(a, b)
    y = c * s
where s has shape [1, hidden] that scales c of shape [batch, hidden]
"""

import torch
from torch.nn import functional as F

import triton
import triton.language as tl
from triton_utils import calculate_settings, silu


@triton.jit
def _swiglu_scale_kernel(
    a_ptr, b_ptr, s_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride
    # no need to advance `s_ptr`, because `s` only has `n_cols` elements

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    s_row = tl.load(s_ptr + col_offsets, mask=mask, other=0)  # every program load same `s` data
    c_row = silu(a_row).cast(b_row.dtype) * b_row * s_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def swiglu_scale_triton(a, b, s):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_scale_kernel[(n_rows,)](
        a,
        b,
        s,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c.view(*ori_shape)


def swiglu_scale_torch(a, b, s):
    return F.silu(a) * b * s
