
'''
written by SINQ authors
'''

import torch
import torch.nn as nn
import numpy as np
from torch import vmap
import torch.nn.functional as F
from .sinkhorn import *

NF4_CODEBOOK = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
     0.0,
     0.07958029955625534,
     0.16093020141124725,
     0.24611230194568634,
     0.33791524171829224,
     0.44070982933044434,
     0.5626170039176941,
     0.7229568362236023,
     1.0
], dtype=torch.float32)

# NF3 codebook (~[-1, 1], 8 non-uniform levels from the paper "Asymmetric Floating Point Quantization for LLMs")
NF3_CODEBOOK = torch.tensor([
    -1.0,
    -0.5350227355957031,
    -0.2469314038753510,
     0.0,
     0.1833375245332718,
     0.3819939494132996,
     0.6229856610298157,
     1.0
], dtype=torch.float32)

def quantize_rtn(
    matrix: torch.Tensor,
    min_max=[],
    niter=None,
    mode: str = "uniform",        # "uniform" | "nf4" | "nf3"
    nf_use_shift: bool = True,    # applies to both NF4 and NF3
    nf_shift_kind: str = "mean"   # "minmax" | "mean"
):
    w = matrix
    orig_dtype = w.dtype
    dev = w.device
    w = w.to(torch.float32)

    if mode.lower() in ("nf4", "nf3"):
        cb = NF4_CODEBOOK if mode.lower() == "nf4" else NF3_CODEBOOK

        if not nf_use_shift:
            scales = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
            zeros  = torch.zeros_like(scales)
            norm   = w / scales
        else:
            if nf_shift_kind == "minmax":
                w_max = w.amax(dim=1, keepdim=True)
                w_min = w.amin(dim=1, keepdim=True)
                denom  = (w_max - w_min).clamp_min(1e-4)
                scales = denom / 2.0
                zeros  = - (w_max + w_min) / denom
                norm   = w / scales + zeros
            elif nf_shift_kind == "mean":
                mu     = w.mean(dim=1, keepdim=True)
                x      = w - mu
                scales = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
                zeros  = (-mu / scales)
                norm   = w / scales + zeros
            else:
                raise ValueError("nf_shift_kind must be 'minmax' or 'mean'")

        cb = cb.to(dev, dtype=norm.dtype)
        q  = (norm.unsqueeze(-1) - cb.view(1,1,-1)).abs().argmin(dim=-1).to(torch.int8)
        return q.contiguous(), scales.to(orig_dtype), zeros.to(orig_dtype), orig_dtype

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = min_max[1]
    min_int = min_max[0]
    scales  = (max_val - min_val).clamp(min=1e-4) / max_int
    zeros   = -torch.round(min_val / scales)
    q       = torch.clamp(torch.round(w / scales + zeros), min_int, max_int).to(torch.int8)
    return q.contiguous(), scales.to(orig_dtype), zeros.to(orig_dtype), orig_dtype


def shrink_lp_op(x, beta, lp_norm):
    if lp_norm == 1:
        out = torch.abs(x)
        out.sub_(1.0 / beta).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out
    else:
        out = torch.abs(x).clip(1e-5, 1e5)
        out.sub_((1.0 / beta) * out.pow(lp_norm - 1)).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out

def optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta=1e1, lp_norm=.7, axis=1):
    W_q = torch.round(W_f * scale + zero).clamp_(min_max[0], min_max[1])
    W_r = (W_q - zero) / scale
    W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
    zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
    return W_r, W_q, zero, scale

def hqq_rtn(W, min_max=[]):
    dtype = W.dtype
    max_v = min_max[1]
    min_v = min_max[0]
    _max = W.max(axis=1, keepdim=True)[0]
    _min = W.min(axis=1, keepdim=True)[0]
    # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
    denom = (_max - _min)
    scale = (max_v / denom)
    scale = torch.where(denom.abs() <= 1e-4, torch.full_like(scale, 1.0), scale) #Avoid small denom values
    scale = scale.clamp(max=2e4) # clamp to avoid half-precision problems
    zero = -_min * scale
    W_q = (W * scale + zero).round_().clamp_(min_max[0], min_max[1])
    return W_q, scale, zero, dtype


def quantize_symmetric_rtn(matrix, min_max, niter=None):
    w = matrix
    dtype = w.dtype
    max_val = w.abs().amax(dim=1, keepdim=True)
    max_int = min_max[1]//2 - 1
    min_int = -min_max[1]//2
    scales = (max_val).clamp(min=1e-5) / max_int
    q = torch.round(w/scales).clip(min_int, max_int)
    return q, scales, dtype

def dequantize_symmetric_rtn(q, scales, dtype):
    return (q*scales).to(dtype)

def quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
    dtype = matrix.dtype
    dev = matrix.device
    matrix = matrix.float()

    # normalize the matrix with sinkhorn inspired std.dev. scaling
    # matrix, mu1, mu2 = min_kurt_vectors_vmap(matrix,32) # for kurtosis exp.
    matrix, mu1, mu2 = sinkhorn_log(matrix, 16)

    if not ('sinq' in method):
        matrix = matrix * mu1 * mu2
        mu1 = torch.ones_like(mu1)
        mu2 = torch.ones_like(mu2)

    if 'awq' in method and awq_scale is not None:
        matrix = matrix * awq_scale
        mu1 = mu1 / awq_scale.float()


    if not ('hqq' in method):
        if 'noz' in method:
            q, scales, _ = quantize_symmetric_rtn(matrix, min_max)
            q = q + min_max[1]//2
            z = torch.tensor(min_max[1] // 2) # note that this is a single constant
        else:
            if "nf4" in method.lower():
                q, scales, z, _ = quantize_rtn(matrix, min_max, mode="nf4")
            elif "nf3" in method.lower():
                q, scales, z, _ = quantize_rtn(matrix, min_max, mode="nf3")
            else:
                q, scales, z, _ = quantize_rtn(matrix, min_max, mode="uniform")

    if 'hqq' in method:
        assert not ('noz' in method), 'noz incompatible with hqq'
        # alternative approach, use hqq after normalization
        q, scales, z, _ = hqq_rtn(matrix, min_max)
        best_error = torch.inf
        best_z = torch.zeros_like(z)
        best_scales = torch.ones_like(scales)
        for i in range(20):
            W_r, W_q, z, scales = optimize_weights_proximal_legacy_step(matrix, scales.clip(1e-5,1e5), z, min_max)
            current_error = torch.abs(matrix - W_r).mean().float()
            take = current_error < best_error
            best_error  = torch.where(take, current_error, best_error)
            best_z      = torch.where(take[..., None],           z, best_z)
            best_scales = torch.where(take[..., None],      scales, best_scales)

        scales = best_scales
        z = best_z
        q = W_q
        scales = 1/scales

    scales2 = torch.ones(1,matrix.shape[1]).to(dev).to(mu1.dtype) * mu1
    scales = scales*mu2

    q = q.to(dtype).to(dev)
    s1 = (scales).to(dtype) 
    s2 = (scales2).to(dtype) 
    z = (z).to(dtype).to(dev)

    return q, s1.to(dev), s2.to(dev), z
 

def tiled_quant_rectangle(M, min_max, block, method='dual', awq_scale=None):
    q = quantize_dual_scale_shift
    mshape = M.shape
    H,W = M.shape
    assert W%block==0, 'block must divide W'
    n_w = W//block

    M = M.view(H, W//block, block)
    M_batched = M.permute(1,0,2).contiguous().view(n_w, H, block)

    if not (awq_scale is None):
        awq_scale = awq_scale.view(n_w, 1, block)
        def process_block(mat, awq_scale):
            return q(mat, min_max, method=method, awq_scale=awq_scale) 
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched, awq_scale)
    else:
        def process_block(mat):
            return q(mat, min_max, method=method) 
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched)

    if 'noz' in method:
        z = z.view(-1)[0]
    else:
        z = z.permute(1,0,2).reshape(-1,1)

    Q = Q.permute(1,0,2).reshape(-1, block)
    s1 = s1.permute(1,0,2).reshape(-1,1)
    s2 = s2.permute(1,0,2).view(1, -1)
    return Q, s1, s2, z


def tiled_quant_square(M, min_max, block, method='dual', awq_scale=None):
    q = quantize_dual_scale_shift
    H,W = M.shape
    assert H%block==0 and W%block==0, 'block must divie H and W'
    n_w = W//block
    n_h = H//block

    M = M.view(H//block, block, W//block, block)
    M_batched = M.permute(0, 2, 1, 3).contiguous().view(-1, block, block)

    if not (awq_scale is None):
        awq_scale = awq_scale.view(1, 1, n_w, block).repeat(n_h, 1, 1, 1).permute(0,2,1,3).contiguous().view(-1, 1, block)
        def process_block(mat, awq_scale):
            return q(mat, min_max, method=method, awq_scale=awq_scale) 
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched, awq_scale)
    else:
        def process_block(mat):
            return q(mat, min_max, method=method) 
        Q, s1, s2, z = vmap(process_block, randomness='different')(M_batched)


    if 'noz' in method:
        z = z.view(-1)[0]
    else:
        z = z.view(-1,block,1)

    s1 = s1.view(-1,block,1)
    s2 = s2.view(-1,1,block)
    Q = Q.view(n_h, n_w, block, block).permute(0, 2, 1, 3).contiguous()
    Q = Q.view(-1, block)
    return Q, s1, s2, z