# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
# modified by SINQ authors 2025
#####################################################
import torch
import numpy as np
from torch import float32, float16, Tensor
from functools import partial
from typing import Union
from .dual_shift import *
from .awq import *

# Shrinking operator
def shrink_lp_op(x: Tensor, beta: float, lp_norm: float) -> Tensor:
    if lp_norm == 1:
        #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
        out = torch.abs(x)
        out.sub_(1.0 / beta).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out
    else:
        #torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1))
        out = torch.abs(x)
        out.sub_((1.0 / beta) * out.pow(lp_norm - 1)).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out
    
def kurtosis(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Standard kurtosis: E[(x-mu)^4] / (E[(x-mu)^2])^2
    Returns a scalar tensor.
    """
    x = x.float().reshape(-1)
    mu = x.mean()
    xc = x - mu
    v = (xc * xc).mean().clamp_min(eps)
    m4 = (xc.pow(4)).mean()
    return m4 / (v * v)

def dequantize_dual_scale_shift(quantized_matrix, scales, scales2, shifts):
    return ((quantized_matrix - shifts) * scales * scales2)

# Proximal solver || W - dequantize(quantize(W))||_p^p
#@torch.compile(fullgraph=True)
def optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta, lp_norm, axis, scale2=None, shape=None):
    if scale2 is None:
        W_q = torch.round(W_f * scale + zero).clamp_(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
    else:
        scale = 1/scale
        scale2 = 1/scale2
        W_q = torch.round( ((W_f / scale).view(shape) / scale2).reshape(-1,W_f.shape[1]) + zero).clamp_(min_max[0], min_max[1])
        W_r = (((W_q - zero) * scale).view(shape) * scale2).view(-1, W_f.shape[1])
        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero_expanded = ((W_q - (W_f - W_e)/ scale).view(shape) / scale2).view(-1, W_f.shape[1])
        zero = torch.mean(zero_expanded, axis=axis, keepdim=True)
        scale = 1/scale
    return W_r, W_q, zero, scale


@torch.inference_mode()
def optimize_weights_proximal_legacy(
    tensor: Tensor,
    layer_activations,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: Union[str, None] = "cpu",
    opt_params: dict = {"lp_norm": .7, "beta": 1e1, "kappa": 1.01, "iters": 20, "kurtosis_threshold": 5.0},
    verbose: bool = False,
    shape = None,
    tiling_mode = '1D',
    method = 'dual'
) -> tuple:
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )
    dtype = float32 if (device == "cpu") else float16  
    W_f   = tensor.to(dtype=dtype, device=device)
    scale = scale.to(dtype=dtype, device=device)
    zero  = zero.to(dtype=dtype, device=device)
    # DEBUG
    # print("in optimize_weights_proximal_legacy; device, tensor.device: ", device, tensor.device)
    tile = W_f.shape[-1]

    if not (layer_activations is None):
        awq_scale = compute_awq_scale(W_f.view(shape), layer_activations, min_max=min_max, tile=tile, method=method)
        awq_scale = awq_scale.unsqueeze(0).to(W_f.device).to(W_f.dtype)
    else: 
        awq_scale = None

    # ---- Kurtosis gate ----
    kurt_th = opt_params.get("kurtosis_threshold", None)
    if kurt_th is not None and ("sinq" in str(method).lower()):
        k0 = float(kurtosis(W_f.view(shape)))
        if k0 > float(kurt_th):
            method = "rtn"
            if verbose:
                print(f"[SINQ] kurtosis={k0:.2f} > {float(kurt_th):.2f} -> forcing method='{method}'")

    # -----------------------
    assert axis==1, 'only supports axis 1 right now'
    if tiling_mode == '1D':
        q, s1, s2, z= tiled_quant_rectangle(W_f.reshape(shape), min_max, tile, method, awq_scale)
    elif tiling_mode == '2D':
        q, s1, s2, z= tiled_quant_square(W_f.reshape(shape), min_max, tile, method, awq_scale)

    torch.cuda.empty_cache()

    awq_scale = None # the awq scale is absorbed in s1

    return q, s1, z, s2, awq_scale

# Default: fast with early stopping
optimize_weights_proximal = optimize_weights_proximal_legacy
