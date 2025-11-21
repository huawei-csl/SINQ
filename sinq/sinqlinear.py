# modified by SINQ authors 2025

import copy
from typing import Union, Optional, List

import torch
from torch import nn, Tensor

from termcolor import colored  # for warning in sinq_base_quant_config
from .quantizer import Quantizer, dq8
from .utils import is_divisible
from .bitpack import BitPack  # for unpacking 4-bit

# GemLite
try:
    import gemlite
    from gemlite.dtypes import TORCH_TO_DTYPE, DType
    _HAVE_GEMLITE = True
    print('Found gemlite installation, fast SINQ-ference for 4-bit models')
except Exception:
    print('No gemlite installation, slow inference! Check the SINQ repo for more info!')
    _HAVE_GEMLITE = False


class SINQLinear(nn.Module):
    """
    Linear that can run either:
      - PyTorch (dequantize + matmul), or
      - GemLite backend via gemlite.core.forward_functional (no persistent GemLite module)

    We auto-enable GemLite for 4-bit, 1D tiling, unless 'nogemlite'
    appears in the quant method string.
    """

    def __init__(
        self,
        linear_layer: Union[nn.Module, None],
        quant_config: Optional[dict] = None,
        del_orig: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        use_unpack_kernel: bool = False,
        layer_activations: Optional[Tensor] = None,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.quant_config = copy.deepcopy(quant_config) if quant_config is not None else None
        self.del_orig = del_orig
        self.use_unpack_kernel = use_unpack_kernel

        self.linear_layer = linear_layer
        self.layer_activations = layer_activations

        self.bias: Optional[Tensor] = None
        self.axis = None
        self.channel_wise = None

        # Decide backend (GemLite vs PyTorch)
        qc = (self.quant_config or {}).get('weight_quant_params', {})
        can_use_gemlite = (
            qc.get('nbits', None) == 4
            and qc.get('tiling_mode', None) == '1D'
            and 'nogemlite' not in str(qc.get('method', '')).lower()
            and self.device.type == "cuda"
            and _HAVE_GEMLITE
        )

        if can_use_gemlite and (self.linear_layer is not None):
            out_f = getattr(self.linear_layer, "out_features", None)
            in_f = getattr(self.linear_layer, "in_features", None)
            if out_f is None or in_f is None or out_f < 16 or in_f < 16:
                can_use_gemlite = False
        
        if can_use_gemlite:
            self.use_gemlite = True
            self.set_forward_backend("gemlite")
        else:
            #print(f'The linear layer im considering in pytorch is: {self.linear_layer}', flush=True)
            self.use_gemlite = False
            self.set_forward_backend("pytorch")

        self.W_q: Optional[Tensor] = None
        self.meta: Optional[dict] = None
        self.ready: bool = False

        # Buffers for GemLite functional call
        self._gl_tensor_args: Optional[List[Tensor]] = None
        self._gl_meta_args: Optional[List[int]] = None
        self._gl_bias: Optional[Tensor] = None
        self._gl_scale2: Optional[Tensor] = None
        self._gl_input_torch_dtype = None
        self._gemlite_ready: bool = False

        # Quantize now if given a dense layer + config; else expect load_state_dict later
        if (self.linear_layer is not None) and (self.quant_config is not None):
            self.initialize()
            if self.use_gemlite:
                desired = getattr(self, "compute_dtype", torch.bfloat16)
                self._prepare_gemlite_args(desired)
            self.ready = True

    # ---------- Initialization (shared) ----------
    def initialize(self):
        assert self.linear_layer is not None, "linear_layer must be provided for initialization"
        assert self.quant_config is not None, "quant_config must be provided for initialization"

        # Handle group_size==None
        if self.quant_config["weight_quant_params"]["group_size"] is None:
            self.quant_config["weight_quant_params"]["group_size"] = (
                self.linear_layer.in_features
                if (self.quant_config["weight_quant_params"]["axis"] == 1)
                else self.linear_layer.out_features
            )

        # Quantize weights using Quantizer
        W_q, meta = Quantizer.quantize(
            self.linear_layer.weight.data,
            self.layer_activations,
            device=self.device,
            compute_dtype=self.compute_dtype,
            **self.quant_config['weight_quant_params'],
            use_unpack_kernel=self.use_unpack_kernel,
            bitpack=True,
        )

        self.W_q = W_q
        self.meta = meta

        # Bias
        self.bias = (
            None if (self.linear_layer.bias is None)
            else self.linear_layer.bias.clone().to(device=self.device, dtype=self.compute_dtype)
        )

        # Clear original parameters if requested
        if self.del_orig:
            for name, _ in self.linear_layer.named_parameters():
                setattr(self.linear_layer, name, None)
            del self.linear_layer
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # For GemLite path we REQUIRE scale2
        if self.use_gemlite and "scale2" not in self.meta:
            raise RuntimeError("GemLite backend requires meta['scale2'] from the quantizer.")

    # ---------- GemLite argument prep (no persistent GemLite module) ----------
    def _prepare_gemlite_args(self, desired_torch_dtype: torch.dtype):
        assert _HAVE_GEMLITE
        assert self.meta is not None and self.W_q is not None

        m = self.meta
        nbits = int(m.get("nbits", 4))
        N, K = int(m["shape"][0]), int(m["shape"][1])

        # ---- Sanity for old checkpoints: stale 'packing' flag ----
        numel = int(self.W_q.numel())
        expected_unpacked = N * K
        expected_packed   = expected_unpacked * nbits // 8
        if m.get("packing"):
            if numel == expected_unpacked:
                # Already unpacked but flagged as packed -> drop the flag
                m.pop("packing", None)
            elif numel != expected_packed:
                raise RuntimeError(
                    f"W_q.numel()={numel} not matching packed={expected_packed} or unpacked={expected_unpacked} "
                    f"for shape {N}x{K}, nbits={nbits}"
                )

        # ---- Map runtime dtype ----
        in_dtype  = TORCH_TO_DTYPE.get(desired_torch_dtype, TORCH_TO_DTYPE[torch.float16])
        out_dtype = TORCH_TO_DTYPE.get(desired_torch_dtype, TORCH_TO_DTYPE[torch.float16])

        # ---- Dequant meta to runtime dtype (not touching W_q) ----
        scales = dq8(m["scale"]).to(self.device, dtype=desired_torch_dtype)
        zeros  = dq8(m["zero"]).to(self.device, dtype=desired_torch_dtype)
        bias   = None if self.bias is None else self.bias.to(self.device, dtype=desired_torch_dtype)

        # ---- TEMPORARY unpack for GemLite only ----
        if m.get("packing"):
            W_for_gl = Quantizer.unpack[m["packing"]](self.W_q, dtype=torch.uint8)
        else:
            # If someone saved unpacked 4-bit already (numel == N*K), just ensure u8
            if self.W_q.dtype != torch.uint8:
                # (shouldn’t happen for 4-bit, but keep it robust)
                W_for_gl = self.W_q.to(torch.uint8)
            else:
                W_for_gl = self.W_q

        with torch.cuda.device(self.device):
            gl = gemlite.GemLiteLinear(
                nbits, group_size=int(m["group_size"]), in_features=K, out_features=N,
                input_dtype=in_dtype, output_dtype=out_dtype, scaled_activations=False
            )
            gl.pack(W_for_gl, scales, zeros, bias)

            tensor_args = gl.get_tensor_args()
            meta_args   = gl.get_meta_args()

        # ---- Stash GemLite call args (these are copies owned by us/GemLite) ----
        self._gl_tensor_args = [t.to(self.device) for t in tensor_args]
        self._gl_meta_args   = [int(v) for v in meta_args]
        self._gl_bias        = bias
        self._gl_scale2      = m["scale2"].to(self.device, dtype=desired_torch_dtype).contiguous()
        self._gl_input_torch_dtype = desired_torch_dtype
        self._gemlite_ready  = True

        # Free temps ASAP
        self._offload_originals_post_pack()

    def _offload_originals_post_pack(self):
        # Offload quantized weights and meta to CPU; runtime only needs GemLite buffers.
        if self.W_q is not None and self.W_q.is_cuda:
            self.W_q = self.W_q.cpu()
        if isinstance(self.meta, dict):
            for k, v in list(self.meta.items()):
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    self.meta[k] = v.cpu()
        torch.cuda.empty_cache()

    # ---------- Forwards ----------
    def forward_gemlite(self, x: torch.Tensor) -> torch.Tensor:
        runtime_dev = x.device
        runtime_dtype = x.dtype

        if runtime_dev.type != "cuda":
            raise RuntimeError(f"GemLite backend requires CUDA tensors; got input on device: {runtime_dev}")

        need_rebuild = (
            not getattr(self, "_gemlite_ready", False)
            or any(t.device != runtime_dev for t in (self._gl_tensor_args or []))
            or (getattr(self, "_gl_scale2", None) is not None and self._gl_scale2.device != runtime_dev)
            or (getattr(self, "_gl_bias", None) is not None and self._gl_bias is not None and self._gl_bias.device != runtime_dev)
            or (torch.device(getattr(self, "device", None)) != runtime_dev)
            or (getattr(self, "_gl_input_torch_dtype", None) != runtime_dtype)
        )
        if need_rebuild:
            print('Rebuilding...')
            self.device = runtime_dev
            self._gemlite_ready = False
            self._prepare_gemlite_args(runtime_dtype)  # <— build for the dtype we *actually* have
        
        self.device = torch.device(self.device)
        with torch.cuda.device(self.device):
            # self._gl_scale2 already matches x.dtype, so no silent upcast to fp32

            return gemlite.forward_functional(
                self._gl_scale2 * x,
                self._gl_bias,
                self._gl_tensor_args,
                self._gl_meta_args,
                -1,
            )
        
    def forward_pytorch(self, x: Tensor) -> Tensor:
        out = torch.matmul(x, self.dequantize().t())
        if self.bias is not None:
            out += self.bias
        return out

        #dev = x.device
        #print(f'The device in pytorch forward is: {dev}', flush=True)
        #print(f'The linear layer im considering is {self}', flush=True)
        #W = self.dequantize().to(dev)
        #out = torch.matmul(x, W.t())
        #if self.bias is not None:
        #    out += self.bias.to(dev)
        #return out

    def set_forward_backend(self, backend: str):
        if backend == "pytorch":
            self._forward_impl = self.forward_pytorch
        elif backend == "gemlite":
            self._forward_impl = self.forward_gemlite
        else:
            raise ValueError(f"Backend {backend} not supported")

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # ---------- Utilities ----------
    def unpack(self, reshape=False, dtype=None):
        if self.ready is False:
            return None
        if self.meta is not None and self.meta.get("packing", None):
            W_r = Quantizer.unpack[self.meta["packing"]](
                self.W_q, dtype=dtype if (dtype is not None) else self.compute_dtype
            )
            return W_r.view(self.meta["shape"]) if (reshape) else W_r

    def dequantize(self):
        assert self.ready, "model was not quantized"
        W_q, meta = self.W_q, self.meta
        device = W_q.device

        del_keys = set()

        # Zero/Scale packed together
        if "zero_scale" in meta:
            zero_scale = meta["zero_scale"].to(device=device)
            if zero_scale.dtype == torch.uint8:
                meta["zero_q"], meta["scale_q"] = zero_scale[0], zero_scale[1]
                del_keys.update({"zero_q", "scale_q"})
            else:
                meta["zero"], meta["scale"] = zero_scale[0], zero_scale[1]
                del_keys.update({"zero", "scale"})

        if meta.get("quant_zero", False):
            meta["zero"] = Quantizer.dequantize(meta["zero_q"].to(device=device), meta["meta_zero"])
            del_keys.add("zero")

        if meta.get("quant_scale", False):
            meta["scale"] = Quantizer.dequantize(meta["scale_q"].to(device=device), meta["meta_scale"])
            del_keys.add("scale")

        W_est = Quantizer.dequantize(W_q, meta, use_unpack_kernel=self.use_unpack_kernel)

        for key in del_keys:
            del meta[key]
        return W_est

    def extra_repr(self) -> str:
        out = ""
        if getattr(self, "meta", None) is not None:
            in_features, out_features = self.meta["shape"][::-1]
            wdev = getattr(getattr(self, "W_q", None), "device", None)
            wdev = str(wdev) if wdev is not None else "None"
            out = (
                f"in_features={in_features}, out_features={out_features}, bias={self.bias is not None}, "
                f"device={self.device}, W_q.device={wdev}"
            )
        return out

    def _meta_to_cpu(self, meta: dict) -> dict:
        if meta is None:
            return None

        def to_cpu(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu()
            if isinstance(v, dict):
                return {k: to_cpu(vi) for k, vi in v.items()}
            if isinstance(v, (list, tuple)):
                # Detect quantAux 4-tuple: (x, s, m, shape)
                if (
                    len(v) == 4
                    and isinstance(v[0], torch.Tensor)
                    and isinstance(v[1], torch.Tensor)
                    and isinstance(v[2], torch.Tensor)
                ):
                    x, s, m, shape = v
                    return {"x": to_cpu(x), "s": to_cpu(s), "m": to_cpu(m), "shape": list(shape)}
                return [to_cpu(e) for e in v]
            return v

        return {k: to_cpu(v) for k, v in meta.items()}

    def state_dict(self, destination=None, prefix: str = '', keep_vars: bool = False):
        """
        Export quantized tensors for saving:
          - W_q (Tensor)
          - bias (Tensor or omitted if None)
          - meta (dict; tensors moved to CPU)
        """
        sd = {}
        if self.W_q is not None:
            sd["W_q"] = self.W_q.detach().cpu()
        if self.bias is not None:
            sd["bias"] = self.bias.detach().cpu()
        if self.meta is not None:
            sd["meta"] = self._meta_to_cpu(self.meta)
        return sd
    
    def _can_use_gemlite_from_meta(self) -> bool:
        if not _HAVE_GEMLITE:
            return False
        if not isinstance(self.meta, dict):
            return False
        m = self.meta
        # Must be CUDA, 4-bit, 1D groups along K, and have scale2
        if str(getattr(self.device, "type", self.device)) != "cuda":
            return False
        if m.get("nbits", 4) != 4:
            return False
        if m.get("tiling_mode", "1D") != "1D":
            return False
        if m.get("axis", 1) != 1:  # GemLite expects groups along K (in_features)
            return False
        if "scale2" not in m:
            return False

        # shape sanity: GemLite kernels need N, K >= 16
        if "shape" not in m:
            return False
        N, K = map(int, m["shape"])
        if N < 16 or K < 16:
            return False

        # group_size sanity
        G = int(m.get("group_size", 0) or 0)
        return (G > 0) and (K % G == 0)


    def load_state_dict(self, state_dict, strict: bool = True):
        self.W_q = state_dict["W_q"].to(device=self.device)
        self.meta = state_dict["meta"]

        b = state_dict.get("bias", None)
        self.bias = b.to(device=self.device, dtype=self.compute_dtype) if b is not None else None

        if isinstance(self.meta, dict) and "shape" in self.meta:
            N, K = map(int, self.meta["shape"])
            nbits = int(self.meta.get("nbits", 4))
            numel = int(self.W_q.numel())
            expected_unpacked = N * K
            expected_packed   = expected_unpacked * nbits // 8
            if self.meta.get("packing") and numel == expected_unpacked:
                self.meta.pop("packing", None)

        self.ready = True

        # Auto-switch to GemLite if possible
        if self._can_use_gemlite_from_meta():
            self.use_gemlite = True
            self.set_forward_backend("gemlite")
            desired = getattr(self, "compute_dtype", torch.bfloat16)
            self._prepare_gemlite_args(desired)
        else:
            self.use_gemlite = False
            self.set_forward_backend("pytorch")

        from torch.nn.modules.module import _IncompatibleKeys
        return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])


def sinq_base_quant_config(
    nbits: int = 4,
    group_size: int = 64,
    quant_zero: bool = False,
    quant_scale: bool = False,
    view_as_float: bool = False,
    axis: int = 1,
    tiling_mode: str = '1D',
    method: str = 'dual',
):
    assert (nbits in Quantizer.SUPPORTED_BITS), "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
    if method == "asinq":
        # Remap sinq_awq_l1_quantAux to behave like asinq (A-SINQ in the paper)
        method = "sinq_awq_l1_quantAux"
    elif method == "sinq":
        # Remap so that users can use sinq_quantAux as sinq (scales and zeros are quantized to 8-bit)
        method = "sinq_quantAux"
    if group_size is not None:
        assert is_divisible(group_size, 8), "Invalid group_size param: the value should be a multiple of 8."

    weight_quant_params = {
        "nbits": nbits,
        "group_size": group_size,
        "round_zero": True if nbits == 4 else False,
        "axis": axis,
        "view_as_float": view_as_float,
        "tiling_mode": tiling_mode,
        "method": method,
    }

    if quant_zero or quant_scale:
        print(colored(
            "Warning: Quantized meta-data is deprecated and will be removed. It is not supported for quantized model serialization.",
            "yellow",
        ))

    scale_quant_params = ({"nbits": 8, "group_size": 128} if (quant_scale) else None)
    zero_quant_params = ({"nbits": 8, "group_size": None} if (quant_zero) else None)

    return {
        "weight_quant_params": weight_quant_params,
        "scale_quant_params": scale_quant_params,
        "zero_quant_params": zero_quant_params
    }


# Alias
BaseQuantizeConfig = sinq_base_quant_config
