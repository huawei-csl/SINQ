# modified by SINQ authors 2025

import copy
from typing import Optional, List

import torch
from torch import nn, Tensor

from termcolor import colored
from .quantizer import Quantizer, dq8
from .utils import is_divisible
from .bitpack import BitPack

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
    Quantized Linear layer for SINQ.

    Can run either:
      - PyTorch backend (dequantize + matmul), or
      - GemLite backend via gemlite.core.forward_functional (faster 4-bit inference)

    Supports two initialization patterns:

    1. **Legacy pattern** (backwards compatible): Pass a linear_layer + quant_config
       to quantize immediately in the constructor.

       >>> sinq = SINQLinear(linear_layer=dense, quant_config=config)

    2. **Two-phase pattern** (recommended for HF integration): Create an empty layer
       with shape info, then call quantize() later with the weight tensor.

       >>> sinq = SINQLinear(in_features=768, out_features=3072, quant_config=config)
       >>> sinq.quantize(weight_tensor)
    """

    def __init__(
        self,
        # New API: accept dimensions directly
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = False,
        # Legacy API: accept linear_layer (backwards compatible)
        linear_layer: Optional[nn.Module] = None,
        # Common parameters
        quant_config: Optional[dict] = None,
        del_orig: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        use_unpack_kernel: bool = False,
        layer_activations: Optional[Tensor] = None,
    ):
        super().__init__()

        # Store shape info
        self._in_features = in_features
        self._out_features = out_features
        self._has_bias = bias

        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.quant_config = copy.deepcopy(quant_config) if quant_config is not None else None
        self.del_orig = del_orig
        self.use_unpack_kernel = use_unpack_kernel

        # Legacy attributes
        self.linear_layer = linear_layer
        self.layer_activations = layer_activations

        self.bias_param: Optional[Tensor] = None
        self.axis = None
        self.channel_wise = None

        # Quantized state
        self.W_q: Optional[Tensor] = None
        self.meta: Optional[dict] = None
        self.ready: bool = False

        # Placeholder weight parameter for HF integration
        # This allows HF's weight loading to find a .weight in the model's state_dict
        # The actual weight will be quantized when quantize() is called
        # We register it as a parameter so it appears in state_dict()
        if in_features is not None and out_features is not None:
            # Create an empty/meta tensor as placeholder
            self.register_parameter(
                "weight",
                nn.Parameter(torch.empty(out_features, in_features, device="meta"), requires_grad=False)
            )
        else:
            self.weight = None

        # Backend selection (will be finalized when we have shape info)
        self.use_gemlite = False
        self._forward_impl = self.forward_pytorch  # Default

        # Buffers for GemLite functional call
        self._gl_tensor_args: Optional[List[Tensor]] = None
        self._gl_meta_args: Optional[List[int]] = None
        self._gl_bias: Optional[Tensor] = None
        self._gl_scale2: Optional[Tensor] = None
        self._gl_input_torch_dtype = None
        self._gemlite_ready: bool = False

        # Determine shape from linear_layer if provided (legacy path)
        if linear_layer is not None:
            self._in_features = linear_layer.in_features
            self._out_features = linear_layer.out_features
            self._has_bias = linear_layer.bias is not None

        # Decide backend based on config and shape (if available)
        self._update_backend_selection()

        # Legacy behavior: if both linear_layer and quant_config are provided,
        # quantize immediately (backwards compatible)
        if (self.linear_layer is not None) and (self.quant_config is not None):
            self._quantize_from_linear_layer()

    def _update_backend_selection(self):
        """Update backend selection based on current config and shape info."""
        qc = (self.quant_config or {}).get('weight_quant_params', {})

        can_use_gemlite = (
            qc.get('nbits', None) == 4
            and qc.get('tiling_mode', None) == '1D'
            and 'nogemlite' not in str(qc.get('method', '')).lower()
            and self.device.type == "cuda"
            and _HAVE_GEMLITE
        )

        # Check shape constraints if we have shape info
        if can_use_gemlite and self._in_features is not None and self._out_features is not None:
            in_f = self._in_features
            out_f = self._out_features
            group_size = qc.get('group_size', 64)

            if out_f < 16 or in_f < 16:
                can_use_gemlite = False
            elif group_size is not None and (in_f % group_size != 0 or in_f % 32 != 0):
                print(f"[SINQ] Input feature dimension is {in_f}, it is not divisible by the provided group size {group_size} or by 32. "
                      "Gemlite kernel cannot be applied for this layer.")
                can_use_gemlite = False

        if can_use_gemlite:
            self.use_gemlite = True
            self.set_forward_backend("gemlite")
        else:
            self.use_gemlite = False
            self.set_forward_backend("pytorch")

    def _quantize_from_linear_layer(self):
        """
        Legacy initialization path: quantize from an nn.Linear layer.
        Called when both linear_layer and quant_config are provided in __init__.
        """
        weight = self.linear_layer.weight.data
        bias = self.linear_layer.bias if hasattr(self.linear_layer, 'bias') and self.linear_layer.bias is not None else None

        # Perform quantization
        self.quantize(weight, bias=bias, layer_activations=self.layer_activations)

        # Clean up original layer if requested
        if self.del_orig:
            for name, _ in self.linear_layer.named_parameters():
                setattr(self.linear_layer, name, None)
            del self.linear_layer
            self.linear_layer = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def quantize(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        layer_activations: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Quantize a weight tensor into this SINQLinear layer.

        This method enables two-phase initialization:
        1. Create SINQLinear with shape/config info (in __init__)
        2. Call quantize() with the actual weights (here)

        Args:
            weight: The dense weight tensor to quantize (shape: [out_features, in_features])
            bias: Optional bias tensor
            layer_activations: Optional activations for A-SINQ calibration
        """
        if self.ready:
            raise RuntimeError("SINQLinear is already quantized. Cannot quantize again.")

        if self.quant_config is None:
            raise RuntimeError("quant_config must be set before calling quantize()")

        # Update shape info from weight tensor
        self._out_features = weight.shape[0]
        self._in_features = weight.shape[1]
        self._has_bias = bias is not None

        # Re-evaluate backend selection now that we have shape info
        self._update_backend_selection()

        # Handle group_size==None
        if self.quant_config["weight_quant_params"]["group_size"] is None:
            axis = self.quant_config["weight_quant_params"]["axis"]
            self.quant_config["weight_quant_params"]["group_size"] = (
                self._in_features if axis == 1 else self._out_features
            )

        # Quantize weights using Quantizer
        W_q, meta = Quantizer.quantize(
            weight,
            layer_activations,
            device=self.device,
            compute_dtype=self.compute_dtype,
            **self.quant_config['weight_quant_params'],
            use_unpack_kernel=self.use_unpack_kernel,
            bitpack=True,
        )

        self.W_q = W_q
        self.meta = meta

        # Handle bias
        self.bias_param = (
            None if bias is None
            else bias.clone().to(device=self.device, dtype=self.compute_dtype)
        )

        # For GemLite path we REQUIRE scale2
        if self.use_gemlite and "scale2" not in self.meta:
            raise RuntimeError("GemLite backend requires meta['scale2'] from the quantizer.")

        # Prepare GemLite if applicable
        if self.use_gemlite:
            self._prepare_gemlite_args(self.compute_dtype)

        # Remove the placeholder weight parameter (no longer needed after quantization)
        # This prevents it from appearing in state_dict() and consuming memory
        if hasattr(self, 'weight') and self.weight is not None:
            del self._parameters['weight']
        self.weight = None

        self.ready = True

    @property
    def in_features(self) -> int:
        """Return input feature dimension."""
        if self._in_features is not None:
            return self._in_features
        if self.meta is not None and "shape" in self.meta:
            return int(self.meta["shape"][1])
        raise RuntimeError("in_features not available - layer not initialized")

    @property
    def out_features(self) -> int:
        """Return output feature dimension."""
        if self._out_features is not None:
            return self._out_features
        if self.meta is not None and "shape" in self.meta:
            return int(self.meta["shape"][0])
        raise RuntimeError("out_features not available - layer not initialized")

    @property
    def bias(self) -> Optional[Tensor]:
        """Return bias tensor (for compatibility)."""
        return self.bias_param

    @bias.setter
    def bias(self, value: Optional[Tensor]):
        """Set bias tensor (for compatibility)."""
        self.bias_param = value

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
        bias   = None if self.bias_param is None else self.bias_param.to(self.device, dtype=desired_torch_dtype)

        # ---- TEMPORARY unpack for GemLite only ----
        if m.get("packing"):
            W_for_gl = Quantizer.unpack[m["packing"]](self.W_q, dtype=torch.uint8)
        else:
            # If someone saved unpacked 4-bit already (numel == N*K), just ensure u8
            if self.W_q.dtype != torch.uint8:
                # (shouldn't happen for 4-bit, but keep it robust)
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

        # Robust device comparison: handle 'cuda' vs 'cuda:0' mismatch
        # torch.device('cuda') != torch.device('cuda:0') is True, which would
        # incorrectly trigger rebuild on every forward pass
        current_device = torch.device(getattr(self, "device", "cuda"))
        device_mismatch = (
            current_device.type != runtime_dev.type
            or (current_device.index is not None and current_device.index != runtime_dev.index)
        )

        need_rebuild = (
            not getattr(self, "_gemlite_ready", False)
            or any(t.device != runtime_dev for t in (self._gl_tensor_args or []))
            or (getattr(self, "_gl_scale2", None) is not None and self._gl_scale2.device != runtime_dev)
            or (getattr(self, "_gl_bias", None) is not None and self._gl_bias is not None and self._gl_bias.device != runtime_dev)
            or device_mismatch
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
        if self.bias_param is not None:
            out += self.bias_param
        return out

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
        if self._in_features is not None and self._out_features is not None:
            wdev = getattr(getattr(self, "W_q", None), "device", None)
            wdev = str(wdev) if wdev is not None else "None"
            out = (
                f"in_features={self._in_features}, out_features={self._out_features}, bias={self.bias_param is not None}, "
                f"device={self.device}, W_q.device={wdev}"
            )
        elif getattr(self, "meta", None) is not None:
            in_features, out_features = self.meta["shape"][::-1]
            wdev = getattr(getattr(self, "W_q", None), "device", None)
            wdev = str(wdev) if wdev is not None else "None"
            out = (
                f"in_features={in_features}, out_features={out_features}, bias={self.bias_param is not None}, "
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

        For unquantized layers (ready=False), includes the placeholder 'weight' parameter
        so that HuggingFace's weight loading can detect it.
        """
        # If not yet quantized, use default PyTorch state_dict to expose placeholder weight
        if not self.ready:
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # For quantized layers, return only the quantized tensors
        sd = {}
        if self.W_q is not None:
            sd["W_q"] = self.W_q.detach().cpu()
        if self.bias_param is not None:
            sd["bias"] = self.bias_param.detach().cpu()
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
        self.bias_param = b.to(device=self.device, dtype=self.compute_dtype) if b is not None else None

        # Update shape info from meta
        if isinstance(self.meta, dict) and "shape" in self.meta:
            N, K = map(int, self.meta["shape"])
            self._out_features = N
            self._in_features = K

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
