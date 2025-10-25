# modified by SINQ authors 2025

import copy
from typing import Union, Optional

import torch
from torch import nn, Tensor

from .quantizer import Quantizer, dq8
from .utils import is_divisible

try:
    import gemlite
    has_gemlite = False
    #gemlite.set_autotune("fast")
    #gemlite.set_kernel_caching(True)
    #print('found gemlite installation')

except:
    has_gemlite = False

class SINQLinear(nn.Module):
    def __init__(
        self,
        linear_layer: Union[nn.Module, None],
        quant_config: Optional[dict] = None,
        del_orig: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        use_unpack_kernel: bool = False,
        layer_activations  = None
    ):
        super().__init__()
        
        # Handle case where quant_config is None (loading pre-quantized weights)
        if quant_config is None:
            qc = None
        else:
            qc = quant_config['weight_quant_params']

        if qc is not None and ('nogemlite' not in qc['method'].lower()) and qc['nbits'] == 4 and qc['tiling_mode'] == '1D' and has_gemlite:
            self.use_gemlite = True
        else:
            self.use_gemlite = False

        if self.use_gemlite:
            self.set_forward_backend("gemlite")
        else:
            self.set_forward_backend("pytorch")

        self.bias = None
        self.axis = None
        self.channel_wise = None
        self.device = device
        self.compute_dtype = compute_dtype
        self.quant_config = copy.deepcopy(quant_config) if quant_config is not None else None
        self.del_orig = del_orig
        self.use_unpack_kernel = use_unpack_kernel

        self.linear_layer = linear_layer
        self.W_q = None
        self.meta = None
        self.layer_activations = layer_activations

        # If we’re quantizing from a dense nn.Linear + have a config, do it now.
        # If we’re going to load pre-quantized tensors, skip initialization.
        if (self.linear_layer is not None) and (self.quant_config is not None):
            if self.use_gemlite:
                self.initialize_gemlite()
            else:
                self.initialize()
            self.ready = True
        else:
            self.ready = False

    def initialize_gemlite(self):
        self.gemlite_linear = gemlite.GemLiteLinear(self.quant_config['weight_quant_params']['nbits'], 
                                                    self.quant_config['weight_quant_params']['group_size'], 
                                                    self.linear_layer.in_features,
                                                    self.linear_layer.out_features, 
                                                    input_dtype=gemlite.DType.FP16, 
                                                    output_dtype=gemlite.DType.FP16)
        
        # gemlite.helper.warmup(shapes=[(self.linear_layer.in_features, self.linear_layer.out_features)], W_nbits=[self.quant_config['weight_quant_params']['nbits']], 
        #                       group_sizes=[self.quant_config['weight_quant_params']['group_size']], mode='static')

        if self.quant_config["weight_quant_params"]["group_size"] == None:
            self.quant_config["weight_quant_params"]["group_size"] = (
                self.linear_layer.in_features
                if (self.quant_config["weight_quant_params"]["axis"] == 1)
                else self.linear_layer.out_features
            )

        W_q, meta = Quantizer.quantize(
            self.linear_layer.weight.data,
            self.layer_activations,
            device=self.device,
            compute_dtype=self.compute_dtype,
            **self.quant_config['weight_quant_params'],
            use_unpack_kernel = self.use_unpack_kernel ,
            bitpack=False       
        )

        self.s2 = meta['scale2']
        scale = dq8(meta['scale'])
        zero = dq8(meta['zero'])  
        bias = None if self.linear_layer.bias is None else self.linear_layer.bias.clone().to(device=self.device, dtype=self.compute_dtype)  
        # print(W_q.shape, self.linear_layer.weight.data.shape)
        self.gemlite_linear.pack(W_q.to(torch.uint8), scale, zero, bias)
        del self.linear_layer

    def initialize(self):

        # Handle group_size==None
        if self.quant_config["weight_quant_params"]["group_size"] == None:
            self.quant_config["weight_quant_params"]["group_size"] = (
                self.linear_layer.in_features
                if (self.quant_config["weight_quant_params"]["axis"] == 1)
                else self.linear_layer.out_features
            )

        self.quantize(self.linear_layer.weight.data, self.layer_activations, **self.quant_config)
        self.bias = (
            None
            if (self.linear_layer.bias is None)
            else self.linear_layer.bias.clone().to(
                device=self.device, dtype=self.compute_dtype
            )
        )

        # Clear-up parameters
        if self.del_orig:
            for name, param in self.linear_layer.named_parameters():
                setattr(self.linear_layer, name, None)
            del self.linear_layer
            torch.cuda.empty_cache()

    def quantize(
        self,
        W: Tensor,
        layer_activations,
        weight_quant_params: dict,
        scale_quant_params: dict,
        zero_quant_params: dict,
    ) -> None:
        quant_scale = scale_quant_params is not None
        quant_zero = zero_quant_params is not None

        self.in_features, self.out_features = W.t().shape


        # Quantize
        # DEBUG
        # print("before Quantizer.quantize, self.device", self.device)
        W_q, meta = Quantizer.quantize(
            W,
            layer_activations,
            device=self.device,
            compute_dtype=self.compute_dtype,
            **weight_quant_params,
            use_unpack_kernel = self.use_unpack_kernel        
        )

        meta.update({"quant_scale": quant_scale, "quant_zero": quant_zero})

        self.W_q = W_q
        self.meta = meta
        self.ready = True

    def unpack(self, reshape=False, dtype=None):
        if self.ready is False:
            return None
        if self.meta["packing"]:
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

        if meta["quant_zero"]:
            meta["zero"] = Quantizer.dequantize(
                meta["zero_q"].to(device=device), meta["meta_zero"]
            )
            del_keys.add("zero")

        if meta["quant_scale"]:
            meta["scale"] = Quantizer.dequantize(
                meta["scale_q"].to(device=device), meta["meta_scale"]
            )
            del_keys.add("scale")

        W_est = Quantizer.dequantize(W_q, meta, use_unpack_kernel=self.use_unpack_kernel)

        # DEBUG
        # print("W_q.device", W_est.device)

        # Cleanup
        for key in del_keys:
            del meta[key]
        return W_est

    def forward_gemlite(self, x:Tensor) -> Tensor:
        out = self.gemlite_linear(self.s2*x)
        return out

    def forward_pytorch(self, x: Tensor) -> Tensor:
        # DEBUG
        # print(x.device, self.dequantize().device)

        out = torch.matmul(x, self.dequantize().t())
        if self.bias is not None:
            out += self.bias
        return out

    def forward_ascendc(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def set_forward_backend(cls, backend: str):
        if backend == "pytorch":
            cls.forward = cls.forward_pytorch
        elif backend == "ascendc":
            cls.forward = cls.forward_ascendc
        elif backend == "gemlite":
            cls.forward = cls.forward_gemlite

    def extra_repr(self) -> str:
        out = ""
        if hasattr(self, "meta"):
            if self.meta is not None:
                in_features, out_features = self.meta["shape"][::-1]
                out = (
                    f"in_features={in_features}, out_features={out_features}, bias={self.bias is not None}, "
                    f"device={self.device}, W_q.device={self.W_q.device}"
                )
        return out
    
    def _meta_to_cpu(self, meta: dict) -> dict:
        if meta is None:
            return None

        def to_cpu(v):
            import torch
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
                    return {
                        "x": to_cpu(x),
                        "s": to_cpu(s),
                        "m": to_cpu(m),
                        "shape": list(shape),  # JSON-friendly
                    }
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

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Restore pre-quantized tensors without re-quantizing.
        Assumes self.device / self.compute_dtype are set by the caller.
        """
        # Required
        self.W_q = state_dict["W_q"].to(device=self.device)
        self.meta = state_dict["meta"]

        # Optional bias
        b = state_dict.get("bias", None)
        self.bias = b.to(device=self.device, dtype=self.compute_dtype) if b is not None else None

        # Infer features for nicer repr and possible use elsewhere
        if isinstance(self.meta, dict) and "shape" in self.meta:
            out_f, in_f = self.meta["shape"]  # meta stores (out_features, in_features)
            self.in_features, self.out_features = in_f, out_f

        self.ready = True

        # Match nn.Module API return
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
    assert (
        nbits in Quantizer.SUPPORTED_BITS
    ), "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
    if method == "asinq":
        # Remap sinq_awq_l1_quantAux to behave like asinq (A-SINQ in the paper)
        method = "sinq_awq_l1_quantAux"
    elif method == "sinq":
        # Remap so that users can use sinq_quantAux as sinq (scales and zeros are quantized to 8-bit)
        method = "sinq_quantAux"
    if group_size is not None:
        assert is_divisible(
            group_size, 8
        ), "Invalid group_size param: the value should be a multiple of 8."

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
        print(
            colored(
                "Warning: Quantized meta-data is deprecated and will be removed. It is not supported for quantized model serialization.",
                "yellow",
            )
        )

    scale_quant_params = (
        {"nbits": 8, "group_size": 128}
        if (quant_scale)
        else None
    )
    zero_quant_params = (
        {"nbits": 8, "group_size": None}
        if (quant_zero)
        else None
    )

    return {
        "weight_quant_params": weight_quant_params,
        "scale_quant_params": scale_quant_params,
        "zero_quant_params": zero_quant_params
    }


# Alias: follow similar Auto-GPTQ naming
BaseQuantizeConfig = sinq_base_quant_config

