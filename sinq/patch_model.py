# modified by SINQ authors 2025
import os
import json
import math
import shutil
import torch
from torch import nn
from torch import float16
from os.path import join as pjoin
from typing import Callable
from tqdm import tqdm
from abc import abstractmethod
from functools import partial
from typing import Union
import transformers
from accelerate import init_empty_weights
from pathlib import Path

from .utils import cleanup
from .sinqlinear import SINQLinear
from .awq import *
from huggingface_hub import snapshot_download

# --- optional safetensors support (adds capability without changing defaults) ---
try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
except Exception:
    _st_save = _st_load = None

# Sharded safetensors constants
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_BASENAME   = "model"  # -> model-00001-of-000NN.safetensors

def _parse_size_to_bytes(x) -> int:
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip().upper()
    if s.endswith("GB"):
        return int(float(s[:-2]) * (1024**3))
    if s.endswith("MB"):
        return int(float(s[:-2]) * (1024**2))
    if s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    return int(s)

# Defined what is qualified as "linear layer"
_QUANT_LAYERS = [nn.Linear, SINQLinear]
_IGNORE_LINEAR = ["lm_head"]

# Finds the parent of a node module named "name"
def find_parent(model, name: str) -> nn.Module:
    module_tree = name.split(".")[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent

# checks if a module is a leaf: doesn't have another module inside
def is_leaf_module(module) -> bool:
    return len(module._modules) == 0

# Get the linear_tag from a modul name. For example: model.layers.31.self_attn.k_proj -> self_attn.k_proj
def name_to_linear_tag(name: str) -> str:
    return ".".join(
        [
            n
            for n in name.split(".")
            if ((n not in ["model", "layers"]) and (not n.isnumeric()))
        ]
    )

# returns all children nodes from model
def get_all_children_from_model(model, ignore: list = []) -> list:
    tags = []
    for name, module in model.named_modules():
        if is_leaf_module(module) and (name.split(".")[-1] not in ignore):
            tags.append(name)
    return tags

# Get all linear tags available
def get_linear_tags_from_model(model, ignore: list) -> list:
    linear_tags = set()
    for name, module in model.named_modules():
        if (type(module) in _QUANT_LAYERS) and (name.split(".")[-1] not in ignore):
            linear_tags.add(name_to_linear_tag(name))
    return list(linear_tags)

def forward_device_hooked(self, *args, **kwargs):
    args = list(args)

    # eddit this to make torch.compile compatible
    for i in range(len(args)):
        if isinstance(
            args[i], (torch.Tensor, torch.nn.Parameter)
        ):  # if hasattr(args[i], "to"):
            args[i] = args[i].to(self.device)

    for i in kwargs:
        if isinstance(
            kwargs[i], (torch.Tensor, torch.nn.Parameter)
        ):  # if hasattr(kwargs[i], "to"):
            kwargs[i] = kwargs[i].to(self.device)

    # return self.__class__.forward(self, *args, **kwargs)
    return self.forward_orig(*args, **kwargs)

def _retie_tied_leaves(model, saved_weights: dict | None = None):
     core = model.model if hasattr(model, "model") else model
     if hasattr(model, "lm_head") and hasattr(core, "embed_tokens"):
        try:
            # If the user saved an explicit lm_head leaf, do NOT re-tie.
            if saved_weights and "lm_head" in saved_weights:
                return
            if model.lm_head.weight.data_ptr() != core.embed_tokens.weight.data_ptr() \
                and model.lm_head.weight.shape == core.embed_tokens.weight.shape:
                 model.lm_head.weight = core.embed_tokens.weight
        except Exception as e:
             print(f"[retie] Skipping retie: {e}")

def _detect_tied_leaves(model) -> set[str]:
    """
    Return the set of leaves that are actually tied *in this instance*.
    We only check lm_head <-> embed_tokens.
    """
    tied = set()
    core = model.model if hasattr(model, "model") else model
    if hasattr(model, "lm_head") and hasattr(core, "embed_tokens"):
        try:
            w_head = getattr(model.lm_head, "weight", None)
            w_tok  = getattr(core.embed_tokens, "weight", None)
            if w_head is not None and w_tok is not None:
                # "Tied" if they share storage
                if w_head.data_ptr() == w_tok.data_ptr() and w_head.shape == w_tok.shape:
                    tied.add("lm_head")
        except Exception:
            pass
    return tied

def _pick_sinq_class(
    *,
    model=None,
    save_dir_or_hub: str | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    token: str | None = None,
):
    """
    Unified classifier for which SINQ HF wrapper to use.

    Can be called in two ways:

      - For SAVING:
            _pick_sinq_class(model=my_model)

      - For LOADING:
            _pick_sinq_class(save_dir_or_hub="repo_or_path", ...)

    Returns one of:
        QwenSINQHFModel, KimiSINQHFModel, or AutoSINQHFModel
    """
    cfg = None
    model_type = ""
    archs_lower = ""

    # ---- If we have a live model, use it first ----
    if model is not None:
        cfg = getattr(model, "config", None)

        # Strong hint: module names
        for m in model.modules():
            cls_name = m.__class__.__name__.lower()
            if "qwen3nextgateddeltanet" == cls_name or "qwen" in cls_name:
                return QwenSINQHFModel
            if "kimi" in cls_name or "moonshot" in cls_name:
                return KimiSINQHFModel

    # ---- If config not known yet, and we have a repo/path, load config ----
    if cfg is None and save_dir_or_hub is not None:
        cfg = transformers.AutoConfig.from_pretrained(
            save_dir_or_hub,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
        )

    # If still no config, fall back to generic Auto
    if cfg is None:
        return AutoSINQHFModel

    model_type = (getattr(cfg, "model_type", "") or "").lower()
    archs = getattr(cfg, "architectures", []) or []
    archs_lower = " ".join(archs).lower()

    # ---- Qwen detection ----
    if "qwen" in model_type or "qwen" in archs_lower:
        return QwenSINQHFModel

    # ---- Kimi/Moonshot detection (tweak as needed) ----
    if (
        "kimi" in model_type
        or "kimi" in archs_lower
        or "moonshot" in model_type
        or "moonshot" in archs_lower
    ):
        return KimiSINQHFModel

    # ---- Default: generic Auto ----
    return AutoSINQHFModel

# Base patching class. Patching defines how nn.Linear and other layers are replaced via a patching function.
class BasePatch:
    # Override these OR override the main patch_model() function
    ############################################
    # This method iterates through layers of the model that are NOT nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_nonlinearlayers(
        cls, model, patch_fct: Callable, verbose: bool = True
    ) -> None:
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) not in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_fct(tmp_mapping[name]),
            )

        cleanup()

    # This method iterates through layers of the model that are nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_linearlayers(
        cls,
        model,
        patch_fct: Callable,
        patch_params: Union[dict, None],
        verbose: bool = True,
    ) -> None:
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            linear_tag = name_to_linear_tag(name)
            patch_param = (
                patch_params[linear_tag] if (linear_tag in patch_params) else None
            )
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_fct(tmp_mapping[name], patch_param),
            )

        cleanup()

    ############################################
    # These tags are used to specfiy parameters of the patching in patch_linearlayers()
    @classmethod
    def set_auto_linear_tags(cls, model, ignore: list = _IGNORE_LINEAR) -> None:
        if hasattr(model, "linear_tags") is False:
            linear_tags = cls.get_linear_tags()
            model.linear_tags = (
                linear_tags
                if len(linear_tags) > 0
                else get_linear_tags_from_model(model, ignore=ignore)
            )
            model.base_class = cls

    # Returns the current linear tags
    @classmethod
    def get_linear_tags(cls) -> list:
        return []

    @classmethod
    def get_ignore_layers(cls, model) -> list:
        """
        Ignore the root module ("") and all non-leaf container modules.
        Leaf modules (which actually own tensors) are NOT ignored.
        """
        layers = {""}
        for name, module in model.named_modules():
            if not is_leaf_module(module):
                layers.add(name)
        return list(layers)


    # Autmatically name modules. This is very important to save/load the weights
    @classmethod
    def autoname_modules(cls, model) -> None:
        for name, module in model.named_modules():
            module.name = name

    # Freeze all layers
    @classmethod
    def freeze_model(cls, model) -> None:
        for param in model.parameters():
            param.requires_grad = False
        try:
            for param in model.model.parameters():
                param.requires_grad = False
        except Exception:
            pass

    # Main patching function
    @classmethod
    def patch_model(
        cls,
        model,
        patch_nonlinear_fct: Callable,
        patch_linear_fct: Callable,
        patch_params: dict,
        verbose: bool = True,
    ) -> None:
        model.eval()
        cls.freeze_model(model)
        cls.autoname_modules(model)
        cls.patch_nonlinearlayers(model, patch_nonlinear_fct, verbose=verbose)
        cls.patch_linearlayers(model, patch_linear_fct, patch_params, verbose=verbose)
        cleanup()

class BaseSINQModel:
    TIED_LEAVES = {"lm_head"}
    @classmethod
    def get_config_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "config.json")

    @classmethod
    def get_weight_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "qmodel.pt")

    # Save weights to disk
    @classmethod
    def save_weights(cls, weights: dict, save_dir: str) -> None:
        torch.save(weights, cls.get_weight_file(save_dir))

    # Load weights from disk
    @classmethod
    def load_weights(cls, save_dir: str, map_location=None):
        return torch.load(
            cls.get_weight_file(save_dir), map_location=map_location, weights_only=True
        )

    # ===========================
    # Sharded safetensors (ONLY)
    # ===========================
    @classmethod
    def load_weights_safetensors(cls, save_dir: str, map_location="cpu", filename: str = "model.safetensors") -> dict:
        """
        Sharded-only loader:
          - reads 'model.safetensors.index.json'
          - loads all listed shards and merges into a flat dict
          - re-groups per-leaf; re-nests '.meta.' entries
          - merges non-tensor sidecar from 'model.safetensors.index.json.meta.json'
        Note: the 'filename' arg is ignored in sharded mode and kept only for API compatibility.
        """
        def _set_nested(d: dict, dotted: str, value):
            parts = dotted.split(".") if dotted else [""]
            cur = d
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value
        if _st_load is None:
            raise ImportError("safetensors not installed. `pip install safetensors`")

        index_path = os.path.join(save_dir, SAFE_WEIGHTS_INDEX_NAME)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing index file: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise RuntimeError("Empty weight_map in index.")

        # Load each shard once, merge tensors
        flat = {}
        shard_to_keys = {}
        for k, fname in weight_map.items():
            shard_to_keys.setdefault(fname, []).append(k)

        for fname, keys in shard_to_keys.items():
            shard_path = os.path.join(save_dir, fname)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(shard_path)
            shard = _st_load(shard_path, device=map_location)  # dict[str, Tensor]
            # Keep only needed keys (defensive)
            for k in keys:
                t = shard.get(k, None)
                if t is None:
                    raise KeyError(f"Key {k} missing from shard {fname}")
                flat[k] = t

        # Re-group per leaf; re-nest meta.* using explicit '.meta.' split
        grouped: dict[str, dict] = {}
        for full, t in flat.items():
            if ".meta." in full:
                leaf, _, subk = full.partition(".meta.")
                tgt_meta = grouped.setdefault(leaf, {}).setdefault("meta", {})
                _set_nested(tgt_meta, subk, t)  # <= re-nest "scale.x" into ["scale"]["x"]
                continue
            if "." in full:
                leaf, key = full.rsplit(".", 1)
            else:
                leaf, key = full, ""
            grouped.setdefault(leaf, {})[key] = t


        # Merge sidecar non-tensors
        sidecar_path = index_path + ".meta.json"
        if os.path.isfile(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)

            import torch as _torch
            _DTYPE_MAP = {
                "torch.float16": _torch.float16, "float16": _torch.float16,
                "torch.bfloat16": _torch.bfloat16, "bfloat16": _torch.bfloat16,
                "torch.float32": _torch.float32, "float32": _torch.float32,
                "torch.float64": _torch.float64, "float64": _torch.float64,
                "torch.int8": _torch.int8, "int8": _torch.int8,
                "torch.int16": _torch.int16, "int16": _torch.int16,
                "torch.int32": _torch.int32, "int32": _torch.int32,
                "torch.int64": _torch.int64, "int64": _torch.int64,
            }

            def _restore(x):
                if isinstance(x, str):
                    if x in _DTYPE_MAP:
                        return _DTYPE_MAP[x]
                    if x == "cpu" or x.startswith("cuda"):
                        try:
                            return _torch.device(x)
                        except Exception:
                            return x
                    return x
                if isinstance(x, list):
                    return [_restore(v) for v in x]
                if isinstance(x, dict):
                    return {k: _restore(v) for k, v in x.items()}
                return x

            for leaf, leaf_dict in meta_json.items():
                tgt = grouped.setdefault(leaf, {})
                restored = _restore(leaf_dict)

                # If sidecar has a nested "meta" dict, merge it into tgt["meta"]
                meta_side = restored.pop("meta", None)
                if meta_side:
                    tgt_meta = grouped.setdefault(leaf, {}).setdefault("meta", {})

                    # meta_side might contain flattened dotted keys (e.g., "scale.shape")
                    for k, v in meta_side.items():
                        _set_nested(tgt_meta, k, v)

                # any remaining top-level non-tensors on the leaf:
                for k, v in restored.items():
                    grouped.setdefault(leaf, {})[k] = v

        return grouped

    # ------------------------------------------------------------------------

    # Set-up model with the necessary data
    @classmethod
    def setup_model(cls, model):
        cls.autoname_modules(model)
        cls.set_auto_linear_tags(model)

    # Main function to quantize a model. Basically goes through the linear layers specfied in the patching function and replaces them with SINQLinear
    @classmethod
    def quantize_model(
        cls,
        model,
        tokenizer,
        quant_config: dict,
        compute_dtype: torch.dtype = float16,
        device: Union[str, list, dict] = "cuda",
        use_unpack_kernel: bool = True,
    ):
        # Check if the model was already quantized
        if getattr(model, "sinq_quantized", False):
            print("Model was already quantized")
            return

        # Set linear tags automatically
        cls.setup_model(model)

        # AWQ
        if 'awq' in quant_config['weight_quant_params']['method']:
            print('computing awq calibration activations')
            # calibration_data = get_calib_dataset(data="pileval", tokenizer=tokenizer,
            #                                      n_samples=128, block_size=512)
            calibration_data = get_simple_calibration_data(tokenizer=tokenizer)
            # calibration_data = get_calib_dataset(tokenizer=tokenizer, n_samples=64)
            torch.cuda.empty_cache()
            try:
                mc = model.bfloat16().cuda()
            except:
                mc = model.bfloat16()
            activations = collect_activations(mc, calibration_data, 128)
            del mc
            torch.cuda.empty_cache()
            print('calibration activations collected.')

        # Use the same quantization config for all linear layers. Use None to skip quantizing a specfic layer.
        if True in [(key in model.linear_tags) for key in quant_config.keys()]:
            # If the user doesn't specify a key from get_linear_tags, the layer is not quantized via (key, None)
            patch_params = {key: None for key in model.linear_tags}
            patch_params.update(quant_config)
        else:
            # Same quant_config for all layers
            patch_params = {k: quant_config for k in model.linear_tags}

        # Get list of all nodes in order
        all_nodes = get_all_children_from_model(model, [])  # ordered nodes
        try:
            # Extract block names: This is following Hugging Face models.
            num_blocks = (
                len(model.model.layers)
                if hasattr(model, "model")
                else len(model.layers)
            )
            all_blocks = ["model.layers." + str(i) for i in range(num_blocks)]
        except Exception:
            all_blocks = None
            print(
                "Default model structure not supported. Make sure you feed device as dictionary as {name_block: device}"
            )

        if isinstance(
            device, dict
        ):  # input as {module block name (str): device (str or torch.device)}
            device_map = device
            num_devices = len(set([device_map[k] for k in device_map]))
            all_blocks = list(device_map.keys())

        node_to_block = {}
        for node in all_nodes:
            res = [block for block in all_blocks if (block in node)]
            node_to_block[node] = res[-1] if (len(res) > 0) else node

        # Set device-map
        if isinstance(device, str):  # single device as str
            device_map = {k: device for k in all_blocks + all_nodes}
            num_devices = 1

        if isinstance(device, list):  # list of devices
            num_devices = len(device)
            device_map = {}
            for node in all_nodes:
                if ".layers" in node:
                    break
                device_map[node] = device[0]

            for node in all_nodes[::-1]:
                if ".layers" in node:
                    break
                device_map[node] = device[-1]

            step, k = len(all_blocks) // num_devices, 0
            for i in range(0, len(all_blocks), step):
                for j in range(i, i + step):
                    device_map[all_blocks[min(j, len(all_blocks) - 1)]] = device[
                        min(k, num_devices - 1)
                    ]
                k += 1

        # Map nodes to block devices
        for node in all_nodes:
            device_map[node] = device_map[node_to_block[node]]

        # We replace the nn.Linear layers with SINQLinear
        def _patch_linear(linear_layer, quant_config):
            if type(linear_layer) is SINQLinear:
                return linear_layer

            current_device = device_map[linear_layer.name]
            # print(linear_layer.name) # the layer's name

            if quant_config is not None:
                if 'awq' in quant_config['weight_quant_params']['method']:
                    layer_activations = activations.get(linear_layer.name, None)
                else:
                    layer_activations = None
                out_module = SINQLinear(
                    linear_layer,
                    quant_config,
                    compute_dtype=compute_dtype,
                    device=current_device,
                    use_unpack_kernel = use_unpack_kernel,
                    layer_activations = layer_activations
                )
            else:
                out_module = linear_layer.to(device=current_device, dtype=compute_dtype)

            out_module.device = current_device
            return out_module

        def _patch_other(layer):
            # Try to look up the device for this module
            current_device = device_map.get(layer.name, None)

            if current_device is None:
                # Derive it from the enclosing block (e.g. "model.layers.0")
                block = None
                if all_blocks is not None:
                    for b in all_blocks:
                        if layer.name.startswith(b):
                            block = b
                            break

                if block is not None and block in device_map:
                    current_device = device_map[block]
                else:
                    # Fallback: single-device case or first device from a list/dict
                    if isinstance(device, str):
                        current_device = device
                    elif isinstance(device, list) and len(device) > 0:
                        current_device = device[0]
                    elif isinstance(device, dict) and len(device) > 0:
                        current_device = next(iter(device.values()))
                    else:
                        current_device = "cuda"

                # Cache it so future calls see it directly
                device_map[layer.name] = current_device

            layer.device = current_device
            return layer.to(device=current_device, dtype=compute_dtype)

        cls.patch_model(model, _patch_other, _patch_linear, patch_params)

        # Insert device switcher
        if num_devices > 1:
            core_model = model if hasattr(model, "layers") else model.model

            # Make sure the input (first node) has the input in the right device during generation
            input_node_child_name = all_nodes[0].split(".")[-1]
            input_node = getattr(core_model, input_node_child_name)
            input_node.device = device_map[all_nodes[0]]
            input_node.forward_orig = input_node.forward
            input_node.forward = partial(forward_device_hooked, input_node)
            setattr(core_model, input_node_child_name, input_node)

            # Make sure all inputs to the blocks are in the right device
            for i in range(len(core_model.layers)):
                core_model.layers[i].device = device_map[core_model.layers[i].name]
                core_model.layers[i].forward_orig = core_model.layers[i].forward
                core_model.layers[i].forward = partial(
                    forward_device_hooked, core_model.layers[i]
                )

        # Set base class
        model.base_class = cls

        model.sinq_quantized = True

        return model

    # Prepares model weights by iterating through modules. It might some parameters that are NOT modules like model.param1
    @classmethod
    def serialize_weights(cls, model, verbose: bool = False) -> dict:
        weights = {}
        ignore_keys = cls.get_ignore_layers(model)
        actually_tied = _detect_tied_leaves(model)

        def _is_leaf(m: nn.Module) -> bool:
            return len(m._modules) == 0

        # 1) Generic leaf capture
        for name, module in model.named_modules():
            if name in ignore_keys or not _is_leaf(module):
                continue
            if name in actually_tied:
                continue
                
            try:
                state = module.state_dict()
                if len(state) == 0:
                    has_params_or_bufs = (
                        any(True for _ in module.parameters(recurse=False))
                        or any(b is not None for b in module.buffers(recurse=False))
                    )
                    if has_params_or_bufs:
                        direct = {}
                        for k, p in module.named_parameters(recurse=False):
                            direct[k] = p.detach().clone().cpu()
                        for k, b in module.named_buffers(recurse=False):
                            if b is not None:
                                direct[k] = b.detach().clone().cpu()
                        state = direct

                if len(state) > 0:
                    weights[name] = state

            except Exception as e:
                if verbose:
                    print(f"[serialize_weights] Skipping {name}: {e}")

        # 2) Let subclasses add model-specific extras
        if hasattr(cls, "extra_serialize_weights"):
            cls.extra_serialize_weights(model, weights)

        if verbose:
            total_bytes = 0
            total_tensors = 0
            for sd in weights.values():
                for v in sd.values():
                    if isinstance(v, torch.Tensor):
                        total_tensors += 1
                        total_bytes += v.numel() * v.element_size()
            gb = total_bytes / (1024 ** 3)
            print(f"[serialize_weights] Saved {len(weights)} modules, "
                  f"{total_tensors} tensors, ~{gb:.2f} GB of tensor data")

        return weights

    # Main function to save a quantized model
    @classmethod
    def save_quantized(cls, model, tokenizer, save_dir: str, verbose: bool = False, write_tokenizer: bool = True):
        # Ensure target directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save config (writes config.json)
        cls.cache_model(model, save_dir)

        if write_tokenizer:
            try:
                BaseSINQHFModel.save_tokenizer_assets(tokenizer, save_dir)
            except Exception as e:
                if verbose:
                    print(f"[save_quantized_safetensors] Could not save tokenizer: {e}")

        # Serialize per-module weights
        weights = cls.serialize_weights(model, verbose=verbose)

        # Save weights blob (e.g., qmodel.pt)
        cls.save_weights(weights, save_dir)

    @classmethod
    def save_quantized_safetensors(cls, model, tokenizer, save_dir: str, filename: str = "model.safetensors", verbose: bool = False, max_shard_size="4GB", write_tokenizer: bool = True):
        """
        Sharded-only: writes multiple *.safetensors shards + a HF-style index file.
        Non-tensor meta goes to 'model.safetensors.index.json.meta.json'.
        Note: the 'filename' arg is ignored (kept for API compatibility).
        """
        if _st_save is None:
            raise ImportError("safetensors not installed. `pip install safetensors`")
        os.makedirs(save_dir, exist_ok=True)
        cls.cache_model(model, save_dir)
        # (NEW) Save tokenizer files first so the folder is self-contained early
        if write_tokenizer:
            try:
                BaseSINQHFModel.save_tokenizer_assets(tokenizer, save_dir)
            except Exception as e:
                if verbose:
                    print(f"[save_quantized_safetensors] Could not save tokenizer: {e}")
        weights = cls.serialize_weights(model, verbose=verbose)
        cls.save_weights_safetensors(weights, save_dir, filename=filename, max_shard_size=max_shard_size)
        
    # Main function to load a SINQ quantized model from either HF hub or locally
    @classmethod
    def from_quantized(
        cls,
        save_dir_or_hub,
        compute_dtype: torch.dtype = float16,
        device="cuda",
        cache_dir: Union[str, None] = "",
        **kwargs,
    ):
        # Local folder only for now (Hub comes next)
        if not os.path.isdir(save_dir_or_hub):
            raise ValueError(
                f"Expected a local directory for 'save_dir_or_hub' (got: {save_dir_or_hub})."
            )
        save_dir = save_dir_or_hub

        # Recreate empty model from config (meta tensors)
        model = cls.create_model(save_dir, kwargs)
        model.save_dir = save_dir
        cls.setup_model(model)

        # ---- Load serialized weights dict ON CPU ----
        try:
            # Important: keep raw weights on CPU, we'll move tensors to `device` per-module.
            weights = cls.load_weights(save_dir, map_location="cpu")
        except Exception as e:
            print("Failed to load the weights")
            raise FileNotFoundError(f"Could not load weights from {save_dir}: {e}")

        DYNAMIC_TIED = _detect_tied_leaves(model)

        # ---- Preflight: every parameterized leaf must have an entry in `weights`
        param_leaves = []
        for name, module in model.named_modules():
            # Only check leaves
            if len(module._modules) == 0:
                has_params_or_buffers = any(True for _ in module.parameters(recurse=False)) or \
                                        any(b is not None for b in module.buffers(recurse=False))
                if has_params_or_buffers:
                    param_leaves.append(name)

        missing = [n for n in param_leaves if (n not in weights and n not in DYNAMIC_TIED)]
        if len(missing) > 0:
            preview = ", ".join(missing[:10])
            raise RuntimeError(
                f"[SINQ] {len(missing)} parameterized leaf modules are missing from saved weights "
                f"(examples: {preview}). This will cause wrong outputs. "
                "Fix serialize_weights() so these leaves are included."
            )
        # ---- End preflight

        @torch.no_grad()
        def _load_module(module, params=None):
            # Stateless leaf (e.g., Dropout/GELU) -> nothing to load or move; keep meta ok
            if module.name not in weights:
                has_params_or_buffers = any(True for _ in module.parameters(recurse=False)) or \
                                        any(b is not None for b in module.buffers(recurse=False))
                if has_params_or_buffers:
                    if module.name in DYNAMIC_TIED:
                        module.device = device
                        return module
                    # Should never happen thanks to preflight
                    raise RuntimeError(f"Missing weights for parameterized leaf: {module.name}")
                module.device = device
                return module

            # Restore from saved dict (CPU tensors)
            state_dict = weights[module.name]

            # Quantized SINQLinear special case
            if "W_q" in state_dict:
                m = SINQLinear(
                    linear_layer=None,
                    quant_config=None,
                    compute_dtype=compute_dtype,
                    device=device,
                )
                m.load_state_dict(state_dict, strict=True)
                m.device = device
                return m

            # Regular leaf with tensors
            for key, tensor in state_dict.items():
                is_param = (key in getattr(module, "_parameters", {}) and getattr(module, "_parameters")[key] is not None)
                is_buffer = key in getattr(module, "_buffers", {})

                if is_param:
                    # cast params to compute_dtype and move to `device`
                    t = tensor.to(device=device, dtype=compute_dtype, non_blocking=True)
                    setattr(module, key, nn.Parameter(t, requires_grad=False))
                elif is_buffer:
                    # keep original buffer dtype, just move device
                    t = tensor.to(device=device, dtype=tensor.dtype, non_blocking=True)
                    module._buffers[key] = t
                else:
                    # fallback for non-registered attrs occasionally present in state_dict
                    t = tensor.to(device=device, non_blocking=True)
                    setattr(module, key, t)

            module.device = device
            return module

        # Patch all leaves (this will move only what we actually attach)
        cls.patch_model(model, _load_module, _load_module, {k: None for k in model.linear_tags})

        if hasattr(cls, "post_module_load"):
            cls.post_module_load(model, weights)

        # Re-tie after modules are in place
        _retie_tied_leaves(model, saved_weights=weights)
        model.sinq_quantized = True
        model.base_class = cls
        model.eval()

        # ---- Free the raw weights dict to release memory ----
        del weights
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return model

    @classmethod
    def save_weights_safetensors(cls, weights: dict, save_dir: str, filename: str = "model.safetensors", max_shard_size="4GB") -> None:
        """
        Sharded-only writer.
        Save tensors across shards <= max_shard_size and write:
          - model.safetensors.index.json           (tensor map)
          - model-00001-of-000NN.safetensors, ...  (shards)
          - model.safetensors.index.json.meta.json (non-tensors / meta)
        Note: the 'filename' arg is ignored; we always use HF-style names.
        """
        if _st_save is None:
            raise ImportError("safetensors not installed. `pip install safetensors`")

        import torch as _torch
        os.makedirs(save_dir, exist_ok=True)
        max_bytes = _parse_size_to_bytes(max_shard_size)

        # Flatten tensors; collect non-tensors into a single sidecar (per-leaf)
        flat = {}     # key -> Tensor (to shard)
        sidecar = {}  # leaf -> { non-tensor fields }, including meta non-tensors

        def _to_jsonable(x):
            if x is None or isinstance(x, (bool, int, float, str)):
                return x
            if isinstance(x, _torch.dtype) or isinstance(x, _torch.device):
                return str(x)
            if isinstance(x, (list, tuple)):
                return [_to_jsonable(v) for v in x]
            if isinstance(x, dict):
                return {k: _to_jsonable(v) for k, v in x.items()}
            try:
                json.dumps(x)
                return x
            except TypeError:
                return repr(x)
            
        def _extract_meta(leaf: str, meta_obj, prefix: str = ""):
            if isinstance(meta_obj, _torch.nn.Parameter):
                meta_obj = meta_obj.data
            if isinstance(meta_obj, _torch.Tensor):
                flat[f"{leaf}.meta{('.' + prefix) if prefix else ''}"] = meta_obj.detach().to("cpu").contiguous()
                return

            if isinstance(meta_obj, dict):
                for k, v in meta_obj.items():
                    _extract_meta(leaf, v, f"{prefix}.{k}" if prefix else k)
                return

            if isinstance(meta_obj, (list, tuple)):
                # store lists/tuples (e.g., shapes) in sidecar
                sidecar.setdefault(leaf, {}).setdefault("meta", {})[prefix] = _to_jsonable(meta_obj)
                return

            # anything else goes to sidecar
            sidecar.setdefault(leaf, {}).setdefault("meta", {})[prefix] = _to_jsonable(meta_obj)

        for leaf, sd in weights.items():
            for k, v in sd.items():
                # 1) meta dict - split tensors vs non-tensors
                if k == "meta" and isinstance(v, dict):
                    if k == "meta" and isinstance(v, dict):
                        _extract_meta(leaf, v)
                        continue
                # 2) Regular entries
                if isinstance(v, _torch.nn.Parameter):
                    v = v.data
                if isinstance(v, _torch.Tensor):
                    key = f"{leaf}.{k}" if k != "" else leaf
                    flat[key] = v.detach().to("cpu").contiguous()
                else:
                    sidecar.setdefault(leaf, {})[k] = _to_jsonable(v)

        if not flat:
            raise ValueError("No tensor entries found to save in safetensors.")

        # Greedy pack into shards (approximate by tensor.nbytes)
        items = list(flat.items())
        shard_maps = []   # list of dict key->Tensor for each shard
        shard_sizes = []
        cur_map, cur_bytes = {}, 0
        for k, t in items:
            nbytes = t.element_size() * t.numel()
            if cur_map and (cur_bytes + nbytes > max_bytes):
                shard_maps.append(cur_map); shard_sizes.append(cur_bytes)
                cur_map, cur_bytes = {}, 0
            cur_map[k] = t
            cur_bytes += nbytes
        if cur_map:
            shard_maps.append(cur_map); shard_sizes.append(cur_bytes)

        num_shards = len(shard_maps)
        if num_shards == 0:
            raise RuntimeError("Sharding produced zero shards.")

        # Write shards + build index json (HF-style)
        index = {
            "metadata": {
                "total_size": int(sum(shard_sizes)),
            },
            "weight_map": {}  # tensor_key -> shard filename
        }

        for i, shard in enumerate(shard_maps, start=1):
            shard_name = f"{SAFE_WEIGHTS_BASENAME}-{i:05d}-of-{num_shards:05d}.safetensors"
            shard_path = os.path.join(save_dir, shard_name)
            _st_save(shard, shard_path)
            for k in shard.keys():
                index["weight_map"][k] = shard_name

        # Write index file
        index_path = os.path.join(save_dir, SAFE_WEIGHTS_INDEX_NAME)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f)

        # Write single sidecar for non-tensors
        sidecar_path = index_path + ".meta.json"
        if sidecar:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f)

    @classmethod
    def from_quantized_safetensors(
        cls,
        save_dir_or_hub,
        compute_dtype: torch.dtype = float16,
        device="cuda",
        filename: str = "model.safetensors",  # ignored (kept for API compatibility)
        cache_dir: Union[str, None] = "",
        # Optional HF Hub knobs (safe defaults)
        revision: Union[str, None] = None,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,  # bool True -> use cached auth
        allow_patterns: Union[list, None] = None,
        **kwargs,
    ):
        """
        Sharded-only loader that mirrors from_quantized but reads shards via index.
        """
        if _st_load is None:
            raise ImportError("safetensors not installed. `pip install safetensors`")
        if os.path.isdir(save_dir_or_hub):
            save_dir = save_dir_or_hub
        else:
            # Treat as HF Hub repo id
            if snapshot_download is None:
                raise ValueError(
                    "huggingface_hub not installed but a repo id was provided. "
                    "Install it with: pip install huggingface_hub"
                )

            # Keep downloads small: we only need weights + the index + config
            # Add more patterns if your loader reads extra files
            if allow_patterns is None:
                allow_patterns = [
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "*.safetensors.index.json.meta.json",
                    "*.py",
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "vocab.json",
                    "merges.txt",
                    "vocab.txt",
                    "tokenizer.model",
                    "sentencepiece.bpe.model",
                    "spiece.model",
                ]

            save_dir = snapshot_download(
                repo_id=save_dir_or_hub,
                revision=revision,
                cache_dir=cache_dir or None,
                local_files_only=local_files_only,
                token=token,
                allow_patterns=allow_patterns,
                # Symlinks are fine; set False if you need real files
                local_dir=None,
                local_dir_use_symlinks=True,
            )

        model = cls.create_model(save_dir, kwargs)
        model.save_dir = save_dir
        cls.setup_model(model)

        # ---- Load sharded safetensors ON CPU ----
        weights = cls.load_weights_safetensors(
            save_dir,
            map_location="cpu",        # <--- key change: keep shards on CPU
            filename=filename,
        )

        DYNAMIC_TIED = _detect_tied_leaves(model)

        # ---- Preflight (same as from_quantized) ----
        param_leaves = []
        for name, module in model.named_modules():
            if len(module._modules) == 0:
                has_params_or_buffers = any(True for _ in module.parameters(recurse=False)) or \
                                        any(b is not None for b in module.buffers(recurse=False))
                if has_params_or_buffers:
                    param_leaves.append(name)

        missing = [n for n in param_leaves if (n not in weights and n not in DYNAMIC_TIED)]

        if len(missing) > 0:
            preview = ", ".join(missing[:10])
            raise RuntimeError(
                f"[SINQ] {len(missing)} parameterized leaf modules are missing from saved weights "
                f"(examples: {preview}). This will cause wrong outputs. "
                "Fix serialize_weights() so these leaves are included."
            )
        # ---- End preflight ----

        @torch.no_grad()
        def _load_module(module, params=None):
            if module.name not in weights:
                has_params_or_buffers = any(True for _ in module.parameters(recurse=False)) or \
                                        any(b is not None for b in module.buffers(recurse=False))
                if has_params_or_buffers:
                    if module.name in DYNAMIC_TIED:
                        module.device = device
                        return module
                    raise RuntimeError(f"Missing weights for parameterized leaf: {module.name}")
                module.device = device
                return module

            state_dict = weights[module.name]

            # Quantized SINQLinear
            if "W_q" in state_dict:
                m = SINQLinear(
                    linear_layer=None,
                    quant_config=None,
                    compute_dtype=compute_dtype,
                    device=device,
                )
                m.load_state_dict(state_dict, strict=True)
                m.device = device
                return m

            # Regular leaf with tensors
            for key, tensor in state_dict.items():
                is_param = (key in getattr(module, "_parameters", {}) and getattr(module, "_parameters")[key] is not None)
                is_buffer = key in getattr(module, "_buffers", {})

                if is_param:
                    # cast params to compute_dtype and move to `device`
                    t = tensor.to(device=device, dtype=compute_dtype, non_blocking=True)
                    setattr(module, key, nn.Parameter(t, requires_grad=False))
                elif is_buffer:
                    # keep original buffer dtype, just move device
                    t = tensor.to(device=device, dtype=tensor.dtype, non_blocking=True)
                    module._buffers[key] = t
                else:
                    t = tensor.to(device=device, non_blocking=True)
                    setattr(module, key, t)

            module.device = device
            return module

        # Patch all leaves
        cls.patch_model(model, _load_module, _load_module, {k: None for k in model.linear_tags})

        if hasattr(cls, "post_module_load"):
            cls.post_module_load(model, weights)

        # Re-tie after modules are in place
        _retie_tied_leaves(model, saved_weights=weights)
        model.sinq_quantized = True
        model.base_class = cls
        model.eval()

        # ---- Free raw weights dict to release CPU/GPU caches ----
        del weights
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return model

    # -------------------------------------------------------------------------------

class BaseSINQHFModel(BaseSINQModel):
    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        # Update model architecture in the config
        model.config.architectures = [model.__class__.__name__]
        # Save config
        model.config.save_pretrained(save_dir)

    # Save tokenizer assets alongside the model
    @classmethod
    def save_tokenizer_assets(cls, tokenizer, save_dir: str):
        """
        Persist the complete tokenizer bundle so that ALL runtime behavior
        (incl. model_max_length, special tokens, merges/SPM, added tokens, etc.)
        is preserved across save/load and on the Hub.

        Primary path: tokenizer.save_pretrained(save_dir)
        Fallback:     manual copy of the common files + a rich tokenizer_config.json
        """
        if tokenizer is None:
            return

        os.makedirs(save_dir, exist_ok=True)

        # --- Primary: let HF do the right thing
        try:
            tokenizer.save_pretrained(save_dir)
            return
        except Exception:
            # fall back to a careful manual writer
            pass

        # --- Fallback path (manual)
        # 1) Write tokenizer.json if we can
        tok_json = getattr(tokenizer, "tokenizer_file", None)
        if tok_json and os.path.isfile(tok_json):
            shutil.copy(tok_json, os.path.join(save_dir, "tokenizer.json"))
        elif hasattr(tokenizer, "backend_tokenizer"):
            with open(os.path.join(save_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
                f.write(tokenizer.backend_tokenizer.to_str())

        # 2) Copy family-specific vocab/merges/SPM files if they exist
        #    (covering BPE/WordPiece/SentencePiece variants)
        possible_files = [
            "vocab.json", "merges.txt", "vocab.txt",
            "tokenizer.model", "sentencepiece.bpe.model", "spiece.model",
        ]
        for attr in ["vocab_file", "merges_file", "sp_model_file"]:
            path = getattr(tokenizer, attr, None)
            if path and os.path.isfile(path):
                basename = os.path.basename(path)
                # normalize to common HF names when possible
                if basename.endswith(".model") and not basename.startswith("tokenizer"):
                    basename = "tokenizer.model"
                shutil.copy(path, os.path.join(save_dir, basename))

        # 3) Persist special tokens & added tokens
        stm = getattr(tokenizer, "special_tokens_map", None)
        if stm:
            with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
                json.dump(stm, f, indent=2, ensure_ascii=False)

        #   added tokens
        added = []
        try:
            # fast/slow tokenizers expose different internals; this works broadly
            if getattr(tokenizer, "added_tokens_encoder", None):
                added = list(tokenizer.added_tokens_encoder.keys())
            elif getattr(tokenizer, "added_tokens", None):
                added = [t.content if hasattr(t, "content") else str(t)
                         for t in tokenizer.added_tokens]
        except Exception:
            pass
        if added:
            with open(os.path.join(save_dir, "added_tokens.json"), "w", encoding="utf-8") as f:
                json.dump(added, f, indent=2, ensure_ascii=False)

        # 4) Write a rich tokenizer_config.json (include behavior-critical fields)
        #    Note: keep keys aligned with HF to avoid surprises on load.
        def _maybe_int(x, default=None):
            try:
                return int(x)
            except Exception:
                return default

        cfg = {
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_max_length": _maybe_int(getattr(tokenizer, "model_max_length", None)),
            "padding_side": getattr(tokenizer, "padding_side", "right"),
            "truncation_side": getattr(tokenizer, "truncation_side", "right"),
            "clean_up_tokenization_spaces": getattr(tokenizer, "clean_up_tokenization_spaces", True),
            "unk_token": getattr(getattr(tokenizer, "unk_token", None), "content", None) or getattr(tokenizer, "unk_token", None),
            "bos_token": getattr(getattr(tokenizer, "bos_token", None), "content", None) or getattr(tokenizer, "bos_token", None),
            "eos_token": getattr(getattr(tokenizer, "eos_token", None), "content", None) or getattr(tokenizer, "eos_token", None),
            "pad_token": getattr(getattr(tokenizer, "pad_token", None), "content", None) or getattr(tokenizer, "pad_token", None),
            "sep_token": getattr(getattr(tokenizer, "sep_token", None), "content", None) or getattr(tokenizer, "sep_token", None),
            "cls_token": getattr(getattr(tokenizer, "cls_token", None), "content", None) or getattr(tokenizer, "cls_token", None),
            "mask_token": getattr(getattr(tokenizer, "mask_token", None), "content", None) or getattr(tokenizer, "mask_token", None),
        }
        # drop Nones
        cfg = {k: v for k, v in cfg.items() if v is not None}
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Create empty model from config
    @classmethod
    def create_model(cls, save_dir, kwargs):
        model_kwargs = {}
        for key in ["attn_implementation"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        config = transformers.AutoConfig.from_pretrained(save_dir)

        auto_class = transformers.AutoModel

        # Todo: add support for other auto models
        archs = config.architectures
        if len(archs) == 1:
            if ("CausalLM" in archs[0]):
                auto_class = transformers.AutoModelForCausalLM
            elif ("SequenceClassification" in archs[0]):
                auto_class = transformers.AutoModelForSequenceClassification

        with init_empty_weights():
            model = auto_class.from_config(config, **model_kwargs)

        return model

class QwenSINQHFModel(BaseSINQHFModel, BasePatch):
    @classmethod
    def extra_serialize_weights(cls, model, weights: dict):
        for name, module in model.named_modules():
            if module.__class__.__name__ != "Qwen3NextGatedDeltaNet":
                continue

            extra = {}
            for attr in ["A_log", "dt_bias"]:
                if hasattr(module, attr):
                    val = getattr(module, attr)
                    if isinstance(val, torch.Tensor):
                        extra[attr] = val.detach().clone().cpu()

            if extra:
                if name in weights and isinstance(weights[name], dict):
                    weights[name].update(extra)
                else:
                    weights[name] = extra

    @classmethod
    def post_module_load(cls, model, weights):
        import torch
        from torch import nn

        device = next(model.parameters()).device
        core = model.model if hasattr(model, "model") else model

        for name, module in core.named_modules():
            if module.__class__.__name__ != "Qwen3NextGatedDeltaNet":
                continue
                
            full_name = name
            if full_name not in weights and hasattr(model, "model"):
                candidate = f"model.{name}"
                if candidate in weights:
                    full_name = candidate

            extras = weights.get(full_name, None)
            if not isinstance(extras, dict):
                continue

            with torch.no_grad():
                for attr in ["A_log", "dt_bias"]:
                    t = extras.get(attr, None)
                    if t is None or not isinstance(t, torch.Tensor):
                        continue

                    # keep dtype as saved (usually bf16), just move to device
                    t = t.to(device=device, non_blocking=True)

                    old = getattr(module, attr, None)
                    if isinstance(old, nn.Parameter):
                        setattr(module, attr, nn.Parameter(t, requires_grad=False))
                    else:
                        setattr(module, attr, t)

class KimiSINQHFModel(BaseSINQHFModel, BasePatch):
    @classmethod
    def extra_serialize_weights(cls, model, weights: dict):
        for name, module in model.named_modules():
            if module.__class__.__name__ != "KimiDeltaAttention":
                continue

            extra = {}
            for attr in ["A_log", "dt_bias"]:
                if hasattr(module, attr):
                    val = getattr(module, attr)
                    if isinstance(val, torch.Tensor):
                        extra[attr] = val.detach().clone().cpu()

            if extra:
                if name in weights and isinstance(weights[name], dict):
                    weights[name].update(extra)
                else:
                    weights[name] = extra

    @classmethod
    def post_module_load(cls, model, weights):
        import torch
        from torch import nn

        device = next(model.parameters()).device
        core = model.model if hasattr(model, "model") else model

        for name, module in core.named_modules():
            if module.__class__.__name__ != "KimiDeltaAttention":
                continue
                
            full_name = name
            if full_name not in weights and hasattr(model, "model"):
                candidate = f"model.{name}"
                if candidate in weights:
                    full_name = candidate

            extras = weights.get(full_name, None)
            if not isinstance(extras, dict):
                continue

            with torch.no_grad():
                for attr in ["A_log", "dt_bias"]:
                    t = extras.get(attr, None)
                    if t is None or not isinstance(t, torch.Tensor):
                        continue

                    # keep dtype as saved (usually bf16), just move to device
                    t = t.to(device=device, non_blocking=True)

                    old = getattr(module, attr, None)
                    if isinstance(old, nn.Parameter):
                        setattr(module, attr, nn.Parameter(t, requires_grad=False))
                    else:
                        setattr(module, attr, t)

# Auto class used for HF models if no architecture was manually setup
class AutoSINQHFModel(BaseSINQHFModel, BasePatch):
    """
    Generic entrypoint.

    You *always* call:

      - AutoSINQHFModel.save_quantized(...)
      - AutoSINQHFModel.save_quantized_safetensors(...)
      - AutoSINQHFModel.from_quantized(...)
      - AutoSINQHFModel.from_quantized_safetensors(...)

    Internally this chooses QwenSINQHFModel / KimiSINQHFModel / AutoSINQHFModel
    based on the model (for saving) or the config (for loading).
    """
    @classmethod
    def save_quantized(
        cls,
        model,
        tokenizer,
        save_dir: str,
        verbose: bool = False,
        write_tokenizer: bool = True,
    ):
        sinq_cls = _pick_sinq_class(model=model)

        if sinq_cls is cls:
            # Use BaseSINQHFModel's implementation for the generic case
            return super().save_quantized(
                model,
                tokenizer,
                save_dir,
                verbose=verbose,
                write_tokenizer=write_tokenizer,
            )

        print(f'{sinq_cls} selected!', flush=True)
        # Forward to model-specific class (e.g. QwenSINQHFModel / KimiSINQHFModel)
        return sinq_cls.save_quantized(
            model,
            tokenizer,
            save_dir,
            verbose=verbose,
            write_tokenizer=write_tokenizer,
        )

    @classmethod
    def save_quantized_safetensors(
        cls,
        model,
        tokenizer,
        save_dir: str,
        filename: str = "model.safetensors",
        verbose: bool = False,
        max_shard_size: str = "4GB",
        write_tokenizer: bool = True,
    ):
        sinq_cls = _pick_sinq_class(model=model)

        if sinq_cls is cls:
            return super().save_quantized_safetensors(
                model,
                tokenizer,
                save_dir,
                filename=filename,
                verbose=verbose,
                max_shard_size=max_shard_size,
                write_tokenizer=write_tokenizer,
            )
        
        print(f'{sinq_cls} selected!', flush=True)
        return sinq_cls.save_quantized_safetensors(
            model,
            tokenizer,
            save_dir,
            filename=filename,
            verbose=verbose,
            max_shard_size=max_shard_size,
            write_tokenizer=write_tokenizer,
        )

    @classmethod
    def from_quantized(
        cls,
        save_dir_or_hub,
        compute_dtype: torch.dtype = float16,
        device="cuda",
        cache_dir: Union[str, None] = "",
        **kwargs,
    ):
        sinq_cls = _pick_sinq_class(
            save_dir_or_hub=save_dir_or_hub,
            cache_dir=cache_dir or None,
            revision=None,
            local_files_only=False,
            token=None,
        )

        if sinq_cls is cls:
            return super().from_quantized(
                save_dir_or_hub,
                compute_dtype=compute_dtype,
                device=device,
                cache_dir=cache_dir,
                **kwargs,
            )
        
        print(f'{sinq_cls} selected!', flush=True)
        return sinq_cls.from_quantized(
            save_dir_or_hub,
            compute_dtype=compute_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )

    @classmethod
    def from_quantized_safetensors(
        cls,
        save_dir_or_hub,
        compute_dtype: torch.dtype = float16,
        device="cuda",
        filename: str = "model.safetensors",  # kept for API compat
        cache_dir: Union[str, None] = "",
        revision: Union[str, None] = None,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        allow_patterns: Union[list, None] = None,
        **kwargs,
    ):
        sinq_cls = _pick_sinq_class(
            save_dir_or_hub=save_dir_or_hub,
            cache_dir=cache_dir or None,
            revision=revision,
            local_files_only=local_files_only,
            token=(token if isinstance(token, str) else None),
        )

        if sinq_cls is cls:
            return super().from_quantized_safetensors(
                save_dir_or_hub,
                compute_dtype=compute_dtype,
                device=device,
                filename=filename,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
                token=token,
                allow_patterns=allow_patterns,
                **kwargs,
            )
        
        print(f'{sinq_cls} selected!', flush=True)
        return sinq_cls.from_quantized_safetensors(
            save_dir_or_hub,
            compute_dtype=compute_dtype,
            device=device,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=allow_patterns,
            **kwargs,
        )
