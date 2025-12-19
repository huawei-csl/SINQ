# -*- coding: utf-8 -*-
"""
HF I/O adapter for SINQ:
- save_pretrained -> BaseSINQHFModel.save_quantized_safetensors (sharded safetensors)
- from_pretrained -> BaseSINQHFModel.from_quantized_safetensors (reads sharded index)
- push_to_hub     -> save to tmp + upload_folder

Usage options:
  A) Global patch (affects all PreTrainedModel):
      from sinq.hf_io import patch_hf_pretrained_io
      patch_hf_pretrained_io()   # call once at startup

  B) Opt-in subclass:
      from sinq.hf_io import SinqHFIO
      class MyModel(SinqHFIO, transformers.AutoModelForCausalLM): pass
      model = MyModel.from_pretrained("...", ...)

Notes:
- Routing decisions are driven by config flags (no .sinq files required).
- If a caller passes a SINQ-like quantization_config to from_pretrained(), we also route via the SINQ path.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional, Union, List

import torch
from huggingface_hub import (
    create_repo,
    upload_folder,
    snapshot_download as _hf_snapshot_download,
    hf_hub_download as _hf_hub_download,
)
from transformers import PreTrainedModel
try:
    from huggingface_hub.utils import is_offline_mode
except Exception:
    try:
        from transformers.utils.hub import is_offline_mode  # some versions
    except Exception:
        def is_offline_mode() -> bool:
            return bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))

# import your working logic
from .patch_model import BaseSINQHFModel, AutoSINQHFModel

SAFE_INDEX = "model.safetensors.index.json"
HF_CONFIG = "config.json"

# -----------------------------------------------------------------------------
# Optional custom downloader (no special file types needed)
# -----------------------------------------------------------------------------
_SINQ_FORCE_DOWNLOADER = os.getenv("SINQ_FORCE_DOWNLOADER", "").strip().lower() in ("1", "true", "yes", "on")
_SINQ_PATCH_HUB = os.getenv("SINQ_PATCH_HUB", "").strip().lower() in ("1", "true", "yes", "on")

try:
    # Provide these in your package if you have a custom transport/cache:
    #   sinq.download.snapshot_download(...)
    #   sinq.download.hf_hub_download(...)
    from sinq import download as _sinq_dl  # type: ignore
except Exception:
    _sinq_dl = None


def _norm_qmethod(x: Any) -> str:
    if x is None:
        return ""
    if hasattr(x, "value"):  # Enum
        try:
            x = x.value
        except Exception:
            pass
    s = str(x).strip().lower()
    return "sinq" if "sinq" in s else s

def _prefer_sinq(repo_id: Optional[str], hinted_sinq: bool) -> bool:
    """Final gate to use custom downloader."""
    if _SINQ_FORCE_DOWNLOADER:
        return True
    if hinted_sinq:
        return True
    # Optional: repo-name heuristic (conservative)
    rid = (repo_id or "").lower()
    return any(tok in rid for tok in ("-sinq", "sinq-"))


def _download_config_json(
    repo_id: str,
    *,
    cache_dir: Optional[str],
    token: Optional[Union[str, bool]],
    revision: Optional[str],
    local_files_only: bool,
) -> Optional[Dict[str, Any]]:
    """
    Cheap single-file fetch to inspect config flags.
    Uses stock hf_hub_download so we don't commit to a snapshot path before deciding.
    """
    try:
        path = _hf_hub_download(
            repo_id=repo_id,
            filename=HF_CONFIG,
            revision=revision,
            cache_dir=cache_dir or None,
            #local_files_only=local_files_only,
            local_files_only=local_files_only or is_offline_mode(),
            token=token,
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _snapshot_download(
    *,
    repo_id: str,
    allow_patterns: Optional[List[str]] = None,
    **kwargs: Any,
) -> str:
    """
    Wrapper that prefers sinq.download.snapshot_download if available/desired,
    else falls back to huggingface_hub.snapshot_download.
    """
    hinted_sinq = kwargs.pop("_hinted_sinq", False)  # internal hint only
    if _sinq_dl is not None and _prefer_sinq(repo_id, hinted_sinq):
        try:
            return _sinq_dl.snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, **kwargs)
        except Exception:
            # Soft fallback to HF
            pass
    return _hf_snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, **kwargs)


# ----------------------------
# Helpers
# ----------------------------
def _is_sinq_sharded_dir(path: str) -> bool:
    # Keep support for sharded safetensors; not required if you don't use them
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, SAFE_INDEX))


def _safe_lower(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def _detect_sinq_from_cfg_dict(cfg: Dict[str, Any]) -> bool:
    if not cfg:
        return False
    qc = cfg.get("quantization_config") or {}

    qm_top = _norm_qmethod(cfg.get("quant_method") or cfg.get("quantization_method"))
    qm_qc  = _norm_qmethod(qc.get("quant_method") or qc.get("quantization_method"))

    return any(
        [
            bool(cfg.get("sinq") or cfg.get("is_sinq") or cfg.get("sinq_quantized")),
            bool(qc.get("sinq") or qc.get("is_sinq") or qc.get("sinq_quantized")),
            qm_top == "sinq",
            qm_qc == "sinq",
            _safe_lower(qc.get("backend")) == "sinq",
            _safe_lower(qc.get("approach")) == "sinq",
            ("sinq_version" in cfg) or ("sinq_version" in qc),
        ]
    )

def _detect_sinq_from_quant_obj(qobj: Any) -> bool:
    if qobj is None:
        return False
    if hasattr(qobj, "quant_method") and _norm_qmethod(getattr(qobj, "quant_method")) == "sinq":
        return True

    # 1) Type/name/module heuristic
    t = type(qobj)
    name = getattr(t, "__name__", "").lower()
    mod = getattr(t, "__module__", "")
    if "sinq" in name or "sinq" in mod.lower():
        return True

    # 2) Dict-like view
    as_dict: Dict[str, Any] = {}
    if isinstance(qobj, dict):
        as_dict = qobj
    else:
        # Best-effort attribute harvest
        for key in ("method", "backend", "approach", "name", "quantization_method", "quant_method", "nbits", "group_size"):
            if hasattr(qobj, key):
                try:
                    as_dict[key] = getattr(qobj, key)
                except Exception:
                    pass
        # Common property accessors
        if hasattr(qobj, "to_dict") and callable(getattr(qobj, "to_dict")):
            try:
                maybe = qobj.to_dict()
                if isinstance(maybe, dict):
                    as_dict.update(maybe)
            except Exception:
                pass

    s = {k: str(v).lower() for k, v in as_dict.items() if isinstance(v, (str, int, float, bool))}
    return any(
        [
            s.get("sinq") in ("true", "1"),
            s.get("quantization_method") == "sinq",
            s.get("quant_method") == "sinq",
            s.get("method") in ("sinq", "asinq"),
            s.get("backend") == "sinq",
            s.get("approach") == "sinq",
            "sinq" in (s.get("name") or ""),
        ]
    )


def _mark_model_as_sinq(model: PreTrainedModel, quant_obj: Optional[Any] = None) -> None:
    """
    Tag the model so later save/push route to SINQ writer.
    - Set a private sentinel on the instance.
    - Ensure config has a quantization_config section that points to SINQ.
    """
    try:
        setattr(model, "_is_sinq_quantized", True)
        if quant_obj is not None and not hasattr(model, "_sinq_quant_config"):
            setattr(model, "_sinq_quant_config", quant_obj)

        cfg = getattr(model, "config", None)
        if cfg is not None:
            # Normalize into config.quantization_config
            try:
                qcfg = dict(getattr(cfg, "quantization_config", {}) or {})
            except Exception:
                qcfg = {}

            qcfg.setdefault("sinq", True)
            qcfg["backend"] = qcfg.get("backend", "sinq")
            qcfg.setdefault("quantization_method", "sinq")
            qcfg.setdefault("sinq_version", qcfg.get("sinq_version", "auto"))

            # Write-back tolerant of frozen config
            try:
                cfg.quantization_config = qcfg  # type: ignore[attr-defined]
            except Exception:
                try:
                    cfg.__dict__["quantization_config"] = qcfg  # type: ignore[index]
                except Exception:
                    pass

            # Top-level hint
            try:
                setattr(cfg, "sinq", True)
            except Exception:
                try:
                    cfg.__dict__["sinq"] = True  # type: ignore[index]
                except Exception:
                    pass
    except Exception:
        # Never break caller
        pass


def _model_has_sinq_flag(model: PreTrainedModel) -> bool:
    """
    Check the loaded model for a SINQ indicator:
    - private sentinel,
    - model.quantization_config attribute,
    - config & config.quantization_config dicts.
    """
    if getattr(model, "_is_sinq_quantized", False):
        return True

    mq = getattr(model, "quantization_config", None)
    if _detect_sinq_from_quant_obj(mq):
        return True

    cfg = {}
    if hasattr(model, "config") and model.config is not None:
        try:
            cfg = model.config.to_dict()
        except Exception:
            # Attempt a best-effort extraction
            try:
                cfg = dict(getattr(model, "config"))
            except Exception:
                cfg = {}
    return _detect_sinq_from_cfg_dict(cfg)


def _path_has_sinq_flag(local_dir: str) -> bool:
    """
    Look for config.json in a local directory and check SINQ flags.
    """
    cfg_path = os.path.join(local_dir, HF_CONFIG)
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return _detect_sinq_from_cfg_dict(cfg)
    except Exception:
        return False


def _download_if_repo_id(
    pretrained_model_name_or_path: str,
    *,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[list] = None,
    token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    hinted_sinq: bool = False,  # <- hint from caller if they passed a SINQ qconfig
) -> str:
    """
    If it's a repo id, download only what BaseSINQHFModel needs, else return the input unchanged.
    Uses config-first inspection to decide whether to route via custom downloader.
    """
    if os.path.isdir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path

    if is_offline_mode():
        # In offline mode, HF returns cached path or raises if not cached.
        local_dir = _hf_snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir or None,
            local_files_only=True,
            token=token,
            allow_patterns=allow_patterns,
            local_dir=None,
            local_dir_use_symlinks=True,
        )
        return local_dir
        #pass

    # Conservative allow patterns – shards + config + tokenizer
    if allow_patterns is None:
        allow_patterns = [
            "*.safetensors",
            "*.safetensors.index.json",
            "*.safetensors.index.json.meta.json",
            "pytorch_model.bin.index.json",
            HF_CONFIG,
            "README*",
            "*.md",
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

    repo_id = pretrained_model_name_or_path

    # 1) Inspect config.json to decide SINQ (no .sinq files needed)
    cfg = _download_config_json(
        repo_id,
        cache_dir=cache_dir,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
    )
    cfg_is_sinq = _detect_sinq_from_cfg_dict(cfg or {})

    # 2) Choose downloader
    use_sinq_downloader = (_sinq_dl is not None) and _prefer_sinq(repo_id, hinted_sinq or cfg_is_sinq)

    if use_sinq_downloader:
        try:
            return _sinq_dl.snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir or None,
                local_files_only=local_files_only,
                token=token,
                allow_patterns=allow_patterns,
                local_dir=None,
                local_dir_use_symlinks=True,
            )
        except Exception:
            # Soft fallback to HF if custom download fails
            pass

    # Fallback: standard hub downloader
    local_dir = _hf_snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir or None,
        local_files_only=local_files_only,
        token=token,
        allow_patterns=allow_patterns,
        local_dir=None,
        local_dir_use_symlinks=True,
    )

    # Fallback if symlinks are not permitted
    if not os.path.isdir(local_dir):
       local_dir = _hf_snapshot_download(
           repo_id=repo_id,
           revision=revision,
           cache_dir=cache_dir or None,
           local_files_only=local_files_only,
           token=token,
           allow_patterns=allow_patterns,
           local_dir=None,
           local_dir_use_symlinks=False,
       )
    return local_dir


# ----------------------------
# Mixin
# ----------------------------
class SinqHFIO:
    """
    Mixin that overrides HF I/O to use SINQ sharded safetensors when flagged as SINQ.
    The routing decision is:
      - SAVE / PUSH: if the *model's config* says SINQ (or marked in-memory) -> use SINQ writer; else fall back to HF.
      - LOAD       : if the *source path's config* or presence of SINQ index says SINQ -> use SINQ loader; else fall back to HF.
    """

    # ----------------------------
    # SAVE API
    # ----------------------------
    def save_pretrained(
        self: PreTrainedModel,
        save_directory: str,
        *,
        safe_serialization: bool = True,
        max_shard_size: Union[int, str] = "4GB",
        save_tokenizer: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        If the model is SINQ-quantized (via config flag), save as SINQ sharded safetensors;
        otherwise, delegate to HF's default save_pretrained.
        """
        os.makedirs(save_directory, exist_ok=True)

        tokenizer = kwargs.pop("tokenizer", None)

        if _model_has_sinq_flag(self) and safe_serialization:
            # Sharded safetensors path (your proven flow)
            AutoSINQHFModel.save_quantized_safetensors(
                model=self,
                tokenizer=tokenizer,
                save_dir=save_directory,
                filename="model.safetensors",  # writer uses index+shards
                max_shard_size=max_shard_size,
                verbose=kwargs.pop("verbose", False),
                write_tokenizer=save_tokenizer,
            )
            return

        # Fallback: either not SINQ or user disabled safe serialization
        return super(SinqHFIO, self).save_pretrained(  # type: ignore[misc]
            save_directory,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
            **({"tokenizer": tokenizer} if tokenizer is not None else {}),
            **kwargs,
        )

    # ----------------------------
    # PUSH API
    # ----------------------------
    def push_to_hub(
        self: PreTrainedModel,
        repo_id: str,
        *,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[str] = None,
        create_pr: bool = False,
        branch: Optional[str] = None,
        safe_serialization: bool = True,
        max_shard_size: Union[int, str] = "4GB",
        save_tokenizer: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        If SINQ-quantized by config, save to a temp dir via SINQ save_pretrained(), then upload_folder(...).
        Otherwise, delegate to HF.
        """
        if not (_model_has_sinq_flag(self) and safe_serialization):
            return super(SinqHFIO, self).push_to_hub(  # type: ignore[misc]
                repo_id,
                commit_message=commit_message,
                private=private,
                token=token,
                create_pr=create_pr,
                branch=branch,
                safe_serialization=safe_serialization,
                max_shard_size=max_shard_size,
                **kwargs,
            )

        # Ensure repo exists (SINQ path)
        create_repo(repo_id, exist_ok=True, private=private, token=token)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(
                tmp,
                safe_serialization=True,  # SINQ path uses safetensors
                max_shard_size=max_shard_size,
                save_tokenizer=save_tokenizer,
                **kwargs,
            )
            info = upload_folder(
                repo_id=repo_id,
                folder_path=tmp,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=branch,
            )
        #return repo_id
        # Mirror HF's API behavior:
        try:
            return getattr(info, "commit_url", None) or getattr(info, "url", None) or repo_id
        except Exception:
            return repo_id

    # ----------------------------
    # LOAD API
    # ----------------------------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ):
        """
        If config/index indicates SINQ sharded safetensors (locally or on the Hub),
        load via BaseSINQHFModel.from_quantized_safetensors. Otherwise, fallback to HF.
        Also: if the caller passed a SINQ-looking quantization_config, mark the model as SINQ after HF load.
        """
        # Pull knobs we pass to the HF Hub & SINQ loader
        cache_dir: Optional[str] = kwargs.pop("cache_dir", None)
        revision: Optional[str] = kwargs.pop("revision", None)
        local_files_only: bool = kwargs.pop("local_files_only", False)
        token: Optional[Union[str, bool]] = kwargs.pop("token", None)

        # Detect if user asked for SINQ quantization in-memory
        user_qcfg = kwargs.get("quantization_config", None)
        user_qcfg_is_sinq = _detect_sinq_from_quant_obj(user_qcfg)

        compute_dtype = kwargs.get(
            "compute_dtype",
            torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        device = kwargs.get(
            "device_map",
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        # If it's a repo id, download just what we need (config + potential shards/index)
        local_path = _download_if_repo_id(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            hinted_sinq=user_qcfg_is_sinq,  # hint: caller's qconfig looks SINQ
        )

        # Decide whether to use SINQ loader:
        # 1) Explicit SINQ sharded index present, or
        # 2) Config has the SINQ flag.
        use_sinq = _is_sinq_sharded_dir(local_path) and _path_has_sinq_flag(local_path)
        print(f'Use sinq flag is {use_sinq}', flush=True)
        
        if use_sinq:
            model = AutoSINQHFModel.from_quantized_safetensors(
                save_dir_or_hub=local_path,
                compute_dtype=compute_dtype,
                device=device,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=True,  # we already downloaded
                token=token,
                **kwargs,
            )
            return model

        # Fallback: HF default path
        model = super(SinqHFIO, cls).from_pretrained(  # type: ignore[misc]
            pretrained_model_name_or_path,
            *model_args,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            **kwargs,
        )

        # If caller provided a SINQ-ish quantization_config, mark the model so future save/push route via SINQ
        if user_qcfg_is_sinq:
            _mark_model_as_sinq(model, user_qcfg)

        return model


# ----------------------------
# Global patch (optional)
# ----------------------------
_ORIG_SAVE = None
_ORIG_PUSH = None
_ORIG_FROM = None
_ORIG_HUB_SNAP = None
_ORIG_HUB_FILE = None


def patch_hf_pretrained_io() -> None:
    """
    Monkey-patch PreTrainedModel to *conditionally* route save/load/push via SINQ I/O
    when the SINQ flag is present in config (or index exists for load) or when the caller
    passes a SINQ-looking quantization_config at load time.

    If env SINQ_PATCH_HUB=1, also patch huggingface_hub.snapshot_download / hf_hub_download
    to prefer your custom downloader based on config-first inspection/hints.
    """
    global _ORIG_SAVE, _ORIG_PUSH, _ORIG_FROM, _ORIG_HUB_SNAP, _ORIG_HUB_FILE
    if _ORIG_SAVE is not None:
        return  # already patched

    _ORIG_SAVE = PreTrainedModel.save_pretrained
    _ORIG_PUSH = PreTrainedModel.push_to_hub
    _ORIG_FROM = PreTrainedModel.from_pretrained

    def _save_pretrained(self, save_directory, **kwargs):
        try:
            if _model_has_sinq_flag(self) and kwargs.get("safe_serialization", True):
                return SinqHFIO.save_pretrained(self, save_directory, **kwargs)
        except Exception:
            pass
        return _ORIG_SAVE(self, save_directory, **kwargs)

    def _push_to_hub(self, repo_id, **kwargs):
        try:
            if _model_has_sinq_flag(self) and kwargs.get("safe_serialization", True):
                return SinqHFIO.push_to_hub(self, repo_id, **kwargs)
        except Exception:
            pass
        return _ORIG_PUSH(self, repo_id, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        cache_dir = kwargs.get("cache_dir")
        revision = kwargs.get("revision")
        local_files_only = kwargs.get("local_files_only", False)
        token = kwargs.get("token")
        user_qcfg = kwargs.get("quantization_config", None)
        user_qcfg_is_sinq = _detect_sinq_from_quant_obj(user_qcfg)

        #_from_pretrained.__func__._sinq_wrapped = True      # type: ignore[attr-defined]
        #_from_pretrained.__func__.__module__ = "sinq.hf_io"

        try:
            local_path = _download_if_repo_id(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
                hinted_sinq=user_qcfg_is_sinq,
            )
            if _is_sinq_sharded_dir(local_path) and _path_has_sinq_flag(local_path):
                # Use SINQ loader path
                print(f'Using SINQ patch functions to download SINQ-quantized models', flush=True)
                return SinqHFIO.from_pretrained.__func__(cls, local_path, *args, **kwargs)  # type: ignore
        except Exception:
            # If anything goes wrong in detection, fall back to original behavior.
            pass

        # Fallback to original HF load
        model = _ORIG_FROM.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)  # type: ignore

        # Mark as SINQ if the caller passed a SINQ quantization_config (in-memory flow)
        if user_qcfg_is_sinq:
            _mark_model_as_sinq(model, user_qcfg)
        return model

    # Assign monkey patches on PreTrainedModel
    PreTrainedModel.save_pretrained = _save_pretrained  # type: ignore[assignment]
    PreTrainedModel.push_to_hub = _push_to_hub          # type: ignore[assignment]
    PreTrainedModel.from_pretrained = _from_pretrained  # type: ignore[assignment]

    # --- Optional: also patch huggingface_hub entry points globally -----------
    if _SINQ_PATCH_HUB:
        try:
            import huggingface_hub as _hub  # type: ignore

            if _ORIG_HUB_SNAP is None:
                _ORIG_HUB_SNAP = _hub.snapshot_download
            if _ORIG_HUB_FILE is None:
                _ORIG_HUB_FILE = _hub.hf_hub_download

            def _wrap_snap(orig):
                def _w(*args, **kwargs):
                    # Extract repo_id and allow_patterns from positional/kw args
                    repo_id = kwargs.get("repo_id") or (args[0] if args else "")
                    allow_patterns = kwargs.get("allow_patterns")
                    hinted = kwargs.pop("_hinted_sinq", False)
                    # We do a quick config peek if not hinted (avoid extra calls if possible)
                    hinted_cfg = hinted
                    if not hinted and isinstance(repo_id, str) and repo_id and not kwargs.get("local_files_only", False):
                        try:
                            cfg = _download_config_json(
                                repo_id,
                                cache_dir=kwargs.get("cache_dir"),
                                token=kwargs.get("token"),
                                revision=kwargs.get("revision"),
                                local_files_only=kwargs.get("local_files_only", False),
                            )
                            hinted_cfg = _detect_sinq_from_cfg_dict(cfg or {})
                        except Exception:
                            hinted_cfg = False
                    if _sinq_dl is not None and _prefer_sinq(repo_id, hinted_cfg):
                        try:
                            return _sinq_dl.snapshot_download(*args, **kwargs)
                        except Exception:
                            pass
                    return orig(*args, **kwargs)
                return _w

            def _wrap_file(orig):
                def _w(*args, **kwargs):
                    repo_id = kwargs.get("repo_id") or (args[0] if args else "")
                    filename = kwargs.get("filename") or (args[1] if len(args) > 1 else "")
                    hinted = kwargs.pop("_hinted_sinq", False)

                    if filename and os.path.basename(filename) == HF_CONFIG:
                        return orig(*args, **kwargs)

                    if _sinq_dl is not None and _prefer_sinq(repo_id, hinted):
                        try:
                            return _sinq_dl.hf_hub_download(*args, **kwargs)
                        except Exception:
                            pass
                    return orig(*args, **kwargs)
                return _w

            _hub.snapshot_download = _wrap_snap(_hub.snapshot_download)  # type: ignore[assignment]
            _hub.hf_hub_download = _wrap_file(_hub.hf_hub_download)      # type: ignore[assignment]
        except Exception:
            # Non-fatal if hub patching fails
            pass


def unpatch_hf_pretrained_io() -> None:
    """
    Restore original HF methods (and hub functions if patched).
    """
    global _ORIG_SAVE, _ORIG_PUSH, _ORIG_FROM, _ORIG_HUB_SNAP, _ORIG_HUB_FILE
    if _ORIG_SAVE is not None:
        try:
            PreTrainedModel.save_pretrained = _ORIG_SAVE  # type: ignore[assignment]
            PreTrainedModel.push_to_hub = _ORIG_PUSH      # type: ignore[assignment]
            PreTrainedModel.from_pretrained = _ORIG_FROM  # type: ignore[assignment]
        except Exception:
            pass
        _ORIG_SAVE = _ORIG_PUSH = _ORIG_FROM = None

    # restore hub
    if _ORIG_HUB_SNAP is not None or _ORIG_HUB_FILE is not None:
        try:
            import huggingface_hub as _hub  # type: ignore
            if _ORIG_HUB_SNAP is not None:
                _hub.snapshot_download = _ORIG_HUB_SNAP  # type: ignore[assignment]
            if _ORIG_HUB_FILE is not None:
                _hub.hf_hub_download = _ORIG_HUB_FILE    # type: ignore[assignment]
        except Exception:
            pass
        _ORIG_HUB_SNAP = _ORIG_HUB_FILE = None
