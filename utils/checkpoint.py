"""Checkpoint I/O shared by training, evaluation and the conversion script.

extract_weights reads training checkpoints (state_dict, model_ema_state_dict),
release checkpoints (state_dict plus weights = "ema" | "live"), and plain state
dictionaries.
"""

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import torch

EMA_KEY = "model_ema_state_dict"
LIVE_KEY = "state_dict"


def atomic_save(obj: Any, path: Union[str, Path]) -> Path:
    """torch.save to a temporary file, then rename it, so a crash keeps the previous file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: Union[str, Path], mmap: bool = False) -> Dict[str, Any]:
    """Load a checkpoint onto the CPU. mmap=True avoids reading every tensor up front."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)


def _is_state_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and len(obj) > 0 and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()
    )


def extract_weights(ckpt: Mapping[str, Any], which: str = "auto") -> Tuple["OrderedDict[str, Any]", str]:
    """Return (state_dict, "ema" | "live"). "auto" prefers EMA; absent weights raise KeyError."""
    if which not in ("auto", "ema", "live"):
        raise ValueError(f"which must be 'auto', 'ema' or 'live', got {which!r}")

    available: Dict[str, Mapping[str, Any]] = {}
    ema = ckpt.get(EMA_KEY)
    if ema:
        available["ema"] = ema
    if ckpt.get(LIVE_KEY):
        kind = ckpt.get("weights", "live")
        if kind not in ("ema", "live"):
            raise ValueError(f"checkpoint 'weights' must be 'ema' or 'live', got {kind!r}")
        if kind in available:
            raise ValueError(f"checkpoint has both '{EMA_KEY}' and EMA weights in '{LIVE_KEY}'")
        available[kind] = ckpt[LIVE_KEY]
    if not available and _is_state_dict(ckpt):
        available["live"] = ckpt
    if not available:
        raise KeyError(f"no '{LIVE_KEY}' or '{EMA_KEY}' in checkpoint (keys: {list(ckpt)[:10]})")

    if which == "auto":
        which = "ema" if "ema" in available else "live"
    if which not in available:
        raise KeyError(f"checkpoint has no {which} weights (available: {sorted(available)})")
    return dict(available[which]), which
