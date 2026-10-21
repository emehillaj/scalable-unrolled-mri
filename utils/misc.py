"""Seeding, config loading and a JSON-lines logger."""

import copy
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch

METHODS = ("ours", "e2e", "gleam", "flat")
ANATOMIES = ("knee", "brain")
CHALLENGES = ("singlecoil", "multicoil")

# fastMRI default center fraction per acceleration
CENTER_FRACTIONS = {4: 0.08, 8: 0.04}

KNOWN_KEYS = frozenset({
    "method", "anatomy", "challenge", "acceleration", "center_fraction", "singlecoil",
    "num_cascades", "chans", "pools", "sens_chans", "sens_pools", "use_time_embed",
    "time_embedded_unet_arch", "bitpack_relu_mask", "exact_suffix", "activation_checkpointing",
    "zero_init_final_conv", "loss", "mse_scale", "crop_loss", "optimizer", "adam_eps",
    "learning_rate", "weight_decay", "effective_batch_size", "grad_clip_norm",
    "coordinate_cascade_sampling", "use_ema", "ema_decay", "max_epochs", "save_every_minutes",
    "val_slices", "num_workers", "seed", "train_file_list", "run_name", "log_every", "data_path",
})


def run_name(cfg: Dict[str, Any]) -> str:
    """cfg["run_name"], else <anatomy>_<method>_<chans>ch_<num_cascades>casc_<acceleration>x."""
    if cfg.get("run_name"):
        return str(cfg["run_name"])
    acc = cfg["acceleration"]
    acc = int(acc) if float(acc).is_integer() else acc
    return f"{cfg['anatomy']}_{cfg['method']}_{cfg['chans']}ch_{cfg['num_cascades']}casc_{acc}x"


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _from_legacy_lists(cfg: Dict[str, Any]) -> None:
    """Map the older list keys (accelerations, center_fractions) to scalar keys, in place."""
    for old, new in (("accelerations", "acceleration"), ("center_fractions", "center_fraction")):
        if old not in cfg:
            continue
        values = cfg.pop(old)
        if new in cfg:
            continue
        values = list(values) if isinstance(values, (list, tuple)) else [values]
        if len(values) != 1:
            raise ValueError(f"'{old}' must hold exactly one value, got {values}")
        cfg[new] = values[0]


def finalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a config in place and fill center_fraction and singlecoil."""
    for key, allowed in (("method", METHODS), ("anatomy", ANATOMIES), ("challenge", CHALLENGES)):
        if key not in cfg:
            raise KeyError(f"config is missing '{key}' (one of {allowed})")
        if cfg[key] not in allowed:
            raise ValueError(f"config '{key}' must be one of {allowed}, got {cfg[key]!r}")
    if cfg["anatomy"] == "brain" and cfg["challenge"] == "singlecoil":
        raise ValueError("fastMRI brain data is multi-coil only (challenge 'multicoil')")

    if "acceleration" not in cfg:
        raise KeyError("config is missing 'acceleration'")
    acc = cfg["acceleration"]
    if isinstance(acc, bool) or not isinstance(acc, (int, float)) or acc <= 0:
        raise ValueError(f"'acceleration' must be a positive number, got {acc!r}")
    if cfg.get("center_fraction") is None:
        if acc not in CENTER_FRACTIONS:
            raise ValueError(
                f"no default center_fraction for acceleration {acc}; set 'center_fraction' "
                f"explicitly (defaults exist for {sorted(CENTER_FRACTIONS)})"
            )
        cfg["center_fraction"] = CENTER_FRACTIONS[acc]
    cf = cfg["center_fraction"]
    if isinstance(cf, bool) or not isinstance(cf, (int, float)) or not 0 < cf < 1:
        raise ValueError(f"'center_fraction' must be in (0, 1), got {cf!r}")

    cfg["singlecoil"] = cfg["challenge"] == "singlecoil"
    unknown = sorted(set(cfg) - KNOWN_KEYS)
    if unknown:
        warnings.warn(f"unknown config keys, ignored: {unknown}")
    return cfg


def load_config(
    path: Union[str, Path, Dict[str, Any]], overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load a JSON config (or copy a dict), apply overrides verbatim and finalize it.

    center_fraction is derived from the acceleration when absent, also when the overrides
    change the acceleration.
    """
    if isinstance(path, dict):
        cfg = copy.deepcopy(path)
    else:
        with open(path) as f:
            cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a JSON object")
    _from_legacy_lists(cfg)

    overrides = dict(overrides or {})
    _from_legacy_lists(overrides)
    if (
        "acceleration" in overrides
        and "center_fraction" not in overrides
        and overrides["acceleration"] != cfg.get("acceleration")
    ):
        cfg.pop("center_fraction", None)
    cfg.update(overrides)
    return finalize_config(cfg)


def parse_set_overrides(items: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Parse `--set KEY=VALUE` items. VALUE is JSON, or a plain string if it is not valid JSON."""
    out: Dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _to_json(obj: Any) -> Any:
    if isinstance(obj, (np.generic, np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


class JsonlLogger:
    """Append-only JSON-lines log. path=None discards everything (for non-zero ranks)."""

    def __init__(self, path: Optional[Union[str, Path]]):
        self.path = None if path is None else Path(path)
        self._file = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")

    def log(self, record: Optional[Dict[str, Any]] = None, **fields: Any) -> None:
        if self._file is None:
            return
        entry = dict(record or {})
        entry.update(fields)
        self._file.write(json.dumps(entry, default=_to_json) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
