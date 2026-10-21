#!/usr/bin/env python3
"""Convert a training checkpoint into a slim release checkpoint.

Usage (from the repository root):
  python scripts/convert_checkpoint.py --checkpoint local_data/results/run/checkpoint.pt \\
      --output brain_ours_32ch_15casc_8x.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling import build_model  # noqa: E402
from utils.checkpoint import atomic_save, extract_weights, load_checkpoint  # noqa: E402
from utils.misc import METHODS, load_config, parse_set_overrides  # noqa: E402

# Machine-specific keys that a published checkpoint should not carry.
PATH_KEYS = ("data_path", "output_dir", "root_dir")

def resolve_config(args, ckpt) -> dict:
    overrides = parse_set_overrides(args.set)
    if args.config:
        with open(args.config) as f:
            raw = json.load(f)
        if "method" not in raw:
            raise SystemExit("the config must name a method")
        if args.method:
            overrides["method"] = args.method
    elif isinstance(ckpt.get("config"), dict):
        raw = ckpt["config"]
        if args.method:
            overrides["method"] = args.method
    else:
        raise SystemExit(f"{args.checkpoint} stores no config: pass --config")
    cfg = load_config(raw, overrides)
    for key in PATH_KEYS:
        cfg.pop(key, None)
    for key in ("num_cascades", "chans"):
        if key not in cfg:
            raise SystemExit(f"the config has no '{key}': add it with --set {key}=...")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="training checkpoint")
    ap.add_argument("--output", required=True, help="release checkpoint to write")
    ap.add_argument("--config", default=None, help="release config JSON")
    ap.add_argument("--method", choices=METHODS, default=None, help="override the config method")
    ap.add_argument("--weights", choices=("auto", "ema", "live"), default="auto",
                    help="auto = EMA weights when the checkpoint has them, else live")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, JSON-parsed (repeatable)")
    args = ap.parse_args(argv)

    try:
        ckpt = load_checkpoint(args.checkpoint, mmap=True)
    except RuntimeError:  # legacy (non-zip) checkpoints cannot be memory-mapped
        ckpt = load_checkpoint(args.checkpoint)
    cfg = resolve_config(args, ckpt)
    state_dict, weights = extract_weights(ckpt, args.weights)
    state_dict = {k: v.detach().clone() for k, v in state_dict.items()}

    model = build_model(cfg)
    model.load_state_dict(state_dict, strict=True)
    num_params = sum(p.numel() for p in model.parameters())

    epoch = ckpt.get("epoch")
    keys = ("samples_consumed_in_epoch", "epoch_position")
    position = next((ckpt[k] for k in keys if k in ckpt), None)
    release = {
        "config": cfg,
        "epoch": epoch,
        "epoch_position": position,
        "weights": weights,
        "state_dict": state_dict,
    }
    out = atomic_save(release, args.output)
    print(
        f"wrote {out}: {cfg['method']} {cfg['anatomy']} {cfg['challenge']} C={cfg['chans']} "
        f"T={cfg['num_cascades']} {cfg['acceleration']}x, epoch {epoch} (position {position}), "
        f"{weights} weights, {len(state_dict)} tensors, {num_params / 1e6:.2f}M parameters "
        "(strict load OK)"
    )


if __name__ == "__main__":
    main()
