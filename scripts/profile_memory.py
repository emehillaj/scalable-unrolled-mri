#!/usr/bin/env python3
"""Peak training memory and step time versus chain depth (and width).

One step is a forward pass through the full chain plus the backward pass of an MSE loss
on a synthetic input (batch 1, no optimizer). Methods:
  ours           shared time-conditioned regularizer, one sampled cascade (j = T // 2) and
                 the activation-light backward through the cascades after it
  e2e            one regularizer per cascade, standard backward through every cascade
  e2e_ckpt       e2e with per-cascade activation checkpointing (same exact gradients, the
                 activations are recomputed in the backward pass)
  shared_fullbp  the shared regularizer of ours with the standard backward through every
                 cascade (training-scheme ablation, the dashed curve of the README figure)
  ours_exact     ours with exact autograd through the cascades after j instead of the
                 activation-light backward (training-scheme ablation)
The first three are profiled by default.
Peak memory is torch.cuda.max_memory_allocated over the warmup and timed steps; static is
the memory held before the first step (parameters and inputs). On the CPU only the step
time is reported. A configuration that runs out of GPU memory is reported as OOM.

Usage (from the repository root):
  python scripts/profile_memory.py --depths 8 15 50 100 --chans 32
  python scripts/profile_memory.py --coils 16 --height 640 --width 320   # brain-like multi-coil
  python scripts/profile_memory.py --device cpu --depths 2 4 --chans 8 --height 64 --width 48
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from modeling import build_model  # noqa: E402

GB = 1024 ** 3
METHODS = ("ours", "e2e", "e2e_ckpt", "shared_fullbp", "ours_exact")
SAMPLED = ("ours", "ours_exact")  # one sampled cascade j = T // 2


def make_inputs(args, device):
    """Synthetic masked k-space, mask (every 8th column plus nlf center columns) and target."""
    H, W = args.height, args.width
    mask = torch.zeros(W, dtype=torch.bool)
    mask[::8] = True
    mask[W // 2 - args.nlf // 2:W // 2 - args.nlf // 2 + args.nlf] = True
    if args.coils == 1:
        kspace = torch.randn(1, H, W, 2)
        mask = mask.view(1, 1, W, 1)
        nlf = None
    else:
        kspace = torch.randn(1, args.coils, H, W, 2)
        mask = mask.view(1, 1, 1, W, 1)
        nlf = args.nlf
    kspace = kspace * mask
    target = torch.randn(1, H, W)
    return kspace.to(device), mask.to(device), nlf, target.to(device)


def build(method, T, chans, args, device):
    cfg = {
        "method": "ours" if method in SAMPLED else "e2e",
        "challenge": "singlecoil" if args.coils == 1 else "multicoil",
        "num_cascades": T,
        "chans": chans,
        "use_time_embed": method in ("ours", "ours_exact", "shared_fullbp"),
        "bitpack_relu_mask": args.bitpack,
        "exact_suffix": method == "ours_exact",
        "activation_checkpointing": method == "e2e_ckpt",
    }
    return build_model(cfg).to(device).train()


def step(model, kspace, mask, nlf, target, j):
    out = model(kspace, mask, nlf, cascade_to_train_idx=j)
    F.mse_loss(out, target).backward()
    model.zero_grad(set_to_none=True)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile(method, T, chans, args, device):
    torch.manual_seed(0)
    cuda = device.type == "cuda"
    row = {"method": method, "T": T, "chans": chans}
    model = None
    try:
        model = build(method, T, chans, args, device)
        row["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        kspace, mask, nlf, target = make_inputs(args, device)
        j = T // 2 if method in SAMPLED else None
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            row["static_GB"] = torch.cuda.memory_allocated(device) / GB
        for _ in range(args.warmup):
            step(model, kspace, mask, nlf, target, j)
        times = []
        for _ in range(args.reps):
            _sync(device)
            t0 = time.perf_counter()
            step(model, kspace, mask, nlf, target, j)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)
        if cuda:
            row["peak_GB"] = torch.cuda.max_memory_allocated(device) / GB
        row["ms_median"] = statistics.median(times)
        row["ms_std"] = statistics.pstdev(times)
    except torch.cuda.OutOfMemoryError:
        row["oom"] = True
    finally:
        del model
        if cuda:
            torch.cuda.empty_cache()
    return row


HEADER = f"{'method':<14}{'T':>5}{'chans':>7}{'params(M)':>11}{'peak(GB)':>10}{'static':>10}{'step(ms)':>12}"


def _fmt(row):
    head = f"{row['method']:<14}{row['T']:>5}{row['chans']:>7}{row.get('params_M', float('nan')):>11.2f}"
    if row.get("oom"):
        return head + f"{'OOM':>10}"
    if "peak_GB" in row:
        mem = f"{row['peak_GB']:>10.3f}{row['static_GB']:>10.3f}"
    else:
        mem = f"{'n/a':>10}{'n/a':>10}"
    return head + mem + f"{row['ms_median']:>12.1f} +- {row['ms_std']:.1f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:3]),
                    help="methods to profile (default: ours e2e e2e_ckpt)")
    ap.add_argument("--depths", nargs="+", type=int, default=[8, 15, 50, 100], help="chain lengths T")
    ap.add_argument("--chans", nargs="+", type=int, default=[32], help="network widths C")
    ap.add_argument("--coils", type=int, default=1,
                    help="1 = single-coil, else multi-coil with this many coils")
    ap.add_argument("--height", type=int, default=640, help="k-space rows (640 for single-coil knee)")
    ap.add_argument("--width", type=int, default=372, help="k-space columns (372 for single-coil knee)")
    ap.add_argument("--nlf", type=int, default=24, help="fully sampled center columns")
    ap.add_argument("--warmup", type=int, default=10, help="untimed steps before the timed ones")
    ap.add_argument("--reps", type=int, default=50, help="timed steps (median and std are reported)")
    ap.add_argument("--no_bitpack", dest="bitpack", action="store_false",
                    help="save ReLU masks as torch.bool, not 1-bit packed (faster, more memory)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=None, help="also write the rows to this JSON file")
    args = ap.parse_args(argv)
    if args.reps < 1:
        ap.error("--reps must be >= 1")
    device = torch.device(args.device)

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu (time only)"
    print(
        f"# {name}: B=1 {args.height}x{args.width} coils={args.coils} bitpack={args.bitpack}, "
        f"median of {args.reps} steps after {args.warmup} warmup steps"
    )
    print(HEADER)
    rows = []
    for method in args.methods:
        for chans in args.chans:
            for T in args.depths:
                row = profile(method, T, chans, args, device)
                rows.append(row)
                print(_fmt(row), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"device": name, "args": vars(args), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
