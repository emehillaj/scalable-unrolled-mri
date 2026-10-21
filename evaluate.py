#!/usr/bin/env python3
"""Evaluate a checkpoint on the fastMRI validation split.

Every validation slice is undersampled with a mask fixed per volume (seeded from the file
name) and reconstructed with the full chain. The output is center-cropped to the h5
reconstruction and scored with NMSE, PSNR and SSIM, with data_range = the slice maximum.

Results go to <tag>.json (pooled and high/low-SNR mean and std), <tag>_slices.npz and, with
--volume_metrics, <tag>_volumes.npz. Results of another checkpoint under the same tag are
kept unless --overwrite is given.

    python evaluate.py --checkpoint ckpt.pt
    python evaluate.py --checkpoint ckpt.pt --acceleration 4 --max_slices 1000

The model config comes from the checkpoint unless --config is given.
"""

import argparse
import json
import time
from collections import OrderedDict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from fastmri.data import transforms as T
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

from data.datasets import build_dataset, build_mask_func
from data.transforms import VarNetDataTransform
from evaluation.metrics import nmse, psnr, ssim
from evaluation.snr import SNR_THRESHOLD_DB, kspace_snr_db, split_by_snr
from local_data.constants import PATH_RESULTS
from modeling import build_model
from utils.checkpoint import extract_weights, load_checkpoint
from utils.misc import CENTER_FRACTIONS, load_config, parse_set_overrides, run_name

REPO_ROOT = Path(__file__).resolve().parent
METRICS = ("nmse", "psnr", "ssim")


class SnrTransform:
    """VarNetDataTransform that also returns the slice's k-space SNR proxy (dB)."""

    def __init__(self, transform: VarNetDataTransform, multicoil: bool):
        self.transform = transform
        self.multicoil = multicoil

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        snr_db = kspace_snr_db(kspace, attrs["padding_left"], attrs["padding_right"], self.multicoil)
        return self.transform(kspace, mask, target, attrs, fname, slice_num), snr_db


def _collate_one(batch):
    (sample, snr_db), = batch
    return default_collate([sample]), snr_db


def _num(x) -> str:
    return f"{x:g}"


def resolve_config(args, ckpt):
    overrides = parse_set_overrides(args.set)
    if args.config:
        with open(args.config) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "method" not in raw and "method" not in overrides:
            raise SystemExit(
                f"{args.config} does not name a method. Pass a release config such as "
                "configs/brain/ours.json."
            )
        return load_config(raw, overrides), f"--config {args.config}"
    if isinstance(ckpt.get("config"), dict):
        return load_config(ckpt["config"], overrides), "checkpoint"
    raise SystemExit(
        f"{args.checkpoint} stores no config. Pass --config (for example "
        "configs/brain/ours.json with --set chans=64 --set acceleration=4)."
    )


def make_tag(cfg, eval_cfg, epoch, weights, selection, snr_threshold, epoch_position=None) -> str:
    tag = f"{run_name(cfg)}_ev{_num(eval_cfg['acceleration'])}x"
    if eval_cfg["center_fraction"] != CENTER_FRACTIONS.get(eval_cfg["acceleration"]):
        tag += f"_cf{_num(eval_cfg['center_fraction'])}"
    if isinstance(epoch, (int, float)) and float(epoch).is_integer():
        epoch = int(epoch)
    tag += f"_ep{epoch if epoch is not None else 'NA'}"
    if epoch_position:  # a mid-epoch checkpoint: samples into the epoch
        tag += f"p{epoch_position}"
    tag += f"_{weights}"
    if snr_threshold != SNR_THRESHOLD_DB:
        tag += f"_snr{_num(snr_threshold)}"
    tag += f"_{selection}" if selection else ""
    return tag.replace("/", "_").replace("\\", "_")  # a run_name may hold a path


def select_indices(dataset, max_slices, slices):
    """Dataset indices to evaluate (sorted) and a short tag for the selection."""
    n = len(dataset)
    if slices:
        wanted = set()
        for item in slices:
            fname, sep, s = item.partition(":")
            wanted.add((Path(fname).name, int(s) if sep else None))
        indices, found = [], set()
        for i, sample in enumerate(dataset.raw_samples):
            name = sample.fname.name
            hits = wanted & {(name, None), (name, int(sample.slice_ind))}
            if hits:
                indices.append(i)
                found |= hits
        if wanted - found:
            missing = sorted(wanted - found, key=str)
            raise ValueError(f"--slices entries not in the validation set: {missing}")
        return indices, f"sel{len(indices)}"
    if max_slices and max_slices < n:
        indices = np.random.RandomState(0).choice(n, max_slices, replace=False)
        return sorted(int(i) for i in indices), f"n{max_slices}"
    return list(range(n)), ""


def _stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(values.mean()), "std": float(values.std())}


def summarize(rows: dict, keep=None) -> dict:
    keep = np.ones(len(rows["psnr"]), dtype=bool) if keep is None else keep
    out = {"n": int(keep.sum())}
    for m in METRICS:
        out[m] = _stats(np.asarray(rows[m])[keep])
    return out


class VolumeAccumulator:
    """Stacks cropped target and output slices per file and scores each volume."""

    def __init__(self, slices_per_file: dict):
        self.slices_per_file = slices_per_file
        self.fname, self.buffer = None, {}
        self.rows = {"fname": [], "num_slices": [], **{m: [] for m in METRICS}}

    def add(self, fname, slice_num, gt, pred):
        if fname != self.fname:
            self.flush()
            self.fname = fname
        self.buffer[slice_num] = (gt, pred)

    def flush(self):
        if not self.buffer:
            return
        order = sorted(self.buffer)
        gt = np.stack([self.buffer[s][0] for s in order])
        pred = np.stack([self.buffer[s][1] for s in order])
        self.rows["fname"].append(self.fname)
        self.rows["num_slices"].append(len(order))
        self.rows["nmse"].append(float(nmse(gt, pred)))
        self.rows["psnr"].append(float(psnr(gt, pred)))
        self.rows["ssim"].append(float(ssim(gt, pred)[0]))
        self.buffer = {}

    def summary(self) -> dict:
        self.flush()
        sizes = zip(self.rows["fname"], self.rows["num_slices"])
        partial = sum(n < self.slices_per_file[f] for f, n in sizes)
        out = summarize(self.rows)
        return {"n_volumes": out.pop("n"), "partial_volumes": int(partial), **out}


def _version(package):
    try:
        return version(package)
    except Exception:  # e.g. a source checkout without package metadata
        return None


def _fmt(group: dict) -> str:
    if not group["n"]:
        return "n=0"
    p, s, e = group["psnr"], group["ssim"], group["nmse"]
    return (
        f"n={group['n']} PSNR {p['mean']:.2f} ({p['std']:.2f}) SSIM {s['mean']:.4f} "
        f"({s['std']:.4f}) NMSE {e['mean']:.4f} ({e['std']:.4f})"
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True,
                    help="training or release checkpoint")
    ap.add_argument("--config", default=None,
                    help="release config JSON (default: the one stored in the checkpoint)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, JSON-parsed (repeatable), e.g. --set chans=64")
    ap.add_argument("--acceleration", type=float, default=None,
                    help="evaluation acceleration (default: the training acceleration)")
    ap.add_argument("--center_fraction", type=float, default=None,
                    help="evaluation center fraction (default: the training center fraction at the "
                         "training acceleration, else 0.08 at 4x and 0.04 at 8x)")
    ap.add_argument("--data_path", default=None,
                    help="fastMRI root (default: PATH_DATASETS in local_data/constants.py)")
    ap.add_argument("--weights", choices=("auto", "ema", "live"), default="auto",
                    help="auto = EMA weights when the checkpoint has them, else live")
    ap.add_argument("--max_slices", type=int, default=0,
                    help="evaluate N slices drawn with RandomState(0) (0 = all)")
    ap.add_argument("--snr_threshold", type=float, default=SNR_THRESHOLD_DB,
                    help="high SNR = snr_db > threshold (dB)")
    ap.add_argument("--volume_metrics", action="store_true",
                    help="also report volume-wise metrics")
    ap.add_argument("--output_dir", default=None, help="default: <PATH_RESULTS>/eval")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace results that another checkpoint wrote under the same tag")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="data loader workers (default: config num_workers, else 4)")
    ap.add_argument("--slices", nargs="+", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.acceleration is not None and args.acceleration.is_integer():
        args.acceleration = int(args.acceleration)
    if args.max_slices < 0:
        ap.error("--max_slices must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)

    try:
        ckpt = load_checkpoint(args.checkpoint, mmap=True)
    except RuntimeError:  # legacy (non-zip) checkpoints cannot be memory-mapped
        ckpt = load_checkpoint(args.checkpoint)
    try:
        cfg, cfg_source = resolve_config(args, ckpt)
        eval_overrides = {"acceleration": args.acceleration or cfg["acceleration"]}
        if args.center_fraction is not None:
            eval_overrides["center_fraction"] = args.center_fraction
        eval_cfg = load_config(cfg, eval_overrides)
        state_dict, weights = extract_weights(ckpt, args.weights)
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise SystemExit(f"error: {e.args[0] if isinstance(e, KeyError) and e.args else e}")
    epoch = ckpt.get("epoch")
    epoch_position = next(
        (ckpt[k] for k in ("samples_consumed_in_epoch", "epoch_position", "idx") if k in ckpt), None
    )
    model = build_model(cfg)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    del ckpt, state_dict

    transform = SnrTransform(
        VarNetDataTransform(build_mask_func(eval_cfg), use_seed=True),
        multicoil=not cfg["singlecoil"],
    )
    dataset = build_dataset(cfg, "val", transform, data_path=args.data_path)
    indices, selection = select_indices(dataset, args.max_slices, args.slices)
    tag = make_tag(cfg, eval_cfg, epoch, weights, selection, args.snr_threshold, epoch_position)
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / PATH_RESULTS / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}.json"
    checkpoint_path = str(Path(args.checkpoint).resolve())
    replacing = json_path.exists()
    if replacing and not args.overwrite:
        try:
            prev = json.loads(json_path.read_text())
        except ValueError:  # not a summary written by evaluate.py
            prev = {}
        if (prev.get("checkpoint"), prev.get("epoch_position")) != (checkpoint_path, epoch_position):
            raise SystemExit(
                f"{json_path} holds the results of {prev.get('checkpoint')} (epoch {prev.get('epoch')}, "
                f"position {prev.get('epoch_position')}). Pass another --output_dir, "
                "--set run_name=NAME or --overwrite."
            )

    print(
        f"{tag}\n  checkpoint {args.checkpoint} (epoch {epoch}, position {epoch_position}, "
        f"{weights} weights, config from {cfg_source})\n"
        f"  {cfg['anatomy']} {cfg['challenge']} val: {len(indices)} of {len(dataset)} slices, "
        f"eval mask {_num(eval_cfg['acceleration'])}x cf {_num(eval_cfg['center_fraction'])}, "
        f"device {device}",
        flush=True,
    )
    if replacing:
        print(f"  overwriting {json_path}", flush=True)

    slices_per_file = {}
    for sample in dataset.raw_samples:
        slices_per_file[sample.fname.name] = slices_per_file.get(sample.fname.name, 0) + 1
    volumes = VolumeAccumulator(slices_per_file) if args.volume_metrics else None
    rows = {"fname": [], "slice": [], "snr_db": [], **{m: [] for m in METRICS}}

    num_workers = cfg.get("num_workers", 4) if args.num_workers is None else args.num_workers
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=_collate_one,
    )
    log_every = max(1, len(indices) // 20)
    t0 = time.time()
    with torch.no_grad():
        for k, (batch, snr_db) in enumerate(loader, 1):
            fname, slice_num = batch.fname[0], int(batch.slice_num[0])
            if batch.target.ndim != 3:
                raise RuntimeError(f"{fname} slice {slice_num} has no target reconstruction")
            try:
                output = model(
                    batch.masked_kspace.to(device),
                    batch.mask.to(device),
                    batch.num_low_frequencies.to(device),
                )
                target, output = T.center_crop_to_smallest(batch.target.to(device), output)
                gt, pred = target.cpu().numpy(), output.cpu().numpy()
                values = (float(nmse(gt, pred)), float(psnr(gt, pred)), float(ssim(gt, pred)[0]))
            except Exception as e:
                raise RuntimeError(f"evaluation failed on {fname} slice {slice_num}") from e

            rows["fname"].append(fname)
            rows["slice"].append(slice_num)
            rows["snr_db"].append(snr_db)
            for m, v in zip(METRICS, values):
                rows[m].append(v)
            if volumes is not None:
                volumes.add(fname, slice_num, gt[0], pred[0])
            if k % log_every == 0 or k == len(indices):
                dt = time.time() - t0
                print(
                    f"  [{k}/{len(indices)}] PSNR {np.mean(rows['psnr']):.2f} "
                    f"SSIM {np.mean(rows['ssim']):.4f} ({dt / k:.2f} s/slice, "
                    f"{dt * (len(indices) - k) / k / 60:.1f} min left)",
                    flush=True,
                )

    if len(rows["psnr"]) != len(indices):
        raise RuntimeError(f"evaluated {len(rows['psnr'])} of {len(indices)} slices")
    nonfinite = sum(int((~np.isfinite(rows[m])).sum()) for m in METRICS)
    if nonfinite:
        print(f"WARNING: {nonfinite} non-finite metric values, their means are nan", flush=True)
    slices_path = out_dir / f"{tag}_slices.npz"
    np.savez(
        slices_path,
        fname=np.array(rows["fname"]),
        slice=np.array(rows["slice"], dtype=np.int64),
        snr_db=np.array(rows["snr_db"], dtype=np.float64),
        **{m: np.array(rows[m], dtype=np.float64) for m in METRICS},
    )

    high, low = split_by_snr(rows["snr_db"], args.snr_threshold)
    summary = OrderedDict(
        tag=tag,
        checkpoint=checkpoint_path,
        config_source=cfg_source,
        method=cfg["method"],
        anatomy=cfg["anatomy"],
        challenge=cfg["challenge"],
        chans=cfg["chans"],
        num_cascades=cfg["num_cascades"],
        train_acceleration=cfg["acceleration"],
        eval_acceleration=eval_cfg["acceleration"],
        eval_center_fraction=eval_cfg["center_fraction"],
        epoch=epoch,
        epoch_position=epoch_position,
        weights=weights,
        num_slices=len(indices),
        num_val_slices=len(dataset),
        selection=selection or "all",
        snr_threshold_db=args.snr_threshold,
        pooled=summarize(rows),
        snr_high=summarize(rows, high),
        snr_low=summarize(rows, low),
    )
    if volumes is not None:
        summary["volumes"] = volumes.summary()
        partial = summary["volumes"]["partial_volumes"]
        if partial:
            print(f"WARNING: {partial} volumes were only partly evaluated", flush=True)
    summary["seconds"] = round(time.time() - t0, 1)
    summary["versions"] = {"torch": torch.__version__, "fastmri": _version("fastmri")}
    summary["config"] = cfg

    if volumes is not None:
        vol = volumes.rows
        np.savez(
            out_dir / f"{tag}_volumes.npz",
            fname=np.array(vol["fname"]),
            num_slices=np.array(vol["num_slices"], dtype=np.int64),
            **{m: np.array(vol[m], dtype=np.float64) for m in METRICS},
        )
    else:
        # drop a stale file from an earlier --volume_metrics run
        (out_dir / f"{tag}_volumes.npz").unlink(missing_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))

    thr = _num(args.snr_threshold)
    print(f"\n{tag}")
    print(f"  pooled    {_fmt(summary['pooled'])}")
    print(f"  >{thr} dB   {_fmt(summary['snr_high'])}")
    print(f"  <={thr} dB  {_fmt(summary['snr_low'])}")
    if volumes is not None:
        v = summary["volumes"]
        print(f"  volumes   {_fmt({'n': v['n_volumes'], **v})}")
    extra = f" and {tag}_volumes.npz" if volumes is not None else ""
    print(f"wrote {json_path}, {slices_path.name}{extra}")
    return summary


if __name__ == "__main__":
    main()
