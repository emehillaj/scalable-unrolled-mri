#!/usr/bin/env python3
"""Train ours, e2e, GLEAM or FLAT on fastMRI (one GPU, or several via torchrun / srun).

    python train.py --config configs/knee/ours.json
    python train.py --config configs/brain/ours.json --chans 64 --acceleration 4
    torchrun --standalone --nproc_per_node=4 train.py --config configs/brain/ours.json
    python train.py --config configs/brain/ours.json --resume      # continue the same run

The run directory (--output_dir, default
local_data/results/<run_name>) holds config.json, log.jsonl and checkpoint.pt.
--set KEY=VALUE overrides any config key (JSON value); the dedicated flags take precedence.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from fastmri.data import transforms as T
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate

try:
    import fcntl
except ImportError:  # not available on Windows
    fcntl = None

from data.datasets import build_dataset, build_mask_func
from data.samplers import ResumableSampler
from data.transforms import VarNetDataTransform
from evaluation.metrics import nmse, psnr, ssim
from local_data.constants import PATH_RESULTS
from modeling import build_model
from training.checkpoint import CHECKPOINT_NAME, restore_training_state, save_training_state
from training.distributed import (
    all_reduce_sum_, barrier, broadcast_flag, broadcast_params, cleanup, init_distributed, print0,
)
from training.ema import ModelEMA
from training.losses import build_loss
from training.methods import (
    STEP_FNS, GradWindow, check_reduction_coverage, clip_and_step, fill_training_defaults,
    mix_seed, reduce_gradients, sample_cascade,
)
from utils.checkpoint import load_checkpoint
from utils.misc import JsonlLogger, load_config, parse_set_overrides, run_name, set_seeds

REPO_ROOT = Path(__file__).resolve().parent
RESUME_KEYS = {"config", "state_dict", "optimizer", "epoch", "samples_consumed_in_epoch", "global_step"}
RESUME_LOCKED_KEYS = {"method", "anatomy", "challenge", "acceleration", "center_fraction", "num_cascades",
                      "chans", "seed", "train_file_list"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON config. Optional with --resume, which then uses the checkpoint's "
                                     "config; a --config given with --resume replaces that config")
    p.add_argument("--acceleration", type=int, help="undersampling factor (center fraction derived)")
    p.add_argument("--center_fraction", type=float, help="fully sampled center fraction")
    p.add_argument("--chans", type=int, help="network width C (channels of the first U-Net level)")
    p.add_argument("--num_cascades", type=int, help="number of cascades T")
    p.add_argument("--data_path", help="fastMRI root holding knee/ and brain/ (default: local_data/constants.py)")
    p.add_argument("--output_dir", help="run directory (default: local_data/results/<run_name>)")
    p.add_argument("--resume", nargs="?", const="auto", default=None, metavar="CHECKPOINT",
                   help="continue from <output_dir>/checkpoint.pt, or from the given file or directory")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config key (repeatable; JSON-parsed value)")
    p.add_argument("--max_steps", type=int, default=None,
                   help="stop (and save) once this many optimizer steps are done in total; for debugging")
    p.add_argument("--allow_config_change", action="store_true",
                   help="with --resume: accept a different method, data, mask, model size or seed")
    args = p.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max_steps must be >= 1")
    return args


def resolve_run(args):
    """Return config, output directory, loaded checkpoint (if resuming) and config changes."""
    overrides = parse_set_overrides(args.set)
    for key in ("acceleration", "center_fraction", "chans", "num_cascades"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    cfg = load_config(args.config, overrides) if args.config else None

    ckpt_path = None
    if args.resume == "auto":
        if args.output_dir:
            ckpt_path = Path(args.output_dir) / CHECKPOINT_NAME
        elif cfg is not None:
            ckpt_path = REPO_ROOT / PATH_RESULTS / run_name(cfg) / CHECKPOINT_NAME
        else:
            raise SystemExit("--resume needs --output_dir or --config to find the checkpoint")
    elif args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / CHECKPOINT_NAME
    elif cfg is None:
        raise SystemExit("--config is required unless --resume is given")

    ckpt = None
    if ckpt_path is not None:
        if not ckpt_path.is_file():
            raise SystemExit(f"no checkpoint to resume from at {ckpt_path}: start without --resume, "
                             "or pass --resume <run dir or checkpoint>")
        # memory-mapped so that the ranks do not each read the whole file into host memory
        try:
            ckpt = load_checkpoint(ckpt_path, mmap=True)
        except RuntimeError:  # legacy (non-zip) checkpoints cannot be memory-mapped
            ckpt = load_checkpoint(ckpt_path)
        missing = sorted(RESUME_KEYS - set(ckpt))
        if missing:
            raise SystemExit(f"{ckpt_path} is not a train.py checkpoint (it has no {missing}). "
                             "--resume needs the checkpoint.pt of a train.py run.")
        if cfg is None:
            cfg = load_config(ckpt["config"], overrides)
    fill_training_defaults(cfg)

    changes = {}
    if ckpt is not None:
        old = ckpt["config"]
        changes = {k: [old.get(k), cfg.get(k)] for k in sorted(set(old) | set(cfg))
                   if old.get(k) != cfg.get(k)}
        if changes:
            print0("config changes on resume: "
                   + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in changes.items()))
        locked = sorted(RESUME_LOCKED_KEYS & changes.keys())
        if locked and not args.allow_config_change:
            raise SystemExit(f"--resume would change {locked}. Repeat the run's original flags, "
                             "or pass --allow_config_change to accept the change.")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif ckpt_path is not None:
        output_dir = ckpt_path.parent
    else:
        output_dir = REPO_ROOT / PATH_RESULTS / run_name(cfg)
    if ckpt is None and (output_dir / CHECKPOINT_NAME).exists():
        raise SystemExit(f"{output_dir / CHECKPOINT_NAME} exists: pass --resume to continue it, "
                         "or choose another --output_dir")
    return cfg, output_dir, ckpt, changes


_RUN_LOCK = None


def _lock_run_dir(output_dir: Path) -> bool:
    """Lock <output_dir>/.train.lock until exit. False if another process holds it."""
    global _RUN_LOCK
    output_dir.mkdir(parents=True, exist_ok=True)
    _RUN_LOCK = open(output_dir / ".train.lock", "w")
    if fcntl is None:
        return True
    try:
        fcntl.flock(_RUN_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    except OSError as e:
        warnings.warn(f"could not lock {output_dir}: {e}")
    return True


def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    return cls(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
               eps=cfg["adam_eps"])


class _SeededSampler(Sampler):
    """Yields (dataset index, mask seed), the seed derived from (seed, epoch, sample position)."""

    def __init__(self, sampler: ResumableSampler, seed: int):
        self.sampler = sampler
        self.seed = seed

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        s = self.sampler
        for k, idx in enumerate(s):
            position = s.start + s.rank + k * s.world_size
            yield idx, mix_seed(self.seed, s.epoch, position, 1) % 2**32


class _SeededMaskDataset(Dataset):
    """Reseeds the training mask RNG with the seed that comes with each index."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        idx, seed = item
        self.dataset.transform.mask_func.rng.seed(seed)
        return self.dataset[idx]


def to_device(batch, device: torch.device, full_kspace: bool = True):
    nb = device.type == "cuda"
    move = lambda t: t.to(device, non_blocking=nb)  # noqa: E731
    return batch._replace(
        masked_kspace=move(batch.masked_kspace),
        mask=move(batch.mask),
        num_low_frequencies=move(batch.num_low_frequencies),
        target=move(batch.target),
        max_value=move(batch.max_value),
        full_kspace=move(batch.full_kspace) if full_kspace else batch.full_kspace,
    )


@torch.no_grad()
def quick_validate(model, dataset, indices, device) -> dict:
    """PSNR / SSIM / NMSE on a few fixed validation slices; a sanity check, not for reporting."""
    was_training = model.training
    model.eval()
    scores = {"psnr": [], "ssim": [], "nmse": []}
    for i in indices:
        batch = to_device(default_collate([dataset[int(i)]]), device, full_kspace=False)
        output = model(batch.masked_kspace, batch.mask, batch.num_low_frequencies)
        target, output = T.center_crop_to_smallest(batch.target, output)
        gt, pred = target.cpu().numpy(), output.cpu().numpy()
        maxval = float(gt.max())
        scores["psnr"].append(float(psnr(gt, pred, maxval)))
        scores["ssim"].append(float(ssim(gt, pred, maxval)[0]))
        scores["nmse"].append(float(nmse(gt, pred)))
    model.train(was_training)
    return {f"val_{k}": float(np.mean(v)) for k, v in scores.items()}


def main(argv=None):
    args = parse_args(argv)
    info = init_distributed()
    try:
        cfg, output_dir, ckpt, config_changes = resolve_run(args)
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise SystemExit(f"error: {e.args[0] if isinstance(e, KeyError) and e.args else e}")
    device, world = info.device, info.world_size
    method, num_cascades = cfg["method"], cfg["num_cascades"]
    if not broadcast_flag(_lock_run_dir(output_dir) if info.is_main else True, device):
        raise SystemExit(f"another train.py is running in {output_dir}")
    if info.is_main:
        # leftovers of a save that was killed
        for stale in output_dir.glob(f".{CHECKPOINT_NAME}*.tmp"):
            stale.unlink(missing_ok=True)

    eff = cfg["effective_batch_size"]
    if eff % world:
        raise SystemExit(f"effective_batch_size {eff} is not divisible by world size {world}")
    accum = eff // world

    # the EMA lives on rank 0 only
    set_seeds(cfg["seed"])
    model = build_model(cfg).to(device)
    check_reduction_coverage(model)
    optimizer = build_optimizer(cfg, model)
    ema = ModelEMA(model, cfg["ema_decay"]) if cfg["use_ema"] and info.is_main else None
    start_epoch, start_samples, global_step = 0, 0, 0
    saved_slices = None
    if ckpt is not None:
        start_epoch, start_samples, global_step = restore_training_state(ckpt, cfg, model, optimizer, ema)
        saved_slices = ckpt.get("train_slices")
        del ckpt
    broadcast_params(model)
    loss_fn = build_loss(cfg).to(device) if method != "flat" else None
    step_fn = STEP_FNS[method]
    need_full_kspace = method == "flat" or not cfg["crop_loss"]

    train_transform = VarNetDataTransform(build_mask_func(cfg), use_seed=False)
    train_set = build_dataset(cfg, "train", train_transform, data_path=args.data_path)
    if saved_slices is not None and saved_slices != len(train_set):
        msg = (f"the training set has {len(train_set)} slices, but the checkpoint was written with "
               f"{saved_slices} (the data directory or train_file_list changed)")
        if start_samples:
            raise SystemExit(f"{msg}. The saved position in the epoch does not apply to it.")
        print0(f"WARNING: {msg}")
    sampler = ResumableSampler(len(train_set), info.rank, world, seed=cfg["seed"], shuffle=True)
    num_workers = cfg["num_workers"]
    loader = DataLoader(
        _SeededMaskDataset(train_set),
        batch_size=1,
        sampler=_SeededSampler(sampler, cfg["seed"]),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_set, val_indices = None, []
    if info.is_main and cfg["val_slices"] > 0:
        val_transform = VarNetDataTransform(build_mask_func(cfg), use_seed=True)
        val_set = build_dataset(cfg, "val", val_transform, data_path=args.data_path)
        val_indices = np.unique(np.linspace(0, len(val_set) - 1, cfg["val_slices"]).round().astype(int))

    logger = JsonlLogger(output_dir / "log.jsonl" if info.is_main else None)
    if info.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
    n_params = sum(p.numel() for p in model.parameters())
    print0(f"{method} {cfg['anatomy']} {cfg['challenge']} {cfg['acceleration']}x: T={num_cascades} "
           f"chans={cfg['chans']} params={n_params / 1e6:.2f}M device={device} world={world} "
           f"accum={accum} train_slices={len(train_set)} output={output_dir}")
    if start_epoch or start_samples or global_step:
        print0(f"resuming at epoch {start_epoch}, sample {start_samples}, step {global_step}")
    logger.log(event="start", method=method, params=n_params, world_size=world, accum=accum,
               epoch=start_epoch, samples_in_epoch=start_samples, step=global_step,
               train_slices=len(train_set), config_changes=config_changes)

    def save(epoch, samples_in_epoch):
        """Save and validate on rank 0, then barrier. Returns the seconds spent."""
        t0 = time.time()
        if info.is_main:
            save_training_state(
                output_dir / CHECKPOINT_NAME, cfg=cfg, model=model, optimizer=optimizer, ema=ema,
                epoch=epoch, samples_consumed_in_epoch=samples_in_epoch, global_step=global_step,
                world_size=world, train_slices=len(train_set),
            )
            logger.log(event="checkpoint", step=global_step, epoch=epoch, samples_in_epoch=samples_in_epoch)
            if val_set is not None:
                scores = quick_validate(ema.module if ema is not None else model, val_set, val_indices, device)
                logger.log(event="val", step=global_step, epoch=epoch, weights="ema" if ema else "live", **scores)
                print0(f"  val step {global_step}: psnr {scores['val_psnr']:.3f} ssim {scores['val_ssim']:.4f}")
        barrier()
        return time.time() - t0

    window = GradWindow(num_cascades)
    log_every = max(1, int(cfg["log_every"]))
    last_save = time.time()
    log_time, log_steps, log_clipped, log_skipped = time.time(), 0, 0, 0
    # (sum, count) of micro-step losses
    log_loss = torch.zeros(2, dtype=torch.float64, device=device)
    epoch_loss = torch.zeros(2, dtype=torch.float64, device=device)
    stop = cfg["max_epochs"] <= start_epoch or (args.max_steps is not None and global_step >= args.max_steps)
    model.train()

    for epoch in range(start_epoch, cfg["max_epochs"]):
        if stop:
            break
        sampler.set_epoch(epoch)
        if epoch == start_epoch:
            sampler.set_start(start_samples)
        consumed = sampler.start
        n_micro = len(sampler)
        window.reset()
        in_window, cascade_idx, epoch_done = 0, None, True
        epoch_loss.zero_()

        for m, batch in enumerate(loader):
            epoch_done = m + 1 == n_micro
            if method == "ours":
                if not cfg["coordinate_cascade_sampling"]:
                    cascade_idx = sample_cascade(num_cascades, cfg["seed"], epoch, consumed + info.rank)
                elif in_window == 0:
                    cascade_idx = sample_cascade(num_cascades, cfg["seed"], epoch, consumed // eff)
            batch = to_device(batch, device, full_kspace=need_full_kspace)
            loss = step_fn(model, batch, loss_fn, window, cfg, cascade_idx)
            in_window += 1
            consumed += world
            log_loss[0] += loss
            log_loss[1] += 1
            epoch_loss[0] += loss
            epoch_loss[1] += 1
            if in_window < accum and not epoch_done:
                continue

            reduce_gradients(model, window, device)
            stats = clip_and_step(optimizer, cfg["grad_clip_norm"])
            window.reset()
            in_window = 0
            global_step += 1
            if stats["skipped"]:
                print0(f"[nan-guard] non-finite gradient at step {global_step}; step skipped")
            elif ema is not None:
                ema.update(model)
            log_steps += 1
            log_clipped += stats["clipped"]
            log_skipped += stats["skipped"]

            last = args.max_steps is not None and global_step >= args.max_steps
            if global_step % log_every == 0 or last:
                all_reduce_sum_(log_loss)
                now = time.time()
                record = dict(
                    event="train", step=global_step, epoch=epoch, samples_in_epoch=consumed,
                    loss=float(log_loss[0] / log_loss[1]), grad_norm=stats["grad_norm"],
                    clipped=bool(stats["clipped"]), clip_rate=log_clipped / log_steps,
                    skipped_steps=log_skipped, lr=optimizer.param_groups[0]["lr"],
                    sec_per_step=(now - log_time) / log_steps,
                )
                if device.type == "cuda":
                    record["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
                logger.log(record)
                print0(f"epoch {epoch} step {global_step} [{consumed}/{sampler.num_samples}] "
                       f"loss {record['loss']:.4g} grad_norm {record['grad_norm']:.3g} "
                       f"{record['sec_per_step']:.2f} s/step")
                log_time, log_steps, log_clipped, log_skipped = now, 0, 0, 0
                log_loss.zero_()

            if last:
                stop = True
            due = broadcast_flag(info.is_main and time.time() - last_save >= 60 * cfg["save_every_minutes"], device)
            if (due or stop) and not epoch_done:
                log_time += save(epoch, consumed)
                last_save = time.time()
            if stop:
                break

        if not epoch_done:
            break
        log_time += save(epoch + 1, 0)
        last_save = time.time()
        all_reduce_sum_(epoch_loss)
        epoch_mean = float(epoch_loss[0] / epoch_loss[1].clamp(min=1))
        logger.log(event="epoch_end", epoch=epoch, step=global_step, loss=epoch_mean,
                   slices=int(epoch_loss[1].item()), partial=epoch == start_epoch and start_samples > 0)
        print0(f"epoch {epoch} done: step {global_step}, mean loss {epoch_mean:.4g}")

    logger.log(event="done", step=global_step)
    logger.close()
    cleanup()


if __name__ == "__main__":
    main()
