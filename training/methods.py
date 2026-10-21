"""Per-method training step on one slice, gradient reduction and the optimizer step.

Step functions run the forward pass, backpropagate the raw loss and record in a GradWindow
which parameters the backward reached. reduce_gradients then sums gradients over ranks and
divides each group (cascade i, shared U-Net, sensitivity net) by its number of backward
passes, so every update is a mean over its contributions.

  ours   one cascade per step (shared by all slices and ranks), one backward per slice
  e2e    every cascade, one backward per slice
  gleam  one backward per cascade j = 0..T-1 per slice, hard data consistency
  flat   like gleam without hard DC, with the k-space velocity loss
"""

import math
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from training.distributed import is_distributed
from training.losses import flat_velocity_loss, image_loss

# defaults for keys a config leaves out
METHOD_DEFAULTS = {
    "ours": dict(loss="mse", mse_scale=1e10, crop_loss=False, optimizer="adamw", adam_eps=1e-17,
                 use_ema=True),
    "e2e": dict(loss="mse", mse_scale=1e10, crop_loss=False, optimizer="adamw", adam_eps=1e-17,
                use_ema=False),
    "gleam": dict(loss="mse", mse_scale=1e10, crop_loss=False, optimizer="adamw", adam_eps=1e-17,
                  use_ema=False),
    "flat": dict(loss="velocity", optimizer="adamw", adam_eps=1e-17, use_ema=False),
}
COMMON_DEFAULTS = dict(
    learning_rate=2e-5,
    weight_decay=0.0,
    effective_batch_size=8,
    grad_clip_norm=None,
    coordinate_cascade_sampling=True,
    ema_decay=0.997,
    max_epochs=200,
    save_every_minutes=30,
    val_slices=8,
    num_workers=4,
    seed=42,
    log_every=50,
)


def fill_training_defaults(cfg: dict) -> dict:
    """Add defaults for missing training keys (in place) and validate them."""
    method = cfg["method"]
    for key, value in {**COMMON_DEFAULTS, **METHOD_DEFAULTS[method]}.items():
        cfg.setdefault(key, value)
    if method == "flat":
        if cfg["loss"] != "velocity":
            raise ValueError("flat trains with the velocity loss; set loss to 'velocity'")
    elif cfg["loss"] not in ("mse", "ssim"):
        raise ValueError(f"loss must be 'mse' or 'ssim' for {method}, got {cfg['loss']!r}")
    if cfg["optimizer"] not in ("adamw", "adam"):
        raise ValueError(f"optimizer must be 'adamw' or 'adam', got {cfg['optimizer']!r}")
    eff = cfg["effective_batch_size"]
    if isinstance(eff, bool) or not isinstance(eff, int) or eff < 1:
        raise ValueError(f"effective_batch_size must be a positive integer, got {eff!r}")
    if cfg["grad_clip_norm"] is not None and not cfg["grad_clip_norm"] > 0:
        raise ValueError(f"grad_clip_norm must be positive or null, got {cfg['grad_clip_norm']!r}")
    for key, low in (("val_slices", 0), ("num_workers", 0), ("max_epochs", 0), ("log_every", 1),
                     ("seed", 0)):
        v = cfg[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < low:
            raise ValueError(f"{key} must be an integer >= {low}, got {v!r}")
    v = cfg["save_every_minutes"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v >= 0:
        raise ValueError(f"save_every_minutes must be a number >= 0, got {v!r}")
    return cfg


def mix_seed(*values: int) -> int:
    """A 64-bit seed derived from non-negative integers (numpy SeedSequence)."""
    lo, hi = np.random.SeedSequence([int(v) for v in values]).generate_state(2, dtype=np.uint32)
    return int(lo) | (int(hi) << 32)


def sample_cascade(num_cascades: int, seed: int, epoch: int, position: int) -> int:
    """Uniform cascade index seeded with (seed, epoch, position), independent of the world size."""
    g = torch.Generator()
    g.manual_seed(mix_seed(seed, epoch, position, 0))
    return int(torch.randint(0, num_cascades, (1,), generator=g).item())


class GradWindow:
    """Backward-pass counts since the last optimizer step."""

    def __init__(self, num_cascades: int):
        self.num_cascades = num_cascades
        self.reset()

    def reset(self) -> None:
        self.cascade_counts = [0] * self.num_cascades
        self.sens_count = 0

    def add(self, cascades: Iterable[int]) -> None:
        """Record one backward pass through ``cascades`` (and the sens-net)."""
        for i in cascades:
            self.cascade_counts[i] += 1
        self.sens_count += 1


def _forward_args(batch):
    return batch.masked_kspace, batch.mask, batch.num_low_frequencies


def ours_step(model, batch, loss_fn, window: GradWindow, cfg: dict, cascade_idx: int) -> float:
    """Ours: full chain, gradients only for cascade ``cascade_idx`` (activation-light suffix)."""
    output = model(*_forward_args(batch), cascade_to_train_idx=cascade_idx)
    loss = image_loss(loss_fn, output, batch, cfg["crop_loss"])
    loss.backward()
    window.add([cascade_idx])
    return loss.item()


def e2e_step(model, batch, loss_fn, window: GradWindow, cfg: dict, cascade_idx=None) -> float:
    """E2E: full chain with standard backprop through every cascade."""
    output = model(*_forward_args(batch), cascade_to_train_idx=None)
    loss = image_loss(loss_fn, output, batch, cfg["crop_loss"])
    loss.backward()
    window.add(range(window.num_cascades))
    return loss.item()


def gleam_step(model, batch, loss_fn, window: GradWindow, cfg: dict, cascade_idx=None) -> float:
    """GLEAM: one backward per cascade j (earlier cascades frozen). Returns the mean loss."""
    total = 0.0
    for j in range(window.num_cascades):
        output = model(*_forward_args(batch), cascade_to_train_idx=j, apply_dc=True)
        loss = image_loss(loss_fn, output, batch, cfg["crop_loss"])
        loss.backward()
        window.add([j])
        total = total + loss.detach()
    return float(total) / window.num_cascades


def flat_step(model, batch, loss_fn, window: GradWindow, cfg: dict, cascade_idx=None) -> float:
    """FLAT: one backward per cascade j with the velocity loss. Returns the mean loss."""
    num_blocks = window.num_cascades
    total = 0.0
    for j in range(num_blocks):
        _, kspace_after, kspace_before = model(
            *_forward_args(batch), cascade_to_train_idx=j, return_velocity=True
        )
        loss = flat_velocity_loss(
            kspace_after, kspace_before, batch.masked_kspace, batch.full_kspace,
            j, num_blocks, cfg["singlecoil"],
        )
        loss.backward()
        window.add([j])
        total = total + loss.detach()
    return float(total) / num_blocks


STEP_FNS = {"ours": ours_step, "e2e": e2e_step, "gleam": gleam_step, "flat": flat_step}


def _groups(model: nn.Module):
    """(name, module) for the sens-net, the shared U-Net and each cascade."""
    groups = []
    for name in ("sens_net", "shared_model"):
        module = getattr(model, name, None)
        if module is not None:
            groups.append((name, module))
    groups += [(f"cascades.{i}", c) for i, c in enumerate(model.cascades)]
    return groups


def check_reduction_coverage(model: nn.Module) -> None:
    """Raise if a parameter lies outside sens_net / shared_model / cascades (it would never be reduced)."""
    covered = {id(p) for _, m in _groups(model) for p in m.parameters()}
    missing = [n for n, p in model.named_parameters() if id(p) not in covered]
    if missing:
        raise RuntimeError(f"parameters not covered by the gradient reduction: {missing[:5]}")


def reduce_gradients(model: nn.Module, window: GradWindow, device: torch.device) -> List[int]:
    """Sum gradients and counts over ranks, then divide each group by its global count.

    Missing gradients are zero-filled under torch.distributed so all ranks reduce the same
    parameters. Returns the global per-cascade counts.
    """
    distributed = is_distributed()
    counts = torch.tensor(window.cascade_counts + [window.sens_count], dtype=torch.long, device=device)
    if distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    counts = counts.tolist()
    cascade_counts, sens_count = counts[:-1], counts[-1]

    plan = []
    if getattr(model, "sens_net", None) is not None and sens_count > 0:
        plan.append((model.sens_net, sens_count))
    total = sum(cascade_counts)
    if getattr(model, "shared_model", None) is not None and total > 0:
        plan.append((model.shared_model, total))
    plan += [(model.cascades[i], c) for i, c in enumerate(cascade_counts) if c > 0]

    for module, divisor in plan:
        for p in module.parameters():
            if distributed:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            if p.grad is not None:
                p.grad /= divisor
    return cascade_counts


def _total_norm(grads) -> torch.Tensor:
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2) for g in grads]), 2)


def clip_and_step(optimizer: torch.optim.Optimizer, grad_clip_norm: Optional[float]) -> dict:
    """Clip the global gradient norm (if set), step and zero the gradients; skip non-finite steps.

    Returns {"grad_norm": pre-clip norm, "clipped": bool, "skipped": bool}.
    """
    params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
    if not params:
        return {"grad_norm": 0.0, "clipped": False, "skipped": True}
    if grad_clip_norm is not None:
        norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip_norm))
    else:
        norm = float(_total_norm([p.grad for p in params]))
    finite = math.isfinite(norm)
    if finite:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "grad_norm": norm,
        "clipped": grad_clip_norm is not None and finite and norm > grad_clip_norm,
        "skipped": not finite,
    }
