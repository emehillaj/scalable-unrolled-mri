"""Training checkpoints: everything needed to continue a run mid-epoch.

<output_dir>/checkpoint.pt holds config, epoch, samples_consumed_in_epoch, global_step,
world_size, train_slices, optimizer_name, state_dict (live weights), optimizer and, with
use_ema, model_ema_state_dict.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from training.ema import ModelEMA
from utils.checkpoint import EMA_KEY, LIVE_KEY, atomic_save

CHECKPOINT_NAME = "checkpoint.pt"


def save_training_state(
    path,
    *,
    cfg: dict,
    model: nn.Module,
    optimizer: Optimizer,
    ema: Optional[ModelEMA],
    epoch: int,
    samples_consumed_in_epoch: int,
    global_step: int,
    world_size: int,
    train_slices: Optional[int] = None,
) -> Path:
    state = {
        "config": cfg,
        "epoch": epoch,
        "samples_consumed_in_epoch": samples_consumed_in_epoch,
        "global_step": global_step,
        "world_size": world_size,
        "optimizer_name": cfg["optimizer"],
        LIVE_KEY: model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if train_slices is not None:
        state["train_slices"] = train_slices
    if ema is not None:
        state[EMA_KEY] = ema.state_dict()
    return atomic_save(state, path)


def restore_training_state(
    ckpt: dict,
    cfg: dict,
    model: nn.Module,
    optimizer: Optimizer,
    ema: Optional[ModelEMA],
) -> Tuple[int, int, int]:
    """Load weights, optimizer and EMA; lr, weight decay and eps are re-applied from cfg.

    Returns (epoch, samples_consumed_in_epoch, global_step).
    """
    saved_optimizer = ckpt.get("optimizer_name", cfg["optimizer"])
    if saved_optimizer != cfg["optimizer"]:
        raise ValueError(
            f"checkpoint was trained with optimizer {saved_optimizer!r}, config says "
            f"{cfg['optimizer']!r}"
        )
    model.load_state_dict(ckpt[LIVE_KEY])
    optimizer.load_state_dict(ckpt["optimizer"])
    # copy CPU tensors so nothing keeps the memory-mapped checkpoint file open
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and value.device.type == "cpu":
                state[key] = value.clone()
    for group in optimizer.param_groups:
        group["lr"] = cfg["learning_rate"]
        group["weight_decay"] = cfg["weight_decay"]
        group["eps"] = cfg["adam_eps"]
    if ema is not None:
        if ckpt.get(EMA_KEY):
            ema.load_state_dict(ckpt[EMA_KEY])
        else:
            warnings.warn("checkpoint has no EMA weights; starting the EMA from the live weights")
            ema.copy_from(model)
    return int(ckpt["epoch"]), int(ckpt["samples_consumed_in_epoch"]), int(ckpt["global_step"])
