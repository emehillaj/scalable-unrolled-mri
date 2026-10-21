"""Training losses: scaled MSE on RSS images, the full-FOV RSS target, and the FLAT
velocity loss in k-space."""

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms as T
from fastmri.losses import SSIMLoss


class ScaledMSE(nn.Module):
    """``scale * mean((pred - target) ** 2)``.

    fastMRI magnitudes are tiny (brain images peak around 4e-4), so the plain MSE is about
    1e-9 and its gradients sit near Adam's eps. The paper scales it by 1e10. Accepts and
    ignores ``data_range`` so it can be called like fastMRI's SSIMLoss.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor, data_range=None) -> torch.Tensor:
        return self.scale * F.mse_loss(pred, target)


def build_loss(cfg: dict) -> nn.Module:
    """Image loss for ours / e2e / gleam: "mse" (ScaledMSE with cfg["mse_scale"]) or "ssim".

    cfg must have been completed by training.methods.fill_training_defaults.
    """
    name = cfg["loss"]
    if name == "mse":
        return ScaledMSE(cfg["mse_scale"])
    if name == "ssim":
        return SSIMLoss()
    raise ValueError(f"unknown loss {name!r}; choose 'mse' or 'ssim'")


def rss_target_full(full_kspace: torch.Tensor) -> torch.Tensor:
    """RSS of |IFFT2(full_kspace)| at the full matrix size, shape (B, H, W).

    Multi-coil k-space is (B, C, H, W, 2). Single-coil k-space (B, H, W, 2) gets a coil axis
    of size 1, so the RSS reduces to the complex magnitude.
    """
    coil_images = fastmri.ifft2c(full_kspace)
    if coil_images.ndim == 4:
        coil_images = coil_images.unsqueeze(1)
    return fastmri.rss(fastmri.complex_abs(coil_images), dim=1)


def image_loss(loss_fn: nn.Module, output: torch.Tensor, batch, crop_loss: bool) -> torch.Tensor:
    """Loss between the RSS output (B, H, W) at the full matrix size and its target.

    crop_loss=False (the default for every method): the target is ``rss_target_full`` of the fully
    sampled k-space, compared over the whole field of view.
    crop_loss=True: output and the h5 reconstruction (batch.target) are
    center-cropped to their common size.
    """
    if crop_loss:
        target, output = T.center_crop_to_smallest(batch.target, output)
    else:
        target = rss_target_full(batch.full_kspace)
    return loss_fn(output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value)


def flat_velocity_loss(kspace_after, kspace_before, x0, x_gt, block_idx, num_blocks, singlecoil):
    """FLAT flow-matching velocity loss (L2 in k-space) for the trained cascade.

    With timesteps t_k = block_idx / num_blocks, the straight-line flow-matching path runs
    from the zero-filled k-space x0 to the fully sampled x_gt:
        x*(t)   = (1 - t) * x0 + t * x_gt
    and the cascade's realized velocity is matched to the ideal one that steers x^{k}
    (kspace_before) onto that path at t_{k+1}:
        v_pred  = (x^{k+1} - x^{k}) / dt          (dt = 1 / num_blocks)
        v_ideal = (x*(t_{k+1}) - x^{k}) / dt
        L       = mean((v_pred - v_ideal) ** 2)
    Only v_pred carries gradients (x^{k} and the path are detached), so this updates only
    the trained cascade (and the sens-net through it).
    """
    if singlecoil:
        # k-space states from the model carry an explicit coil axis of size 1;
        # match it on the raw (coil-less) single-coil k-space.
        x0 = x0.unsqueeze(1)
        x_gt = x_gt.unsqueeze(1)
    t_kp1 = (block_idx + 1) / num_blocks
    dt = 1.0 / num_blocks
    x_star = (1.0 - t_kp1) * x0 + t_kp1 * x_gt
    v_pred = (kspace_after - kspace_before) / dt
    v_ideal = (x_star - kspace_before) / dt
    return F.mse_loss(v_pred, v_ideal)
