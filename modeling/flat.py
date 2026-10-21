"""FLAT-style baseline model (FlatVarNet)."""

from typing import Optional, Union

import torch
import torch.nn as nn
import fastmri

from modeling.varnet_blocks import NormUnet, SensitivityModel, VarNetBlock


class FlatVarNet(nn.Module):
    """
    FLAT-style baseline: flow-matching-inspired velocity supervision. As in the
    paper, only the local velocity loss is used; FLAT's additional end-to-end
    term is not.

    Similar to GLEAM it runs cascades ``0..cascade_to_train_idx`` (the trained cascade
    keeps gradients, the earlier cascades run under ``no_grad``), but it is
    trained with a k-space velocity loss instead of an image loss on the
    cascade output. With ``return_velocity=True`` the forward also returns the
    k-space state just after the trained cascade (``x^{k+1}``) and just before it
    (``x^{k}``); the trainer compares the realized velocity ``x^{k+1} - x^{k}``
    with the ideal straight-line velocity toward the fully sampled k-space
    (``flat_velocity_loss`` in training/losses.py).
    """
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        singlecoil: bool = False,
        zero_init_final_conv: bool = True,
    ):
        super().__init__()

        self.singlecoil = singlecoil
        if not singlecoil:
            self.sens_net = SensitivityModel(
                chans=sens_chans,
                num_pools=sens_pools,
                mask_center=mask_center,
            )
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools))
             for _ in range(num_cascades)]
        )
        if zero_init_final_conv:
            # As in the paper runs: the last conv of every cascade U-Net starts at
            # zero, so each cascade starts as the identity map. The sensitivity
            # network keeps its default initialization.
            for cascade in self.cascades:
                final_conv = cascade.model.unet.up_conv[-1][1]
                nn.init.zeros_(final_conv.weight)
                nn.init.zeros_(final_conv.bias)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[Union[int, torch.Tensor]] = None,
        cascade_to_train_idx: Optional[int] = None,
        return_velocity: bool = False,
    ):
        if self.singlecoil:
            # Single-coil data has no coil dim: (B, H, W, 2) -> (B, 1, H, W, 2)
            masked_kspace = masked_kspace.unsqueeze(1)
            mask = mask.unsqueeze(1)
            # Identity sens maps (real 1, imag 0): sens reduce / expand become ifft2c / fft2c
            sens_maps = torch.zeros_like(masked_kspace)
            sens_maps[..., 0] = 1.0
        else:
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()

        total_cascades = len(self.cascades)

        # x^{k}: k-space entering the trained cascade (the input for a full pass)
        kspace_before_block = masked_kspace

        if cascade_to_train_idx is not None:
            end_train_idx = min(cascade_to_train_idx, total_cascades - 1)
            start_train_idx = max(0, end_train_idx)

            # Cascades before the trained cascade: no gradients
            with torch.no_grad():
                for i in range(0, start_train_idx):
                    kspace_pred = self.cascades[i](kspace_pred, masked_kspace, mask, sens_maps)

            kspace_before_block = kspace_pred.detach().clone()

            # Trained cascade: with gradients. Nothing after it is run.
            for i in range(start_train_idx, end_train_idx + 1):
                kspace_pred = self.cascades[i](kspace_pred, masked_kspace, mask, sens_maps)
        else:
            for cascade in self.cascades:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        output = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

        if return_velocity:
            # (output, x^{k+1}, x^{k}); the k-space states keep the coil axis
            # (size 1 for single-coil data).
            return output, kspace_pred, kspace_before_block

        return output
