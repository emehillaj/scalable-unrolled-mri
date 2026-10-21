"""GLEAM baseline model (GleamVarNet)."""

from typing import Optional, Union

import torch
import torch.nn as nn
import fastmri

from modeling.varnet_blocks import NormUnet, SensitivityModel, VarNetBlock


class GleamVarNet(nn.Module):
    """
    GLEAM baseline: greedy, cascade-by-cascade training of a VarNet.

    With ``cascade_to_train_idx = j`` only cascades ``0..j`` are run and the
    loss is taken at the output of cascade j. Cascade j runs with gradients,
    the earlier ones under ``no_grad``, and no later cascade is run. With
    ``cascade_to_train_idx=None`` the full chain runs (eval).

    ``apply_dc=True`` replaces the acquired lines of the output k-space by the
    measured ones (hard data consistency, ``masked_kspace * mask + (1 - mask) *
    pred``) before the RSS image is formed. The paper GLEAM runs train with it
    and evaluate without it (the default).
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
        apply_dc: bool = False,
    ) -> torch.Tensor:
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

        if cascade_to_train_idx is not None:
            end_train_idx = min(cascade_to_train_idx, total_cascades - 1)
            start_train_idx = max(0, end_train_idx)

            # Cascades before the trained cascade: no gradients
            with torch.no_grad():
                for i in range(0, start_train_idx):
                    kspace_pred = self.cascades[i](kspace_pred, masked_kspace, mask, sens_maps)

            # Trained cascade: with gradients. Nothing after it is run.
            for i in range(start_train_idx, end_train_idx + 1):
                kspace_pred = self.cascades[i](kspace_pred, masked_kspace, mask, sens_maps)
        else:
            for cascade in self.cascades:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        if apply_dc:
            # Hard DC: keep the measured lines, use the prediction elsewhere
            mask_float = mask.to(kspace_pred.dtype)
            kspace_pred = masked_kspace * mask_float + (1 - mask_float) * kspace_pred

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
