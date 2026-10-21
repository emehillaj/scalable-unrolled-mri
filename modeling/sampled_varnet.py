"""SampledVarNet: ours (sampled cascade with an activation-light suffix) and e2e."""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import fastmri

from modeling.activation_light import SwitchableLeakyReLU
from modeling.blocks import (
    NormUnet,
    SensitivityModel,
    TimeNormUnet,
    TimeVarNetBlock,
    VarNetBlock,
)


class SampledVarNet(nn.Module):
    """
    VarNet that trains one sampled cascade per step (ours) or the whole chain
    (e2e), returning the RSS image ``(B, H, W)``.

    With ``cascade_to_train_idx = j`` the forward runs in the three parts of
    the paper:

    prefix (cascades before j): ``no_grad``, nothing is stored.
    sampled cascade j:          standard forward and backward.
    suffix (cascades after j):  gradients flow through to cascade j, but the
             suffix parameters get no gradient and the activation-light
             autograd Functions save almost nothing:
               - Conv / ConvTranspose save only the weight (no input).
               - LeakyReLU saves a 1-bit packed sign mask (1-byte bool mask
                 with ``bitpack_relu_mask=False``).
               - InstanceNorm uses a linearized backward (grad / std) and saves
                 only the per-channel std.
               - sens reduce / expand and the soft DC use analytic adjoints;
                 they do not back-propagate into the sensitivity maps, so the
                 sens-net is trained only through cascade j.

    With ``cascade_to_train_idx=None`` all cascades run with standard autograd
    (e2e). ``activation_checkpointing=True`` then recomputes each cascade in the
    backward (``torch.utils.checkpoint``, exact gradients, less memory, more
    compute). It only applies in training mode.

    ``use_time_embed=True`` shares one time-conditioned regularizer
    (``shared_model``) across all cascades; each cascade keeps its own
    ``dc_weight``. Otherwise every cascade has its own NormUnet.

    ``exact_suffix=True`` is an ablation: the suffix uses exact autograd with all
    parameters frozen, so the update set matches the activation-light path but
    the backward is exact (it also propagates into the sensitivity maps through
    the suffix). It costs the full suffix activation memory, so it is only
    practical for short chains.
    """
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        use_time_embed: bool = False,
        time_embedded_unet_arch: str = "standard",
        singlecoil: bool = False,
        bitpack_relu_mask: bool = True,
        exact_suffix: bool = False,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.exact_suffix = exact_suffix
        self.activation_checkpointing = activation_checkpointing
        self.use_time_embed = use_time_embed
        self.time_embedded_unet_arch = time_embedded_unet_arch
        self.singlecoil = singlecoil
        self.bitpack_relu_mask = bitpack_relu_mask

        if not singlecoil:
            self.sens_net = SensitivityModel(
                chans=sens_chans,
                num_pools=sens_pools,
                mask_center=mask_center,
            )

        if use_time_embed:
            # One time-conditioned U-Net shared by all cascades.
            self.shared_model = TimeNormUnet(
                chans, pools,
                time_embedded_unet_arch=time_embedded_unet_arch,
            )
            # Scale the last conv of the shared U-Net by 1/sqrt(T).
            if num_cascades > 1:
                final_conv = self.shared_model.unet.final_conv.conv
                with torch.no_grad():
                    scale = 1.0 / math.sqrt(num_cascades)
                    final_conv.weight.mul_(scale)
                    if final_conv.bias is not None:
                        final_conv.bias.mul_(scale)
            self.cascades = nn.ModuleList(
                [TimeVarNetBlock(self.shared_model)
                 for _ in range(num_cascades)]
            )
        else:
            self.cascades = nn.ModuleList(
                [VarNetBlock(NormUnet(chans, pools), num_cascades=num_cascades)
                 for _ in range(num_cascades)]
            )

        self.set_bitpack_relu_mask(self.bitpack_relu_mask)

    def set_bitpack_relu_mask(self, enabled: bool):
        """Set the saved-mask format of every SwitchableLeakyReLU. True (default)
        saves a 1-bit packed mask (8x smaller, about 20% slower per ReLU call);
        False saves a torch.bool mask. Lossless either way."""
        self.bitpack_relu_mask = enabled
        for m in self.modules():
            if isinstance(m, SwitchableLeakyReLU):
                m.bitpack_mask = enabled

    def _run_cascade(
        self,
        i: int,
        kspace_pred: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Run cascade ``i``; with ``use_time_embed`` the cascade index is the
        time input."""
        if self.use_time_embed:
            B = masked_kspace.shape[0]
            t = torch.full((B,), i, device=masked_kspace.device, dtype=torch.float32)
            return self.cascades[i](kspace_pred, masked_kspace, mask, sens_maps, t)
        return self.cascades[i](kspace_pred, masked_kspace, mask, sens_maps)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[Union[int, torch.Tensor]] = None,
        cascade_to_train_idx: Optional[int] = None,
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
            start_train_idx = max(0, cascade_to_train_idx)
            end_train_idx = min(cascade_to_train_idx, total_cascades - 1)

            # Prefix: cascades before the sampled cascade, no gradients
            with torch.no_grad():
                for i in range(0, start_train_idx):
                    kspace_pred = self._run_cascade(i, kspace_pred, masked_kspace, mask, sens_maps)

            # Sampled cascade: standard autograd
            for i in range(start_train_idx, end_train_idx + 1):
                kspace_pred = self._run_cascade(i, kspace_pred, masked_kspace, mask, sens_maps)

            # Suffix: the remaining cascades pass the gradient back to the sampled cascade
            suffix = range(end_train_idx + 1, total_cascades)
            if len(suffix) > 0:
                frozen = []
                try:
                    for i in suffix:
                        self.cascades[i].set_memory_efficient_backprop(not self.exact_suffix)
                    if self.exact_suffix:
                        # Freeze every parameter of the module that runs the suffix,
                        # not only cascades[i]: with use_time_embed the U-Net lives
                        # on shared_model and cascades[i] holds only dc_weight. The
                        # graph of the sampled cascade is already built, so its own
                        # gradients are unaffected.
                        for p in self.parameters():
                            if p.requires_grad:
                                p.requires_grad_(False)
                                frozen.append(p)
                    for i in suffix:
                        kspace_pred = self._run_cascade(i, kspace_pred, masked_kspace, mask, sens_maps)
                finally:
                    for i in suffix:
                        self.cascades[i].set_memory_efficient_backprop(False)
                    for p in frozen:
                        p.requires_grad_(True)
        elif self.activation_checkpointing and self.training:
            # e2e with activation checkpointing: recompute each cascade in backward
            for i in range(total_cascades):
                kspace_pred = checkpoint(
                    self._run_cascade, i, kspace_pred, masked_kspace, mask, sens_maps,
                    use_reentrant=False,
                )
        else:
            # e2e: all cascades with standard autograd
            for i in range(total_cascades):
                kspace_pred = self._run_cascade(i, kspace_pred, masked_kspace, mask, sens_maps)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
