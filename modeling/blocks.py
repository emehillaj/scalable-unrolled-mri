"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Activation-light VarNet building blocks (adapted from fastMRI's VarNet), used by
SampledVarNet (ours and e2e): U-Net, NormUnet, SensitivityModel and the
cascade block, plus the shared time-conditioned regularizer (TimeUnet,
TimeNormUnet, TimeVarNetBlock; ``time_embedded_unet_arch="standard"``). The
optional attention / AdaLN regularizers live in ``modeling.time_unet_variants``.
"""
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import fastmri
from fastmri.data import transforms

from modeling.activation_light import (
    MemEffAvgPool2dFunction,
    MemEffDenormalizeFunction,
    MemEffNormalizeFunction,
    MemEffSensExpandFunction,
    MemEffSensReduceFunction,
    MemEffSoftDCCombineFunction,
    SwitchableConv2d,
    SwitchableConvTranspose2d,
    SwitchableInstanceNorm,
    SwitchableLeakyReLU,
    SwitchableLinear,
    set_memory_efficient,
)


def pad_to_skip(output: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Reflect-pad on the right/bottom when odd input sizes left the upsampled
    map one pixel smaller than the skip connection."""
    padding = [0, 0, 0, 0]
    if output.shape[-1] != skip.shape[-1]:
        padding[1] = 1  # padding right
    if output.shape[-2] != skip.shape[-2]:
        padding[3] = 1  # padding bottom
    if sum(padding) != 0:
        output = F.pad(output, padding, "reflect")
    return output


# =============================================================================
# Per-cascade regularizer (e2e / non-shared)
# =============================================================================

class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234-241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                SwitchableConv2d(ch, self.out_chans, kernel_size=1, stride=1, bias=True),
            )
        )

    def set_memory_efficient_backprop(self, enabled: bool):
        """Toggle the activation-light backward of every switchable layer.

        When enabled:
          - SwitchableConv2d / SwitchableConvTranspose2d save only the weight
            (already resident as a parameter) instead of the input activation.
          - SwitchableLeakyReLU saves only the sign mask (1 bit per element by
            default, 1 byte with ``bitpack_mask=False``).
          - SwitchableInstanceNorm uses a linearized backward (grad / std) and
            saves only the per-channel std.
          - every parameter of this U-Net gets ``requires_grad=False``, so no
            parameter of a suffix cascade is updated. Per-cascade U-Nets are
            not shared, so this only affects this cascade.
        """
        set_memory_efficient(self, enabled)
        for p in self.parameters():
            p.requires_grad_(not enabled)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = MemEffAvgPool2dFunction.apply(output, 2, 2)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = pad_to_skip(transpose_conv(output), downsample_layer)
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            SwitchableConv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            SwitchableInstanceNorm(out_chans),
            SwitchableLeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            SwitchableConv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            SwitchableInstanceNorm(out_chans),
            SwitchableLeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            SwitchableConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            SwitchableInstanceNorm(out_chans),
            SwitchableLeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


class _NormPadMixin:
    """Complex/channel reshapes, group normalization and padding shared by
    NormUnet and TimeNormUnet. In memory-efficient mode the normalization
    treats mean and std as constants and saves only std."""

    _memory_efficient = False

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        # Expand mean and std to match c channels (mix real and imaginary)
        mean = mean.expand(b, c, 1, 1)
        std = std.expand(b, c, 1, 1)

        if self._memory_efficient:
            # Suffix: a custom Function that saves only std (the plain
            # composition would keep full-resolution x and x - mean alive).
            mean = mean.detach()
            std = std.detach()
            x_normed = MemEffNormalizeFunction.apply(x, mean, std)
        else:
            x_normed = (x - mean) / std

        return x_normed, mean, std

    def denorm(self, x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # only undo the std scaling, no mean shift (the output is a correction,
        # not a full signal)
        if self._memory_efficient:
            return MemEffDenormalizeFunction.apply(x, std)
        return x * std

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]


class NormUnet(_NormPadMixin, nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

        self._memory_efficient = False

    def set_memory_efficient_backprop(self, enabled: bool):
        """Toggle the inner U-Net and the linearized norm / denorm."""
        self._memory_efficient = enabled
        self.unet.set_memory_efficient_backprop(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)

        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)

        x = self.denorm(x, std)
        return self.chan_complex_to_last_dim(x)


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self,
        mask: torch.Tensor,
        num_low_frequencies: Optional[Union[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # num_low_frequencies may be a (B,) tensor for batch > 1, so reduce
        # explicitly instead of relying on a scalar truth test.
        if num_low_frequencies is None:
            use_mask = True
        elif torch.is_tensor(num_low_frequencies):
            use_mask = bool((num_low_frequencies == 0).all())
        else:
            use_mask = num_low_frequencies == 0
        if use_mask:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            nlf = num_low_frequencies
            if torch.is_tensor(nlf):
                nlf = nlf.to(device=mask.device)
            num_low_frequencies_tensor = nlf * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class _SensMixin:
    """Coil expand / reduce of the VarNet data-consistency step."""

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)


# Note on the DC sign: the activation-light blocks below return
# ``current - soft_dc + model_term``, while the fastMRI-style VarNetBlock in
# modeling/varnet_blocks.py (GLEAM / FLAT) returns ``current - soft_dc -
# model_term``. Both conventions come from the research code that produced the
# paper checkpoints; the regularizer learns either sign, so they are kept as is.

class VarNetBlock(_SensMixin, nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module, num_cascades: int = 1):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
            num_cascades: Total number of cascades; used to scale the last
                conv of the regularizer U-Net by 1/sqrt(num_cascades).
        """
        super().__init__()

        self.model = model
        self.num_cascades = num_cascades
        self.memory_efficient_dc = False
        self.dc_weight = nn.Parameter(torch.ones(1))
        # Scale the last conv by 1/sqrt(T); dc_weight stays at 1 for strong DC.
        if num_cascades > 1:
            scale = 1.0 / math.sqrt(num_cascades)
            final_conv = self.model.unet.up_conv[-1][1].conv
            with torch.no_grad():
                final_conv.weight.mul_(scale)
                if final_conv.bias is not None:
                    final_conv.bias.mul_(scale)

    def set_memory_efficient_backprop(self, enabled: bool):
        """Toggle memory-efficient mode on the regularizer and freeze dc_weight."""
        self.model.set_memory_efficient_backprop(enabled)
        self.dc_weight.requires_grad_(not enabled)
        self.memory_efficient_dc = enabled

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        dc_w = self.dc_weight

        if self.memory_efficient_dc:
            # Suffix: the linear DC steps use analytic backwards, so no
            # full-resolution k-space activation is saved per cascade.
            image_in = MemEffSensReduceFunction.apply(current_kspace, sens_maps)
            image_out = self.model(image_in)
            model_term = MemEffSensExpandFunction.apply(image_out, sens_maps)
            return MemEffSoftDCCombineFunction.apply(
                current_kspace, ref_kspace, mask, dc_w, model_term
            )

        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * dc_w
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )
        return current_kspace - soft_dc + model_term


# =============================================================================
# Shared time-conditioned regularizer ("standard" arch)
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for cascade index conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeConvBlock(nn.Module):
    """
    Convolutional block with time (cascade index) conditioning.
    Structure: Conv, InstanceNorm, LeakyReLU, Dropout, (+ time), Conv, InstanceNorm, LeakyReLU, Dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, time_emb_dim: int):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers1 = nn.Sequential(
            SwitchableConv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            SwitchableInstanceNorm(out_chans),
            SwitchableLeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            SwitchableLinear(time_emb_dim, out_chans),
        )
        self.layers2 = nn.Sequential(
            SwitchableConv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            SwitchableInstanceNorm(out_chans),
            SwitchableLeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape ``(N, in_chans, H, W)``.
            t_emb: Time embedding of shape ``(N, time_emb_dim)``.

        Returns:
            Output tensor of shape ``(N, out_chans, H, W)``.
        """
        x = self.layers1(image)
        t = self.time_mlp(t_emb)
        x = x + t[:, :, None, None]
        x = self.layers2(x)
        return x


class TimeUnet(nn.Module):
    """
    U-Net with sinusoidal cascade-index (time) conditioning, added between the
    two convs of every block. Uses the same switchable layers as ``Unet``.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        # Time embedding: cascade index -> sinusoidal -> MLP
        time_dim = chans * 6
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(chans),
            SwitchableLinear(chans, time_dim),
            nn.GELU(),
            SwitchableLinear(time_dim, time_dim),
        )

        # Encoder
        self.down_sample_layers = nn.ModuleList(
            [TimeConvBlock(in_chans, chans, drop_prob, time_dim)]
        )
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(TimeConvBlock(ch, ch * 2, drop_prob, time_dim))
            ch *= 2
        self.conv = TimeConvBlock(ch, ch * 2, drop_prob, time_dim)

        # Decoder
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(TimeConvBlock(ch * 2, ch, drop_prob, time_dim))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(TimeConvBlock(ch * 2, ch, drop_prob, time_dim))

        # Final 1x1 projection (no time conditioning)
        self.final_conv = SwitchableConv2d(ch, out_chans, kernel_size=1, stride=1, bias=True)

    def set_memory_efficient_backprop(self, enabled: bool):
        """Toggle the activation-light backward of every switchable layer.

        Unlike the per-cascade ``Unet`` this does not touch ``requires_grad``:
        the model is shared with the sampled cascade, which still needs weight
        gradients. The custom Functions return no weight gradient and the
        SwitchableLinear layers detach their parameters, so the suffix adds
        nothing to the shared weights.
        """
        set_memory_efficient(self, enabled)

    def forward(self, image: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape ``(N, in_chans, H, W)``.
            t: Cascade index tensor of shape ``(N,)``.

        Returns:
            Output tensor of shape ``(N, out_chans, H, W)``.
        """
        t_emb = self.time_mlp(t)

        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output, t_emb)
            stack.append(output)
            output = MemEffAvgPool2dFunction.apply(output, 2, 2)

        output = self.conv(output, t_emb)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = pad_to_skip(transpose_conv(output), downsample_layer)
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output, t_emb)

        output = self.final_conv(output)
        return output


TIME_UNET_ARCHES = ("standard", "standard_attn", "standard_adaln", "standard_adaln_attn")


class TimeNormUnet(_NormPadMixin, nn.Module):
    """
    Normalized U-Net with time (cascade index) conditioning.

    Same normalization and padding as ``NormUnet``, wrapped around a
    time-conditioned U-Net chosen by ``time_embedded_unet_arch``:

    - ``"standard"`` (default, used in the paper): ``TimeUnet``, additive time
      conditioning.
    - ``"standard_attn"``, ``"standard_adaln"``, ``"standard_adaln_attn"``:
      the optional variants in ``modeling.time_unet_variants`` (bottleneck
      linear attention and/or AdaLN scale+shift conditioning).
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        time_embedded_unet_arch: str = "standard",
    ):
        super().__init__()

        self.time_embedded_unet_arch = time_embedded_unet_arch
        if time_embedded_unet_arch == "standard":
            unet_cls = TimeUnet
        elif time_embedded_unet_arch in TIME_UNET_ARCHES:
            # imported here because the variants build on the blocks in this file
            from modeling.time_unet_variants import TIME_UNET_VARIANTS
            unet_cls = TIME_UNET_VARIANTS[time_embedded_unet_arch]
        else:
            raise ValueError(
                f"Unknown time_embedded_unet_arch: {time_embedded_unet_arch!r}. "
                f"Choose from {', '.join(repr(a) for a in TIME_UNET_ARCHES)}."
            )
        self.unet = unet_cls(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

        self._memory_efficient = False

    def set_memory_efficient_backprop(self, enabled: bool):
        """Toggle the inner U-Net and the linearized norm / denorm."""
        self._memory_efficient = enabled
        self.unet.set_memory_efficient_backprop(enabled)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)

        x, pad_sizes = self.pad(x)
        x = self.unet(x, t)
        x = self.unpad(x, *pad_sizes)

        x = self.denorm(x, std)
        return self.chan_complex_to_last_dim(x)


class TimeVarNetBlock(_SensMixin, nn.Module):
    """
    VarNet block that passes the cascade index to a shared time-conditioned
    regularizer.

    The shared model is held in a plain list so it is not registered as a
    submodule here (no duplicate ``state_dict`` entries); it is registered once
    as ``SampledVarNet.shared_model``. Its last conv is scaled by 1/sqrt(T)
    there; dc_weight stays at 1 for strong data consistency.
    """

    def __init__(self, shared_model: nn.Module):
        super().__init__()
        self._shared_model_ref = [shared_model]
        self.memory_efficient_dc = False
        self.dc_weight = nn.Parameter(torch.ones(1))

    @property
    def model(self):
        return self._shared_model_ref[0]

    def set_memory_efficient_backprop(self, enabled: bool):
        """Toggle memory-efficient mode on the shared model and freeze dc_weight."""
        self.model.set_memory_efficient_backprop(enabled)
        self.dc_weight.requires_grad_(not enabled)
        self.memory_efficient_dc = enabled

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        dc_w = self.dc_weight

        if self.memory_efficient_dc:
            # Suffix: see VarNetBlock.forward
            image_in = MemEffSensReduceFunction.apply(current_kspace, sens_maps)
            image_out = self.model(image_in, t)
            model_term = MemEffSensExpandFunction.apply(image_out, sens_maps)
            return MemEffSoftDCCombineFunction.apply(
                current_kspace, ref_kspace, mask, dc_w, model_term
            )

        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * dc_w
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps), t),
            sens_maps,
        )
        return current_kspace - soft_dc + model_term
