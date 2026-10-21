"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Optional time-conditioned U-Net variants for the shared regularizer of
SampledVarNet, selected with ``time_embedded_unet_arch``:

- ``"standard_attn"``: ``AttentionTimeUnet``, the standard additive time
  conditioning plus linear attention at the bottleneck.
- ``"standard_adaln"``: ``AdaLNTimeUnet``, time conditioning through AdaLN-Zero
  scale and shift after every instance norm.
- ``"standard_adaln_attn"``: ``AttentionAdaLNTimeUnet``, both of the above.

The paper uses ``"standard"`` (``modeling.blocks.TimeUnet``). All variants share
its encoder/decoder skeleton and switchable layers, so the suffix keeps the
activation-light backward. The bottleneck attention always uses standard
autograd (small at that resolution) but detaches its parameters in the suffix.
"""
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from modeling.activation_light import (
    MemEffAvgPool2dFunction,
    SwitchableConv2d,
    SwitchableInstanceNorm,
    SwitchableLeakyReLU,
    SwitchableLinear,
    set_memory_efficient,
)
from modeling.blocks import (
    SinusoidalPosEmb,
    TimeConvBlock,
    TransposeConvBlock,
    pad_to_skip,
)


# =============================================================================
# Linear-attention helpers (bottleneck only)
# =============================================================================

class _LayerNorm2d(nn.Module):
    """Channel LayerNorm; detaches its gain and bias in memory-efficient mode."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.memory_efficient = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        if self.memory_efficient:
            return (x - mean) / (var + self.eps).sqrt() * self.g.detach() + self.b.detach()
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class _Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = _LayerNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class _LinearAttention(nn.Module):
    """Linear self-attention (Shen et al., 2021); detaches its conv weights in
    memory-efficient mode."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.memory_efficient = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if self.memory_efficient:
            qkv = F.conv2d(x, self.to_qkv.weight.detach(), bias=None).chunk(3, dim=1)
        else:
            qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [t.reshape(b, self.heads, self.dim_head, h * w) for t in qkv]
        q = q * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = out.reshape(b, self.heads * self.dim_head, h, w)
        if self.memory_efficient:
            out_b = self.to_out.bias.detach() if self.to_out.bias is not None else None
            return F.conv2d(out, self.to_out.weight.detach(), bias=out_b)
        return self.to_out(out)


# =============================================================================
# AdaLN (adaptive scale and shift) helpers
# =============================================================================

class MemEffFiLMFunction(torch.autograd.Function):
    """
    FiLM modulation ``y = gamma * x + beta`` that saves only ``gamma``.

    Backward: grad_x = gamma * grad_y; the gradients w.r.t. gamma and beta are
    dropped, so the AdaLN MLP receives no update from the suffix. Composed after the
    linearized instance norm this gives grad_x = gamma * grad_y / std.
    """

    @staticmethod
    def forward(ctx, x, gamma, beta):
        ctx.save_for_backward(gamma)
        return gamma * x + beta

    @staticmethod
    def backward(ctx, grad_output):
        gamma, = ctx.saved_tensors
        return gamma * grad_output, None, None


class SwitchableAdaLN(nn.Module):
    """
    Instance norm followed by time-conditioned scale and shift (AdaLN-Zero).

    ``(g, b) = MLP(t_emb)`` and ``y = (1 + g) * InstanceNorm(x) + b``. The MLP
    output layer is zero-initialized, so at step 0 the block is a plain
    instance norm. In memory-efficient mode the norm is linearized (saves
    std), the modulation uses ``MemEffFiLMFunction`` (saves gamma) and the
    MLP receives no gradient.
    """

    def __init__(self, num_channels: int, time_emb_dim: int):
        super().__init__()
        self.num_channels = num_channels

        self.instance_norm = SwitchableInstanceNorm(num_channels)

        self.ada_mlp = nn.Sequential(
            nn.SiLU(),
            SwitchableLinear(time_emb_dim, 2 * num_channels),
        )
        nn.init.zeros_(self.ada_mlp[-1].weight)
        nn.init.zeros_(self.ada_mlp[-1].bias)

        self.memory_efficient = False

    def _gamma_beta(
        self, t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-channel scale (with the +1 offset) and shift, each (B, C, 1, 1)."""
        gb = self.ada_mlp(t_emb)                               # (B, 2C)
        g, b = gb.chunk(2, dim=-1)                             # (B, C) each
        gamma = (1.0 + g).view(-1, self.num_channels, 1, 1)
        beta = b.view(-1, self.num_channels, 1, 1)
        return gamma, beta

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self._gamma_beta(t_emb)
        x_norm = self.instance_norm(x)

        if self.memory_efficient:
            return MemEffFiLMFunction.apply(x_norm, gamma, beta)
        return gamma * x_norm + beta


class AdaLNTimeConvBlock(nn.Module):
    """
    Conv block with AdaLN time conditioning, a drop-in replacement for
    ``TimeConvBlock``:

        Conv, AdaLN(t_emb), LeakyReLU, Dropout, Conv, AdaLN(t_emb), LeakyReLU, Dropout
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        drop_prob: float,
        time_emb_dim: int,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.conv1 = SwitchableConv2d(
            in_chans, out_chans, kernel_size=3, padding=1, bias=False
        )
        self.adaln1 = SwitchableAdaLN(out_chans, time_emb_dim)
        self.act1 = SwitchableLeakyReLU(negative_slope=0.2, inplace=True)
        self.drop1 = nn.Dropout2d(drop_prob)

        self.conv2 = SwitchableConv2d(
            out_chans, out_chans, kernel_size=3, padding=1, bias=False
        )
        self.adaln2 = SwitchableAdaLN(out_chans, time_emb_dim)
        self.act2 = SwitchableLeakyReLU(negative_slope=0.2, inplace=True)
        self.drop2 = nn.Dropout2d(drop_prob)

    def forward(
        self, image: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(image)
        x = self.adaln1(x, t_emb)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.adaln2(x, t_emb)
        x = self.act2(x)
        x = self.drop2(x)
        return x


# =============================================================================
# U-Net variants (same interface as modeling.blocks.TimeUnet)
# =============================================================================

class _TimeUnetVariant(nn.Module):
    """Shared skeleton: sinusoidal time MLP, encoder/decoder of ``block_cls``
    with skip connections, optional bottleneck attention, final 1x1 conv."""

    block_cls = TimeConvBlock
    bottleneck_attention = False

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        block = self.block_cls

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        # Time embedding: cascade index -> sinusoidal -> MLP
        time_dim = chans * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(chans),
            SwitchableLinear(chans, time_dim),
            nn.GELU(),
            SwitchableLinear(time_dim, time_dim),
        )

        # Encoder
        self.down_sample_layers = nn.ModuleList(
            [block(in_chans, chans, drop_prob, time_dim)]
        )
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(block(ch, ch * 2, drop_prob, time_dim))
            ch *= 2

        # Bottleneck (the only place attention runs)
        self.conv = block(ch, ch * 2, drop_prob, time_dim)
        if self.bottleneck_attention:
            self.mid_attn = _Residual(_PreNorm(ch * 2, _LinearAttention(ch * 2)))

        # Decoder
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(block(ch * 2, ch, drop_prob, time_dim))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(block(ch * 2, ch, drop_prob, time_dim))

        # Final 1x1 projection (no time conditioning)
        self.final_conv = SwitchableConv2d(ch, out_chans, kernel_size=1, stride=1, bias=True)

    def set_memory_efficient_backprop(self, enabled: bool):
        """Toggle every switchable layer (including the attention and AdaLN
        helpers). ``requires_grad`` is not touched because the model is shared
        with the sampled cascade (see ``TimeUnet``)."""
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

        for layer in self.down_sample_layers:
            output = layer(output, t_emb)
            stack.append(output)
            output = MemEffAvgPool2dFunction.apply(output, 2, 2)

        output = self.conv(output, t_emb)
        if self.bottleneck_attention:
            output = self.mid_attn(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = pad_to_skip(transpose_conv(output), downsample_layer)
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output, t_emb)

        return self.final_conv(output)


class AttentionTimeUnet(_TimeUnetVariant):
    """Additive time conditioning plus linear attention at the bottleneck.

    Attention runs only at the bottleneck: at encoder/decoder levels it saved
    several full-resolution maps per call (no activation-light backward), about
    16x the bottleneck cost at the top level, while only the bottleneck sees
    the whole field of view anyway.
    """
    block_cls = TimeConvBlock
    bottleneck_attention = True


class AdaLNTimeUnet(_TimeUnetVariant):
    """AdaLN-Zero time conditioning (``AdaLNTimeConvBlock``), no attention.
    Transpose-conv blocks are not time-conditioned, as in ``TimeUnet``."""
    block_cls = AdaLNTimeConvBlock
    bottleneck_attention = False


class AttentionAdaLNTimeUnet(_TimeUnetVariant):
    """AdaLN-Zero time conditioning plus linear attention at the bottleneck."""
    block_cls = AdaLNTimeConvBlock
    bottleneck_attention = True


TIME_UNET_VARIANTS = {
    "standard_attn": AttentionTimeUnet,
    "standard_adaln": AdaLNTimeUnet,
    "standard_adaln_attn": AttentionAdaLNTimeUnet,
}
