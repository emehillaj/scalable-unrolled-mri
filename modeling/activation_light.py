"""
Activation-light autograd Functions and switchable layers.

They are used in the suffix (the cascades after the sampled one):
gradients flow through them to the sampled cascade, but their parameters receive
no gradient and almost no activations are saved for backward.

Each Switchable* layer behaves exactly like its torch counterpart until its
``memory_efficient`` flag is set (see ``set_memory_efficient``); the forward
values are the same in both modes, only the backward and the saved tensors
differ.

The forward of ``MemEffSensReduceFunction`` / ``MemEffSensExpandFunction`` is
the ``sens_reduce`` / ``sens_expand`` of fastMRI's VarNet
(https://github.com/facebookresearch/fastMRI, Copyright (c) Facebook, Inc. and
its affiliates, MIT license).
"""
import torch
from torch import nn
from torch.nn import functional as F
import fastmri


# =============================================================================
# Custom autograd Functions
# =============================================================================

class MemEfficientLeakyReLUFunction(torch.autograd.Function):
    """
    LeakyReLU that saves only the sign mask for backward, in one of two formats:

      * ``bitpack=True`` (default): 8 booleans packed into one ``uint8``, i.e.
        1 bit per element (8x smaller than a ``torch.bool`` mask). Packing and
        unpacking add a few cheap ops (about 20% slower on the ReLU alone).
      * ``bitpack=False``: a ``torch.bool`` mask (1 byte per element).

    The backward is identical to ``F.leaky_relu`` in both cases:
    ``grad * (mask ? 1 : negative_slope)``.
    """

    @staticmethod
    def _pack(mask: torch.Tensor):
        """Flatten, pad to a multiple of 8 and pack 8 bools per uint8.

        Returns (packed_uint8, original_shape, original_numel).
        """
        shape = tuple(mask.shape)
        n = mask.numel()
        flat = mask.reshape(-1).to(torch.uint8)
        pad = (-n) % 8
        if pad:
            flat = F.pad(flat, (0, pad))
        shifts = torch.arange(8, device=flat.device, dtype=torch.uint8)
        # bit i of byte j = flat[8j + i]
        packed = (flat.view(-1, 8) << shifts).sum(dim=1, dtype=torch.uint8)
        return packed, shape, n

    @staticmethod
    def _unpack(packed: torch.Tensor, shape, numel: int) -> torch.Tensor:
        shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
        bits = (packed.unsqueeze(1) >> shifts) & 1   # (n_bytes, 8) in {0, 1}
        flat = bits.reshape(-1)[:numel]
        return flat.view(shape).bool()

    @staticmethod
    def forward(ctx, input, negative_slope, bitpack):
        mask = input > 0
        if bitpack:
            packed, shape, numel = MemEfficientLeakyReLUFunction._pack(mask)
            ctx.save_for_backward(packed)
            ctx.mask_shape = shape
            ctx.mask_numel = numel
        else:
            ctx.save_for_backward(mask)
        ctx.bitpack = bitpack
        ctx.negative_slope = negative_slope
        return torch.where(mask, input, input * negative_slope)

    @staticmethod
    def backward(ctx, grad_output):
        (saved,) = ctx.saved_tensors
        if ctx.bitpack:
            mask = MemEfficientLeakyReLUFunction._unpack(
                saved, ctx.mask_shape, ctx.mask_numel
            )
        else:
            mask = saved
        grad_input = torch.where(mask, grad_output, grad_output * ctx.negative_slope)
        return grad_input, None, None


class LinearizedInstanceNormFunction(torch.autograd.Function):
    """
    InstanceNorm2d forward with a linearized backward.

    Forward:  y = (x - mean) / std    (standard instance norm, eps 1e-5)
    Backward: grad_input = grad_output / std    (drops the mean/variance terms)

    Only ``std`` (one value per sample and channel) is saved instead of the
    full activation.
    """
    @staticmethod
    def forward(ctx, input):
        mean = input.mean(dim=(2, 3), keepdim=True)
        var = input.var(dim=(2, 3), keepdim=True, unbiased=False) + 1e-5
        std = var.sqrt()
        output = (input - mean) / std
        ctx.save_for_backward(std)          # shape (N, C, 1, 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        std, = ctx.saved_tensors
        return grad_output / std,


class MemEfficientConv2dFunction(torch.autograd.Function):
    """
    Conv2d that saves only the weight for backward, not the input activation.

    grad_input is computed with ``conv_transpose2d(grad_output, weight)``.
    grad_weight and grad_bias are intentionally not computed, so suffix
    cascades do not update conv parameters. In the non-shared case the caller
    also sets ``requires_grad=False`` on the weights.
    """
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(weight)
        ctx.input_size = input.shape
        ctx.stride = stride if not isinstance(stride, int) else (stride, stride)
        ctx.padding = padding if not isinstance(padding, int) else (padding, padding)
        ctx.dilation = dilation if not isinstance(dilation, int) else (dilation, dilation)
        ctx.groups = groups
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation
        kernel_h, kernel_w = weight.shape[2], weight.shape[3]

        # output_padding so that conv_transpose2d recovers the input size exactly
        h_nominal = (grad_output.shape[2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + 1
        w_nominal = (grad_output.shape[3] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + 1
        output_padding = (ctx.input_size[2] - h_nominal, ctx.input_size[3] - w_nominal)

        grad_input = F.conv_transpose2d(
            grad_output, weight, bias=None,
            stride=stride, padding=padding,
            output_padding=output_padding,
            groups=ctx.groups, dilation=dilation,
        )
        # No grad_weight / grad_bias: suffix cascades must not update conv parameters.
        return grad_input, None, None, None, None, None, None


class MemEfficientConvTranspose2dFunction(torch.autograd.Function):
    """
    ConvTranspose2d that saves only the weight for backward, not the input
    activation. grad_input is computed with ``conv2d(grad_output, weight)``;
    grad_weight and grad_bias are intentionally not computed.
    """
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding, groups, dilation):
        ctx.save_for_backward(weight)
        ctx.stride = stride if not isinstance(stride, int) else (stride, stride)
        ctx.padding = padding if not isinstance(padding, int) else (padding, padding)
        ctx.dilation = dilation if not isinstance(dilation, int) else (dilation, dilation)
        ctx.groups = groups
        return F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_input = F.conv2d(
            grad_output, weight, bias=None,
            stride=ctx.stride, padding=ctx.padding,
            dilation=ctx.dilation, groups=ctx.groups,
        )
        return grad_input, None, None, None, None, None, None, None


class MemEffSensReduceFunction(torch.autograd.Function):
    """
    Memory-efficient ``sens_reduce``:

        image = sum_c IFFT(kspace_c) * conj(sens_maps_c)    (linear in kspace)

    The backward is the analytic adjoint, computed from ``grad_image`` and
    ``sens_maps`` alone, so no full-resolution intermediate (the IFFT output,
    the complex product) is saved. ``sens_maps`` is already alive, so saving
    it costs nothing.

    ``grad_sens_maps`` is intentionally ``None``: the suffix does not
    back-propagate into the sensitivity maps. For multi-coil data the sens-net
    therefore receives gradient only through the sampled cascade (the prefix
    runs under ``no_grad``).
    """

    @staticmethod
    def forward(ctx, kspace, sens_maps):
        ctx.save_for_backward(sens_maps)
        return fastmri.complex_mul(
            fastmri.ifft2c(kspace), fastmri.complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)

    @staticmethod
    def backward(ctx, grad_image):
        (sens_maps,) = ctx.saved_tensors
        # Adjoint of (ifft, multiply by conj(s), sum over coils) is
        # (broadcast over coils, multiply by s, fft).
        grad_image_coil = grad_image.expand_as(sens_maps)
        grad_kspace = fastmri.fft2c(
            fastmri.complex_mul(grad_image_coil, sens_maps)
        )
        return grad_kspace, None


class MemEffSensExpandFunction(torch.autograd.Function):
    """
    Memory-efficient ``sens_expand``:

        kspace_out = FFT(image * sens_maps)    (linear in image)

    ``image`` has coil dim 1 and broadcasts over the coils. The backward is the
    analytic adjoint from ``grad_kspace_out`` and ``sens_maps`` alone; the input
    image and the complex product are not saved. ``grad_sens_maps`` is
    intentionally ``None`` (see ``MemEffSensReduceFunction``).
    """

    @staticmethod
    def forward(ctx, image, sens_maps):
        ctx.save_for_backward(sens_maps)
        return fastmri.fft2c(fastmri.complex_mul(image, sens_maps))

    @staticmethod
    def backward(ctx, grad_kspace_out):
        (sens_maps,) = ctx.saved_tensors
        # Adjoint of (multiply by s, fft) is (ifft, multiply by conj(s)); the
        # sum over coils undoes the broadcast of image (B, 1, H, W, 2).
        grad_image_coil = fastmri.complex_mul(
            fastmri.ifft2c(grad_kspace_out),
            fastmri.complex_conj(sens_maps),
        )
        return grad_image_coil.sum(dim=1, keepdim=True), None


class MemEffSoftDCCombineFunction(torch.autograd.Function):
    """
    Memory-efficient soft data-consistency combine of the activation-light
    VarNet block:

        soft_dc = where(mask, current_kspace - ref_kspace, 0) * dc_w
        output  = current_kspace - soft_dc + model_term

    Everything is linear in (current_kspace, ref_kspace, model_term), so the
    backward is analytic and only ``mask`` and the scalar ``dc_w`` are saved.

    The gradient w.r.t. ``dc_w`` is not supported: the suffix freezes
    ``dc_weight`` (``set_memory_efficient_backprop``). If a caller still asks
    for it, the backward raises instead of returning a silently wrong value.
    """

    @staticmethod
    def forward(ctx, current_kspace, ref_kspace, mask, dc_w, model_term):
        ctx.save_for_backward(mask, dc_w)
        zero = torch.zeros((), device=current_kspace.device, dtype=current_kspace.dtype)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * dc_w
        return current_kspace - soft_dc + model_term

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[3]:
            raise RuntimeError(
                "MemEffSoftDCCombineFunction: grad w.r.t. dc_w is not "
                "supported in memory-efficient mode. Suffix cascades must call "
                "set_memory_efficient_backprop(True), which freezes dc_weight."
            )
        mask, dc_w = ctx.saved_tensors
        zero = torch.zeros_like(dc_w)
        masked_dc_w = torch.where(mask, dc_w, zero)
        grad_current = grad_output * (1.0 - masked_dc_w)
        grad_ref = grad_output * masked_dc_w if ctx.needs_input_grad[1] else None
        grad_model_term = grad_output
        return grad_current, grad_ref, None, None, grad_model_term


class MemEffNormalizeFunction(torch.autograd.Function):
    """
    ``(x - mean) / std`` with mean and std treated as constants (the suffix
    detaches them). Saves only ``std``.

    The plain composition routes through SubBackward and DivBackward, which
    save ``x`` and ``x - mean`` even when mean and std need no gradient, so
    one full-resolution activation per cascade would stay alive.
    """

    @staticmethod
    def forward(ctx, x, mean, std):
        ctx.save_for_backward(std)
        return (x - mean) / std

    @staticmethod
    def backward(ctx, grad_y):
        (std,) = ctx.saved_tensors
        return grad_y / std, None, None


class MemEffDenormalizeFunction(torch.autograd.Function):
    """
    ``x * std`` (the undo-std step at the end of ``NormUnet.forward``). Saves
    only ``std``; unlike ``torch.mul`` it does not save ``x``, which would only
    be needed for the gradient w.r.t. std (detached in the suffix).
    """

    @staticmethod
    def forward(ctx, x, std):
        ctx.save_for_backward(std)
        return x * std

    @staticmethod
    def backward(ctx, grad_y):
        (std,) = ctx.saved_tensors
        return grad_y * std, None


class MemEffAvgPool2dFunction(torch.autograd.Function):
    """
    ``F.avg_pool2d`` without a saved input.

    Average pooling is linear, so the backward needs only the input shape and
    the kernel size:

        grad_input = upsample(grad_output, factor=k) / k**2

    zero-padded to the input size when H or W is not a multiple of ``k`` (the
    U-Net pads to multiples of 16, so this is a no-op there). Requires
    non-overlapping pooling (``stride == kernel_size``, ``padding == 0``).
    The forward value is identical to ``F.avg_pool2d``, so it is used in both
    modes.
    """

    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        if kernel_size != stride or kernel_size <= 0:
            raise ValueError(
                "MemEffAvgPool2dFunction requires non-overlapping pooling "
                f"(kernel_size == stride > 0); got kernel_size={kernel_size}, "
                f"stride={stride}."
            )
        ctx.kernel_size = kernel_size
        ctx.input_hw = (x.shape[-2], x.shape[-1])
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)

    @staticmethod
    def backward(ctx, grad_output):
        k = ctx.kernel_size
        in_h, in_w = ctx.input_hw
        grad = F.interpolate(grad_output, scale_factor=k, mode="nearest") / (k * k)
        pad_h = in_h - grad.shape[-2]
        pad_w = in_w - grad.shape[-1]
        if pad_h > 0 or pad_w > 0:
            grad = F.pad(grad, (0, pad_w, 0, pad_h))
        return grad, None, None


# =============================================================================
# Switchable layers
# =============================================================================

def set_memory_efficient(module: nn.Module, enabled: bool) -> None:
    """Set the ``memory_efficient`` flag of every switchable layer in ``module``
    (any submodule that defines that attribute)."""
    for m in module.modules():
        if hasattr(m, "memory_efficient"):
            m.memory_efficient = enabled


class SwitchableLeakyReLU(nn.Module):
    """LeakyReLU with an optional activation-light backward.

    ``bitpack_mask`` (default True) selects the saved mask format: 1 bit per
    element (True) or a torch.bool mask with 1 byte per element (False). It is
    set for the whole model by ``SampledVarNet.set_bitpack_relu_mask``.
    """
    def __init__(self, negative_slope: float = 0.2, inplace: bool = True):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.memory_efficient = False
        self.bitpack_mask = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.memory_efficient:
            return MemEfficientLeakyReLUFunction.apply(
                x, self.negative_slope, self.bitpack_mask
            )
        return F.leaky_relu(x, self.negative_slope, self.inplace)


class SwitchableInstanceNorm(nn.Module):
    """InstanceNorm2d with an optional linearized backward."""
    def __init__(self, num_features: int):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features)
        self.memory_efficient = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.memory_efficient:
            return LinearizedInstanceNormFunction.apply(x)
        return self.instance_norm(x)


class SwitchableConv2d(nn.Module):
    """Conv2d whose activation-light backward saves only the weight.

    In memory-efficient mode no weight gradient is produced (see
    ``MemEfficientConv2dFunction``).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.memory_efficient = False

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.memory_efficient:
            return MemEfficientConv2dFunction.apply(
                x, self.conv.weight, self.conv.bias,
                self.conv.stride, self.conv.padding,
                self.conv.dilation, self.conv.groups,
            )
        return self.conv(x)


class SwitchableConvTranspose2d(nn.Module):
    """ConvTranspose2d with the same activation-light backward as SwitchableConv2d."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias, dilation=dilation,
        )
        self.memory_efficient = False

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.memory_efficient:
            return MemEfficientConvTranspose2dFunction.apply(
                x, self.conv.weight, self.conv.bias,
                self.conv.stride, self.conv.padding,
                self.conv.output_padding, self.conv.groups,
                self.conv.dilation,
            )
        return self.conv(x)


class SwitchableLinear(nn.Module):
    """Linear layer that detaches its parameters in memory-efficient mode.

    The inputs are small per-sample time embeddings of shape ``(N, D)``, so no
    custom Function is needed: detaching is enough to keep the suffix from
    accumulating gradients on the shared time MLPs.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.memory_efficient = False

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.memory_efficient:
            w = self.linear.weight.detach()
            b = self.linear.bias.detach() if self.linear.bias is not None else None
            return F.linear(x, w, b)
        return self.linear(x)
