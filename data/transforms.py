"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Adapted from fastmri.data.transforms (VarNetSample, VarNetDataTransform); the sample also
carries the fully sampled k-space.
"""

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
from fastmri.data.subsample import MaskFunc
from fastmri.data.transforms import apply_mask, to_tensor


class VarNetSample(NamedTuple):
    """fastMRI's ``VarNetSample`` plus ``full_kspace``, the fully sampled k-space (loss target)."""

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: torch.Tensor
    crop_size: Tuple[int, int]
    full_kspace: torch.Tensor


class VarNetDataTransform:
    """Undersamples one slice into a ``VarNetSample``.

    use_seed=True seeds the mask from the file name (one fixed mask per volume, for
    validation); False draws from ``mask_func.rng`` (training).
    """

    def __init__(self, mask_func: MaskFunc, use_seed: bool = True):
        if mask_func is None:
            raise ValueError("VarNetDataTransform needs a mask_func")
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: Optional[np.ndarray],
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> VarNetSample:
        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        seed = tuple(map(ord, fname)) if self.use_seed else None
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        masked_kspace, mask_torch, num_low_frequencies = apply_mask(
            kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
        )

        return VarNetSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=torch.tensor(num_low_frequencies),
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=torch.tensor([max_value]),
            crop_size=crop_size,
            full_kspace=kspace_torch,
        )
