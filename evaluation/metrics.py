"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

NMSE, PSNR and SSIM as in fastMRI's evaluate.py. data_range defaults to the target
maximum of the array passed in (one slice, unless called on a volume).
"""

from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Normalized mean squared error ||gt - pred||^2 / ||gt||^2."""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Peak signal-to-noise ratio in dB, with data_range = maxval (default gt.max())."""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Structural similarity averaged over the leading axis of (N, H, W) arrays, shape (1,)."""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval)

    return ssim / gt.shape[0]
