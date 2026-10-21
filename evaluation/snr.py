"""K-space SNR proxy behind the paper's high/low SNR split.

Computed per slice on the raw, fully sampled k-space (before undersampling), DC at the
array center:

    p      = |k|^2 (single-coil) or sum_c |k_c|^2 (multi-coil), shape (H, W)
    signal = mean of p over the S x S window at the center (S = 32)
    noise  = mean of the four S x S corner window means; the corners sit inside the
             acquired columns [padding_left, padding_right) so zero padding cannot fake
             zero noise
    snr_db = 10 log10(signal / noise), rounded to 3 decimals
"""

from typing import Optional, Tuple

import numpy as np

WINDOW = 32
SNR_THRESHOLD_DB = 35.0


def round_snr_db(snr_db):
    """Round to 3 decimals the way the paper's CSVs did (formatting with '%.3f')."""
    if np.ndim(snr_db) == 0:
        return float(f"{float(snr_db):.3f}")
    arr = np.asarray(snr_db, dtype=np.float64)
    return np.array([float(f"{v:.3f}") for v in arr.ravel()]).reshape(arr.shape)


def _acquired_columns(
    p: np.ndarray, padding_left: Optional[int], padding_right: Optional[int], multicoil: bool
) -> Tuple[int, int]:
    W = p.shape[1]
    pl = int(padding_left or 0)
    pr = int(padding_right or W)
    pl = max(0, min(pl, W))
    pr = max(0, min(pr, W))
    if multicoil:
        if pr - pl < 2 * WINDOW or (pl == 0 and pr == W):
            nz = np.where(p.sum(0) > 0)[0]
            if len(nz) and (nz.max() + 1 - nz.min()) >= 2 * WINDOW:
                pl, pr = int(nz.min()), int(nz.max()) + 1
            else:
                pl, pr = 0, W
    elif pr - pl < 2 * WINDOW:
        pl, pr = 0, W
    return pl, pr


def kspace_snr_db(
    kspace: np.ndarray,
    padding_left: Optional[int],
    padding_right: Optional[int],
    multicoil: bool,
) -> float:
    """SNR proxy of one slice in dB, rounded to 3 decimals.

    Args:
        kspace: complex k-space as read from the h5 file, (H, W) for single-coil or
            (C, H, W) for multi-coil. Pass it unconverted (complex64) to reproduce the
            paper's values exactly.
        padding_left, padding_right: the slice's acquired column range, from the
            SliceDataset attrs. None falls back to 0 and W.
        multicoil: selects the coil sum and the multi-coil column fallback.
    """
    k = np.asarray(kspace)
    if not np.iscomplexobj(k):
        raise TypeError(f"kspace must be complex, got dtype {k.dtype}")
    if k.ndim != (3 if multicoil else 2):
        raise ValueError(
            f"expected {'(C, H, W)' if multicoil else '(H, W)'} k-space, got shape {k.shape}"
        )
    p = (np.abs(k) ** 2).sum(0) if multicoil else np.abs(k) ** 2
    H, W = p.shape
    S = WINDOW
    if H < S or W < S:
        raise ValueError(f"k-space {H}x{W} is smaller than the {S}x{S} SNR window")
    pl, pr = _acquired_columns(p, padding_left, padding_right, multicoil)

    cy, cx = H // 2, W // 2
    sig = float(p[cy - S // 2:cy + S // 2, cx - S // 2:cx + S // 2].mean())
    corners = [p[:S, pl:pl + S], p[:S, pr - S:pr], p[H - S:, pl:pl + S], p[H - S:, pr - S:pr]]
    noise = float(np.mean([c.mean() for c in corners]))
    snr = sig / max(noise, 1e-30)
    return round_snr_db(10 * np.log10(max(snr, 1e-30)))


def split_by_snr(snr_db, threshold: float = SNR_THRESHOLD_DB) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean masks (high, low) with high = snr_db > threshold and low = the rest.

    Values are rounded to 3 decimals first (a no-op for kspace_snr_db output), matching
    the paper. Non-finite values raise instead of landing silently in either group.
    """
    snr_db = round_snr_db(np.asarray(snr_db, dtype=np.float64))
    snr_db = np.atleast_1d(snr_db)
    if not np.all(np.isfinite(snr_db)):
        raise ValueError("snr_db contains non-finite values")
    high = snr_db > threshold
    return high, ~high
