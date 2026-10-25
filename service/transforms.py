"""Continuous-block calibration transforms."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class TextImageTransform(str, Enum):
    RAW = "raw"
    PERCENTILE = "percentile"
    TAIL_LOG = "tail_log"


def midrank_percentile(x: FloatArray, sample: FloatArray) -> FloatArray:
    """Vectorized mid-rank percentile in [0, 1]."""
    finite = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=np.float64)
    if sample.size == 0:
        return out
    s = np.sort(sample[np.isfinite(sample)])
    n = s.size
    if n == 0:
        return out
    xv = x[finite].astype(np.float64)
    left = np.searchsorted(s, xv, side="left")
    right = np.searchsorted(s, xv, side="right")
    out[finite] = (left.astype(np.float64) + right.astype(np.float64)) / (2.0 * n)
    return out


def apply_tail_log(p: FloatArray, n_sample: int) -> FloatArray:
    eps = 1.0 / (2.0 * n_sample)
    denom = -np.log10(eps)
    clipped = np.clip(1.0 - p, eps, 1.0)
    return np.asarray(-np.log10(clipped) / denom, dtype=np.float64)


def transform_continuous(
    raw: FloatArray,
    sample: FloatArray,
    transform: TextImageTransform,
) -> FloatArray:
    """Apply transform; NaN/inf in raw stay NaN in output."""
    if transform == TextImageTransform.RAW:
        return raw.astype(np.float64)
    p = midrank_percentile(raw, sample)
    if transform == TextImageTransform.PERCENTILE:
        return p
    n = int(np.sum(np.isfinite(sample)))
    if n == 0:
        n = 1
    return apply_tail_log(p, n)


def median_imputed(sample: FloatArray, transform: TextImageTransform) -> float:
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return 0.5
    if transform == TextImageTransform.RAW:
        return float(np.median(finite))
    p = midrank_percentile(finite, finite)
    if transform == TextImageTransform.PERCENTILE:
        return float(np.median(p[np.isfinite(p)]))
    n = finite.size
    t = apply_tail_log(p, n)
    return float(np.median(t[np.isfinite(t)]))
