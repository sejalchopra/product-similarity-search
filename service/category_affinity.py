"""Precomputed category affinity tables for fast block_batch."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from service.transforms import TextImageTransform, transform_continuous

FloatArray = NDArray[np.float64]


def build_centroid_cosine_matrix(centroids: FloatArray) -> FloatArray:
    """Shape (n_categories, n_categories), unit-norm centroid cosines."""
    return centroids @ centroids.T


def build_affinity_tables(
    centroids: FloatArray,
    cos_sample: FloatArray,
) -> dict[str, FloatArray]:
    """Calibrated category affinity per text/image transform. Diagonal = 1.0."""
    cos = build_centroid_cosine_matrix(centroids)
    n = cos.shape[0]
    out: dict[str, FloatArray] = {}
    for ti in TextImageTransform:
        table = np.full((n, n), np.nan, dtype=np.float64)
        for i in range(n):
            table[i, i] = 1.0
        off = ~np.eye(n, dtype=bool)
        raw = cos[off]
        cal = transform_continuous(raw, cos_sample, ti)
        table[off] = cal
        out[ti.value] = table
    return out
