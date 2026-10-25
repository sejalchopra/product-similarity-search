"""Cluster bootstrap on precomputed per-query metrics."""

from __future__ import annotations

import numpy as np

N_BOOT_DEFAULT = 2000


def cluster_bootstrap_ci(
    values: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = 0,
) -> dict[str, float]:
    """Resample title groups; values and group_ids aligned per query row."""
    mask = np.isfinite(values)
    if not np.any(mask):
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    vals = values[mask]
    gids = group_ids[mask]
    base_mean = float(np.mean(vals))
    by_group: dict[int, list[float]] = {}
    for v, g in zip(vals, gids, strict=True):
        by_group.setdefault(int(g), []).append(float(v))
    group_list = list(by_group.keys())
    if not group_list:
        return {"mean": base_mean, "lo": base_mean, "hi": base_mean}
    rng = np.random.default_rng(seed)
    n_g = len(group_list)
    means: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(group_list, size=n_g, replace=True)
        sample_vals = [by_group[int(g)] for g in picked]
        flat = [x for sub in sample_vals for x in sub]
        means.append(float(np.mean(flat)))
    arr = np.array(means)
    return {
        "mean": base_mean,
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
    }


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap mean(a) - mean(b) with the same resampled title groups."""
    diff = values_a - values_b
    out = cluster_bootstrap_ci(diff, group_ids, n_boot=n_boot, seed=seed)
    out["mean"] = float(np.mean(diff[np.isfinite(diff)])) if np.any(np.isfinite(diff)) else 0.0
    return out
