"""Vectorized late fusion scoring (no I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from service.transforms import (
    TextImageTransform,
    midrank_percentile,
)

FloatArray = NDArray[np.float64]

AUDIENCE_CLASSES: Final[tuple[str, ...]] = (
    "women",
    "men",
    "girls",
    "boys",
    "baby",
    "kids",
    "unisex",
    "unknown",
)

BLOCK_NAMES: Final[tuple[str, ...]] = (
    "text",
    "image",
    "category",
    "audience",
    "brand",
    "color",
    "price",
    "rating",
    "weight",
)

DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "text": 0.30,
    "image": 0.15,
    "category": 0.15,
    "audience": 0.08,
    "brand": 0.10,
    "color": 0.08,
    "price": 0.08,
    "rating": 0.04,
    "weight": 0.02,
}


def _build_audience_table() -> NDArray[np.float64]:
    n = len(AUDIENCE_CLASSES)
    idx = {c: i for i, c in enumerate(AUDIENCE_CLASSES)}
    t = np.zeros((n, n), dtype=np.float64)
    women_set = {idx["women"], idx["girls"]}
    men_set = {idx["men"], idx["boys"]}
    kids_set = {idx["kids"], idx["girls"], idx["boys"], idx["baby"]}
    for i in range(n):
        for j in range(n):
            if i == j:
                t[i, j] = 1.0
            elif i in women_set and j in women_set:
                t[i, j] = 0.5
            elif i in men_set and j in men_set:
                t[i, j] = 0.5
            elif i in kids_set and j in kids_set:
                t[i, j] = 0.5
            elif idx["unisex"] in (i, j):
                t[i, j] = 0.5
            else:
                t[i, j] = 0.0
    return t


AUDIENCE_TABLE: Final[NDArray[np.float64]] = _build_audience_table()


@dataclass(frozen=True)
class FusionWeights:
    w: dict[str, float]
    beta: float
    text_image_transform: TextImageTransform = TextImageTransform.TAIL_LOG

    def vector(self) -> FloatArray:
        return np.array([self.w[b] for b in BLOCK_NAMES], dtype=np.float64)

    def normalized(self) -> FusionWeights:
        v = self.vector()
        s = v.sum()
        if s <= 0:
            return self
        w = {b: float(self.w[b] / s) for b in BLOCK_NAMES}
        return FusionWeights(w=w, beta=self.beta, text_image_transform=self.text_image_transform)


@dataclass
class ScoreBreakdown:
    blocks: dict[str, float]
    imputed: dict[str, bool]
    raw: dict[str, float | None]
    score: float


def audience_expected_impute(
    observed_code: int,
    audience_freq: FloatArray,
    audience_table: NDArray[np.float64],
    both_missing_sum: float,
) -> float:
    if observed_code < 0:
        return both_missing_sum
    pi = audience_freq
    return float(np.dot(pi, audience_table[observed_code, :]))


def price_similarity(p1: float, p2: float) -> float:
    if p1 <= 0 or p2 <= 0:
        return 0.0
    return float(np.exp(-abs(np.log(p1) - np.log(p2)) / 0.5))


def rating_similarity(r1: float, r2: float) -> float:
    return float(max(0.0, 1.0 - abs(r1 - r2) / 4.0))


def weight_similarity(w1: float, w2: float) -> float:
    if w1 <= 0 or w2 <= 0:
        return 0.0
    return float(np.exp(-abs(w1 - w2) / max(w1, w2, 1.0)))


def shrink_rating(r: float, n: int, catalog_mean: float) -> float:
    if n <= 0:
        return catalog_mean
    return (n / (n + 5.0)) * r + (5.0 / (n + 5.0)) * catalog_mean


def fuse_scores(
    block_matrix: FloatArray,
    weights: FusionWeights,
) -> FloatArray:
    """block_matrix shape (n_candidates, n_blocks) in [0,1] or imputed."""
    w = weights.normalized().vector()
    return block_matrix @ w


@dataclass
class RankDedupResult:
    picked: list[int]
    duplicates_collapsed: int


def rank_with_dedup_detailed(
    scores: FloatArray,
    title_groups: NDArray[Any],
    query_group: int,
    exclude_indices: NDArray[np.bool_] | None,
    num_similar: int,
    rating: FloatArray,
    price: FloatArray,
    uniq_ids: list[str],
) -> RankDedupResult:
    n = scores.shape[0]
    mask = np.ones(n, dtype=bool)
    if exclude_indices is not None:
        mask &= ~exclude_indices
    if query_group >= 0:
        mask &= title_groups != query_group
    order = np.argsort(-scores)
    picked: list[int] = []
    seen_groups: set[int] = set()
    duplicates_collapsed = 0
    for idx in order:
        if not mask[idx]:
            continue
        if len(picked) >= num_similar:
            break
        g = int(title_groups[idx])
        if g >= 0 and g in seen_groups:
            duplicates_collapsed += 1
            continue
        picked.append(int(idx))
        if g >= 0:
            seen_groups.add(g)
    if len(picked) < num_similar:
        for idx in order:
            if not mask[idx] or idx in picked:
                continue
            if len(picked) >= num_similar:
                break
            picked.append(int(idx))
    picked.sort(
        key=lambda i: (
            -round(float(scores[i]), 6),
            -float(rating[i]) if np.isfinite(rating[i]) else -1.0,
            float(price[i]) if np.isfinite(price[i]) else float("inf"),
            uniq_ids[i],
        )
    )
    return RankDedupResult(
        picked=picked[:num_similar],
        duplicates_collapsed=duplicates_collapsed,
    )


def rank_with_dedup(
    scores: FloatArray,
    title_groups: NDArray[Any],
    query_group: int,
    exclude_indices: NDArray[np.bool_] | None,
    num_similar: int,
    rating: FloatArray,
    price: FloatArray,
    uniq_ids: list[str],
) -> list[int]:
    return rank_with_dedup_detailed(
        scores,
        title_groups,
        query_group,
        exclude_indices,
        num_similar,
        rating,
        price,
        uniq_ids,
    ).picked


def percentile_block(
    raw: FloatArray,
    sample: FloatArray,
    missing_mask: FloatArray,
    mu: float,
) -> tuple[FloatArray, FloatArray]:
    """Returns (values, imputed_mask)."""
    p = midrank_percentile(raw, sample)
    out = np.where(missing_mask, mu, p)
    imputed = missing_mask.copy()
    return out.astype(np.float64), imputed
