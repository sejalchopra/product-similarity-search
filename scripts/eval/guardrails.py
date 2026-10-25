"""Demographic guardrails for tuned weight acceptance."""

from __future__ import annotations

from typing import Any

import numpy as np

from service.catalog import Catalog
from service.cleaning import category_family_key
from service.config import Settings
from service.scoring import FusionWeights
from service.scoring_profile import ScoringProfile
from service.similarity import find_similar_products

GUARDRAIL_MARGIN = 0.03


def _top10_rows(
    catalog: Catalog,
    qrow: int,
    settings: Settings,
    fw: FusionWeights,
    profile: ScoringProfile,
) -> list[int]:
    ids = find_similar_products(
        catalog,
        catalog.uniq_ids[qrow],
        10,
        settings,
        weights=fw,
        profile=profile,
    )
    return [catalog.row_for(pid) for pid in ids]


def _share(rows: list[int], qrow: int, fn: Any) -> float:
    if not rows:
        return 0.0
    hit = sum(1 for r in rows if fn(qrow, r))
    return hit / len(rows)


def guardrail_shares_for_query(
    catalog: Catalog,
    qrow: int,
    settings: Settings,
    fw: FusionWeights,
    profile: ScoringProfile,
) -> dict[str, float]:
    picked = _top10_rows(catalog, qrow, settings, fw, profile)
    qa = int(catalog.audience_code[qrow])
    qc = int(catalog.category_code[qrow])
    qf = category_family_key(catalog.categories[qrow] or None)

    def aud_hit(_q: int, r: int) -> bool:
        return int(catalog.audience_code[r]) == qa and qa >= 0

    def child_hit(_q: int, r: int) -> bool:
        return int(catalog.category_code[r]) == qc and qc >= 0

    def fam_hit(_q: int, r: int) -> bool:
        if not qf:
            return False
        return category_family_key(catalog.categories[r] or None) == qf

    return {
        "same_audience_share10": _share(picked, qrow, aud_hit),
        "same_child_share10": _share(picked, qrow, child_hit),
        "same_family_share10": _share(picked, qrow, fam_hit),
    }


def guardrail_query_arrays(
    catalog: Catalog,
    rows: list[int],
    settings: Settings,
    fw: FusionWeights,
    profile: ScoringProfile,
) -> dict[str, np.ndarray]:
    """Per-query top-10 shares, aligned with `rows` (NaN if top-10 is empty)."""
    keys = ("same_audience_share10", "same_child_share10", "same_family_share10")
    out = {k: np.full(len(rows), np.nan, dtype=np.float64) for k in keys}
    for i, qrow in enumerate(rows):
        shares = guardrail_shares_for_query(catalog, qrow, settings, fw, profile)
        for k in keys:
            out[k][i] = shares[k]
    return out


def mean_guardrails(
    catalog: Catalog,
    rows: list[int],
    settings: Settings,
    fw: FusionWeights,
    profile: ScoringProfile,
) -> dict[str, float]:
    acc = {"same_audience_share10": 0.0, "same_child_share10": 0.0, "same_family_share10": 0.0}
    n = 0
    for qrow in rows:
        rel = guardrail_shares_for_query(catalog, qrow, settings, fw, profile)
        for k in acc:
            acc[k] += rel[k]
        n += 1
    if n == 0:
        return acc
    return {k: v / n for k, v in acc.items()}


def guardrails_pass(default: dict[str, float], tuned: dict[str, float]) -> tuple[bool, str | None]:
    for k in default:
        if tuned[k] < default[k] - GUARDRAIL_MARGIN:
            return False, f"{k} {tuned[k]:.3f} < default {default[k]:.3f} - {GUARDRAIL_MARGIN}"
    return True, None
