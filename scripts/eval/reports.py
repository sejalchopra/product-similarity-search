"""Eval report tables: title groups, Blazze comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from service.catalog import Catalog
from service.config import Settings
from service.explore import explore_similar
from service.scoring import DEFAULT_WEIGHTS, FusionWeights
from service.scoring_profile import ScoringProfile


def top_title_groups(catalog: Catalog, k: int = 10) -> list[dict[str, Any]]:
    by_g: dict[int, list[int]] = defaultdict(list)
    for i in range(catalog.n):
        by_g[int(catalog.title_group[i])].append(i)
    ranked = sorted(by_g.items(), key=lambda x: len(x[1]), reverse=True)[:k]
    out: list[dict[str, Any]] = []
    for g, rows in ranked:
        sample = rows[:3]
        out.append(
            {
                "title_group": g,
                "row_count": len(rows),
                "sample_titles": [catalog.product_names[r] for r in sample],
                "sample_brands": [catalog.brands[r] for r in sample],
            }
        )
    return out


def _explore_top10(
    catalog: Catalog,
    settings: Settings,
    qrow: int,
    fw: FusionWeights,
    profile: ScoringProfile,
) -> list[dict[str, Any]]:
    catalog.active_profile = profile
    data = explore_similar(catalog, catalog.uniq_ids[qrow], 10, settings, fw)
    rows: list[dict[str, Any]] = []
    for r in data["results"]:
        rows.append(
            {
                "rank": r["rank"],
                "name": r["name"],
                "audience": r["audience"],
                "child_category": r["child_category"],
                "score": round(float(r["score"]), 4),
            }
        )
    return rows


def blazze_comparison_table(
    catalog: Catalog,
    settings: Settings,
    *,
    default_fw: FusionWeights,
    legacy_smoke_fw: FusionWeights | None,
    selected_fw: FusionWeights,
    selected_variant: str,
    variant_a_fw: FusionWeights | None = None,
) -> dict[str, Any]:
    from service.config import DEMO_QUERY_PRODUCT_ID

    qrow = catalog.id_to_row.get(DEMO_QUERY_PRODUCT_ID)
    if qrow is None:
        return {"error": "query not found"}
    prof = ScoringProfile.from_variant(selected_variant)
    binary = ScoringProfile()
    out: dict[str, Any] = {
        "query_id": catalog.uniq_ids[qrow],
        "query_name": catalog.product_names[qrow],
        "default_weights": _explore_top10(catalog, settings, qrow, default_fw, binary),
    }
    if legacy_smoke_fw is not None:
        out["legacy_smoke_weights"] = _explore_top10(
            catalog, settings, qrow, legacy_smoke_fw, binary
        )
    out["selected_weights"] = _explore_top10(catalog, settings, qrow, selected_fw, prof)
    if variant_a_fw is not None:
        out["variant_a"] = _explore_top10(
            catalog,
            settings,
            qrow,
            variant_a_fw,
            ScoringProfile(category_mode="affinity"),
        )
    catalog.active_profile = ScoringProfile()
    return out


def default_fusion_weights(settings: Settings) -> FusionWeights:
    return FusionWeights(
        w=dict(DEFAULT_WEIGHTS), beta=1.0, text_image_transform=settings.text_image_transform
    )
