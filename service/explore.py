"""Exact-mode explore payload for the demo UI."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from service.catalog import Catalog
from service.cleaning import humanize_category_key
from service.config import Settings
from service.query_overrides import QueryOverrides
from service.scoring import BLOCK_NAMES, FusionWeights, fuse_scores, rank_with_dedup_detailed


def _query_card(catalog: Catalog, row: int) -> dict[str, Any]:
    p = float(catalog.price[row])
    r = float(catalog.rating[row])
    cat_raw = catalog.categories[row] or ""
    return {
        "id": catalog.uniq_ids[row],
        "name": catalog.product_names[row],
        "brand": catalog.brands[row] or "",
        "price": float(p) if np.isfinite(p) else None,
        "image_url": catalog.image_urls[row] or None,
        "child_category": humanize_category_key(cat_raw or None),
        "child_category_key": cat_raw,
        "audience": catalog.audiences[row] or "",
        "rating": float(r) if int(catalog.rating_n[row]) > 0 and np.isfinite(r) else None,
        "weight_g": float(catalog.weight_g[row]) if np.isfinite(catalog.weight_g[row]) else None,
        "color_missing": bool(catalog.color_missing[row]),
    }


def explore_similar(
    catalog: Catalog,
    product_id: str,
    n: int,
    settings: Settings,
    weights: FusionWeights,
    overrides: QueryOverrides | None = None,
) -> dict[str, Any]:
    ov = overrides or QueryOverrides()
    q = catalog.row_for(product_id)
    t0 = time.perf_counter()

    scores = catalog.scores_for_query(q, weights, None, overrides=ov)
    row_map = np.arange(catalog.n, dtype=np.intp)
    exclude = row_map == q
    qg = int(catalog.title_group[q])
    dedup = rank_with_dedup_detailed(
        scores,
        catalog.title_group[row_map],
        qg,
        exclude,
        n,
        catalog.rating[row_map],
        catalog.price[row_map],
        [catalog.uniq_ids[int(i)] for i in row_map],
    )
    picked = dedup.picked
    picked_rows = np.array(picked, dtype=np.intp)
    batch = catalog.block_details(q, picked_rows, weights, overrides=ov)
    fused = fuse_scores(batch.values, weights)
    w_norm = weights.normalized().w

    hidden_ids = sorted(
        catalog.uniq_ids[i]
        for i in range(catalog.n)
        if int(catalog.title_group[i]) == qg and i != q
    )
    hidden_count = len(hidden_ids)

    results: list[dict[str, Any]] = []
    for rank, row in enumerate(picked, start=1):
        row = int(row)
        blocks: dict[str, Any] = {}
        li = rank - 1
        for bi, name in enumerate(BLOCK_NAMES):
            val = float(batch.values[li, bi])
            wt = float(w_norm[name])
            raw_arr = batch.raw_sim.get(name)
            raw_v: float | None = None
            if raw_arr is not None:
                rv = float(raw_arr[li])
                if np.isfinite(rv):
                    raw_v = rv
            blocks[name] = {
                "value": val,
                "raw_sim": raw_v,
                "weight": wt,
                "contribution": wt * val,
                "imputed": bool(batch.imputed[li, bi]),
                "imputed_reason": batch.imputed_reason[li][bi],
            }
        card = _query_card(catalog, row)
        results.append(
            {
                **card,
                "score": float(fused[li]),
                "rank": rank,
                "blocks": blocks,
            }
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    meta = {
        "candidates_scored": catalog.n,
        "title_groups_total": int(len(np.unique(catalog.title_group))),
        "hidden_query_group_count": hidden_count,
        "hidden_query_group_ids": hidden_ids[:20],
        "duplicates_collapsed": dedup.duplicates_collapsed,
        "mode": "exact",
        "transform": weights.text_image_transform.value,
        "weights": {k: float(w_norm[k]) for k in w_norm},
        "latency_ms": latency_ms,
        "cache_hit": False,
        "currency_symbol": settings.currency_symbol,
    }
    return {"query": _query_card(catalog, q), "results": results, "meta": meta}
