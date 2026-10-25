"""find_similar_products and explain."""

from __future__ import annotations

import numpy as np

from service.catalog import Catalog
from service.config import Settings
from service.scoring import (
    BLOCK_NAMES,
    DEFAULT_WEIGHTS,
    FusionWeights,
    fuse_scores,
    rank_with_dedup,
)
from service.scoring_profile import ScoringProfile


def _default_weights(settings: Settings) -> FusionWeights:
    return FusionWeights(
        w=dict(DEFAULT_WEIGHTS),
        beta=1.0,
        text_image_transform=settings.text_image_transform,
    )


def find_similar_products(
    catalog: Catalog,
    product_id: str,
    num_similar: int,
    settings: Settings,
    weights: FusionWeights | None = None,
    profile: ScoringProfile | None = None,
) -> list[str]:
    w = weights or _default_weights(settings)
    prof = profile or catalog.active_profile
    q = catalog.row_for(product_id)
    if settings.is_ann:
        cand = catalog.ann_candidate_rows(q, settings)
        scores = catalog.scores_for_query(q, w, cand, profile=prof)
        row_map = cand
    else:
        scores = catalog.scores_for_query(q, w, None, profile=prof)
        row_map = np.arange(catalog.n, dtype=np.intp)

    exclude = row_map == q
    picked_local = rank_with_dedup(
        scores,
        catalog.title_group[row_map],
        int(catalog.title_group[q]),
        exclude,
        num_similar,
        catalog.rating[row_map],
        catalog.price[row_map],
        [catalog.uniq_ids[int(row_map[i])] for i in range(len(row_map))],
    )
    if settings.is_ann:
        return [catalog.uniq_ids[int(row_map[i])] for i in picked_local]
    return [catalog.uniq_ids[i] for i in picked_local]


def explain_one(
    catalog: Catalog,
    product_id: str,
    neighbor_id: str,
    settings: Settings,
    weights: FusionWeights | None = None,
) -> dict[str, object]:
    w = weights or _default_weights(settings)
    q = catalog.row_for(product_id)
    n = catalog.row_for(neighbor_id)
    batch = catalog.block_details(q, np.array([n], dtype=np.intp), w)
    mat = batch.values
    score = float(fuse_scores(mat, w)[0])
    blocks: dict[str, float] = {}
    raw_map: dict[str, float | None] = {}
    imputed: dict[str, bool] = {}
    for i, name in enumerate(BLOCK_NAMES):
        blocks[name] = float(mat[0, i])
        imputed[name] = bool(batch.imputed[0, i])
        raw_arr = batch.raw_sim.get(name)
        if raw_arr is not None and np.isfinite(raw_arr[0]):
            raw_map[name] = float(raw_arr[0])
        else:
            raw_map[name] = None
    raw_map["text"] = float(catalog.text[n] @ catalog.text[q])
    return {
        "product_id": neighbor_id,
        "product_name": catalog.product_names[n],
        "score": score,
        "blocks": blocks,
        "raw": raw_map,
        "imputed": imputed,
        "transform": w.text_image_transform.value,
    }
