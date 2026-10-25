"""Pool vs production scoring fidelity diagnostics."""

from __future__ import annotations

import random

import numpy as np

from scripts.eval_pooled import top10_groups_exact_on_pool, top10_groups_on_pool
from service.catalog import Catalog
from service.config import Settings
from service.scoring import FusionWeights, fuse_scores
from service.scoring_profile import ScoringProfile
from service.similarity import find_similar_products


def audit_pool_fidelity(
    catalog: Catalog,
    settings: Settings,
    rows: list[int],
    fw: FusionWeights,
    profile: ScoringProfile,
    store_by_row: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    n_audit: int = 20,
    seed: int = 0,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    pick = rows.copy()
    rng.shuffle(pick)
    pick = pick[: min(n_audit, len(pick))]
    max_score_diff = 0.0
    rank_mismatches = 0
    matvec_top10_match = 0
    pool_vs_full_match = 0
    for qrow in pick:
        pool, blocks = store_by_row[qrow]
        mat_scores = fuse_scores(blocks.astype(np.float64), fw)
        prod_scores = catalog.scores_for_query(qrow, fw, pool, profile=profile)
        diff = float(np.max(np.abs(mat_scores - prod_scores)))
        max_score_diff = max(max_score_diff, diff)
        order_m = np.argsort(-mat_scores)
        order_p = np.argsort(-prod_scores)
        if not np.array_equal(order_m, order_p):
            rank_mismatches += 1
        g_mat = top10_groups_on_pool(catalog, qrow, pool, blocks, fw)
        g_prod = top10_groups_exact_on_pool(catalog, qrow, pool, fw, profile)
        if g_mat == g_prod:
            matvec_top10_match += 1
        full_ids = find_similar_products(
            catalog, catalog.uniq_ids[qrow], 10, settings, weights=fw, profile=profile
        )
        full_groups = [int(catalog.title_group[catalog.row_for(pid)]) for pid in full_ids]
        if g_mat == full_groups:
            pool_vs_full_match += 1
    n = max(len(pick), 1)
    return {
        "audit_queries": len(pick),
        "max_abs_score_diff_on_pool": max_score_diff,
        "rank_order_mismatches_on_pool": rank_mismatches,
        "matvec_equals_exact_on_pool": matvec_top10_match / n,
        "pool_top10_equals_full_catalog_top10": pool_vs_full_match / n,
    }
