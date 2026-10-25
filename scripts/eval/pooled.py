"""Pooled block caches and production-aligned ranking on a fixed candidate pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scripts.eval.pool import eval_candidate_pool, relevant_groups_in_pool
from scripts.eval.progress import StageTimer
from service.catalog import Catalog
from service.scoring import DEFAULT_WEIGHTS, FusionWeights, fuse_scores, rank_with_dedup
from service.scoring_profile import ScoringProfile
from service.transforms import TextImageTransform


def profile_key(profile: ScoringProfile) -> str:
    return f"{profile.category_mode}:{profile.text_matrix}"


def top10_groups_on_pool(
    catalog: Catalog,
    qrow: int,
    pool: np.ndarray,
    blocks: np.ndarray,
    weights: FusionWeights,
) -> list[int]:
    """Matvec on cached blocks + production rank_with_dedup (pool-local indices)."""
    scores = fuse_scores(blocks.astype(np.float64), weights)
    exclude = pool == qrow
    picked = rank_with_dedup(
        scores,
        catalog.title_group[pool],
        int(catalog.title_group[qrow]),
        exclude,
        10,
        catalog.rating[pool],
        catalog.price[pool],
        [catalog.uniq_ids[int(r)] for r in pool],
    )
    return [int(catalog.title_group[pool[i]]) for i in picked]


def top10_groups_exact_on_pool(
    catalog: Catalog,
    qrow: int,
    pool: np.ndarray,
    weights: FusionWeights,
    profile: ScoringProfile,
) -> list[int]:
    """Full block_batch on pool only — reference for fidelity tests."""
    scores = catalog.scores_for_query(qrow, weights, pool, profile=profile)
    exclude = pool == qrow
    picked = rank_with_dedup(
        scores,
        catalog.title_group[pool],
        int(catalog.title_group[qrow]),
        exclude,
        10,
        catalog.rating[pool],
        catalog.price[pool],
        [catalog.uniq_ids[int(r)] for r in pool],
    )
    return [int(catalog.title_group[pool[i]]) for i in picked]


@dataclass
class QueryPoolEntry:
    pool: np.ndarray
    blocks: np.ndarray  # float16 (m, 9)
    relevant_in_pool: float | None = None


@dataclass
class PooledQueryStore:
    """Per-query pools and block matrices for one (transform, profile, beta)."""

    catalog: Catalog
    by_row: dict[int, QueryPoolEntry] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        catalog: Catalog,
        query_rows: list[int],
        relevants: dict[int, set[int]],
        transform: TextImageTransform,
        profile: ScoringProfile,
        beta: float,
        *,
        inject_relevant_groups: bool,
        timer: StageTimer | None = None,
    ) -> PooledQueryStore:
        store = cls(catalog=catalog)
        fw = FusionWeights(
            w=dict(DEFAULT_WEIGHTS),
            beta=beta,
            text_image_transform=transform,
        )
        n = len(query_rows)
        stage = timer or StageTimer("pool+blocks")
        stage.start(n)
        for qi, qrow in enumerate(query_rows):
            rel_g = relevants.get(qrow, set())
            pool = eval_candidate_pool(
                catalog,
                qrow,
                rel_g if inject_relevant_groups else None,
            )
            mat = catalog.block_matrix(qrow, pool, fw, profile=profile)
            rip: float | None = None
            if rel_g and not inject_relevant_groups:
                rip = relevant_groups_in_pool(pool, catalog, rel_g)
            store.by_row[qrow] = QueryPoolEntry(
                pool=pool,
                blocks=mat.astype(np.float32),
                relevant_in_pool=rip,
            )
            stage.tick(qi + 1)
        stage.finish()
        return store

    def mean_ndcg_hit(
        self,
        query_rows: list[int],
        relevants: dict[int, set[int]],
        weights: FusionWeights,
        ndcg_fn: Any,
        hit_fn: Any,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        from scripts.eval.tune import hit_at_k, ndcg_at_k

        ndcgs: list[float] = []
        hits: list[float] = []
        ndcg_arr = np.zeros(len(query_rows), dtype=np.float64)
        hit_arr = np.zeros(len(query_rows), dtype=np.float64)
        for i, qrow in enumerate(query_rows):
            rel = relevants.get(qrow, set())
            if not rel:
                continue
            ent = self.by_row[qrow]
            groups = top10_groups_on_pool(self.catalog, qrow, ent.pool, ent.blocks, weights)
            n = ndcg_at_k(groups, rel, 10)
            h = hit_at_k(groups, rel, 10)
            ndcg_arr[i] = n
            hit_arr[i] = h
            ndcgs.append(n)
            hits.append(h)
        if not ndcgs:
            return 0.0, 0.0, ndcg_arr, hit_arr
        return float(np.mean(ndcgs)), float(np.mean(hits)), ndcg_arr, hit_arr
