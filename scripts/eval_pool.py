"""Candidate pools for offline tuning and pooled eval."""

from __future__ import annotations

import numpy as np

from service.catalog import Catalog


def rows_for_title_groups(catalog: Catalog, groups: set[int]) -> np.ndarray:
    if not groups:
        return np.array([], dtype=np.intp)
    tg = catalog.title_group
    out: list[int] = []
    for g in groups:
        out.extend(int(r) for r in np.where(tg == g)[0])
    return np.unique(np.array(out, dtype=np.intp))


def eval_candidate_pool(
    catalog: Catalog,
    qrow: int,
    relevant_groups: set[int] | None,
    text_cap: int = 3000,
    list_cap: int = 500,
) -> np.ndarray:
    """Top text ∪ same-brand ∪ same-child-category ∪ optional relevant groups (train only)."""
    scores = catalog.text @ catalog.text[qrow]
    k = min(text_cap, catalog.n - 1)
    top = np.argpartition(-scores, k)[:k]

    parts: list[np.ndarray] = [top.astype(np.intp)]
    qb = int(catalog.brand_code[qrow])
    if qb >= 0:
        brand_rows = np.where(catalog.brand_code == qb)[0]
        if brand_rows.size > list_cap:
            sub = scores[brand_rows]
            pick = np.argpartition(-sub, list_cap)[:list_cap]
            brand_rows = brand_rows[pick]
        parts.append(brand_rows.astype(np.intp))

    qc = int(catalog.category_code[qrow])
    if qc >= 0:
        cat_rows = np.where(catalog.category_code == qc)[0]
        if cat_rows.size > list_cap:
            sub = scores[cat_rows]
            pick = np.argpartition(-sub, list_cap)[:list_cap]
            cat_rows = cat_rows[pick]
        parts.append(cat_rows.astype(np.intp))

    if relevant_groups:
        parts.append(rows_for_title_groups(catalog, relevant_groups))
    return np.unique(np.concatenate(parts)).astype(np.intp)


def tune_candidate_pool(
    catalog: Catalog,
    qrow: int,
    relevant_groups: set[int],
    text_cap: int = 3000,
    list_cap: int = 500,
) -> np.ndarray:
    """Train split: inject co-purchase relevant title groups into the pool."""
    return eval_candidate_pool(catalog, qrow, relevant_groups, text_cap, list_cap)


def relevant_groups_in_pool(pool: np.ndarray, catalog: Catalog, relevant: set[int]) -> float:
    if not relevant:
        return 1.0
    pool_groups = {int(catalog.title_group[r]) for r in pool}
    hit = sum(1 for g in relevant if g in pool_groups)
    return hit / len(relevant)
