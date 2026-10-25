"""Fast eval helpers: pool ranking fidelity, bootstrap, selection guard."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.eval.bootstrap import cluster_bootstrap_ci, paired_bootstrap_ci
from scripts.eval.pooled import top10_groups_exact_on_pool, top10_groups_on_pool
from scripts.eval.selection import select_variant_on_validation
from service.scoring import DEFAULT_WEIGHTS, FusionWeights
from service.scoring_profile import ScoringProfile


def test_pooled_matvec_matches_exact_on_pool() -> None:
    from tests.conftest import make_synthetic_catalog

    catalog = make_synthetic_catalog()
    rng = np.random.default_rng(0)
    rows = rng.choice(catalog.n, size=20, replace=False).tolist()
    fw = FusionWeights(w=dict(DEFAULT_WEIGHTS), beta=1.0)
    prof = ScoringProfile()
    for qrow in rows:
        scores = catalog.text @ catalog.text[qrow]
        k = min(500, catalog.n - 1)
        pool = np.argpartition(-scores, k)[:k].astype(np.intp)
        blocks = catalog.block_matrix(qrow, pool, fw, profile=prof)
        a = top10_groups_on_pool(catalog, qrow, pool, blocks, fw)
        b = top10_groups_exact_on_pool(catalog, qrow, pool, fw, prof)
        assert a == b


def test_bootstrap_ci_width_and_paired_zero() -> None:
    rng = np.random.default_rng(0)
    n_groups = 40
    per_group = 5
    values = []
    gids = []
    for g in range(n_groups):
        chunk = rng.normal(loc=g * 0.01, scale=0.02, size=per_group)
        values.extend(chunk.tolist())
        gids.extend([g] * per_group)
    arr = np.array(values, dtype=np.float64)
    gid = np.array(gids, dtype=np.int64)
    wide_lo, wide_hi = 0.0, 0.0
    for seed in range(20):
        out = cluster_bootstrap_ci(arr, gid, n_boot=500, seed=seed)
        w = out["hi"] - out["lo"]
        if w > wide_hi - wide_lo:
            wide_lo, wide_hi = out["lo"], out["hi"]
    narrow = cluster_bootstrap_ci(arr, gid, n_boot=5000, seed=99)
    assert (narrow["hi"] - narrow["lo"]) <= (wide_hi - wide_lo) * 1.05
    same = paired_bootstrap_ci(arr, arr, gid, n_boot=200, seed=2)
    assert same["mean"] == 0.0
    assert same["lo"] <= 0.0 <= same["hi"]


def test_selection_rejects_test_metrics() -> None:
    with pytest.raises(ValueError, match="test metrics"):
        select_variant_on_validation({"test_leak": np.array([1.0])}, np.array([0]))
