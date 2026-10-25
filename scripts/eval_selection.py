"""Validation-only model selection (must not see test metrics)."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.eval_bootstrap import paired_bootstrap_ci

SIMPLICITY = {"binary": 0, "A": 1, "B": 1, "A+B": 2}
DEFAULT_TRANSFORM = "tail_log"


def select_transform_on_validation(
    val_ndcg_per_query: dict[str, np.ndarray],
    group_ids: np.ndarray,
    *,
    default: str = DEFAULT_TRANSFORM,
    n_boot: int = 2000,
) -> tuple[str, str, dict[str, Any]]:
    _assert_no_test_keys(val_ndcg_per_query)
    means = {k: float(np.nanmean(v)) for k, v in val_ndcg_per_query.items()}
    meta: dict[str, Any] = {"validation_means": means, "paired_vs_default": {}}
    shipped = default
    reason = f"kept default transform {default}"
    for challenger in sorted(means, key=lambda k: means[k], reverse=True):
        if challenger == default or challenger not in val_ndcg_per_query:
            continue
        paired = paired_bootstrap_ci(
            val_ndcg_per_query[challenger],
            val_ndcg_per_query[default],
            group_ids,
            n_boot=n_boot,
        )
        meta["paired_vs_default"][challenger] = paired
        if paired["lo"] > 0:
            shipped = challenger
            reason = f"paired bootstrap vs {default} excludes 0 ({challenger})"
            break
    if shipped == default and means.get(default, 0.0) == max(means.values()):
        reason = f"default {default} tied or led validation mean NDCG"
    return shipped, reason, meta


def select_variant_on_validation(
    val_ndcg_per_query: dict[str, np.ndarray],
    group_ids: np.ndarray,
    *,
    n_boot: int = 2000,
) -> tuple[str, str, dict[str, Any]]:
    _assert_no_test_keys(val_ndcg_per_query)
    means = {k: float(np.nanmean(v)) for k, v in val_ndcg_per_query.items()}
    winner = max(means, key=lambda k: (means[k], -SIMPLICITY.get(k, 9)))
    rival = "binary"
    if winner == rival:
        return (
            winner,
            "only binary variant available or binary won outright",
            {"validation_means": means},
        )
    if rival not in val_ndcg_per_query:
        return winner, f"validation leader {winner}", {"validation_means": means}
    paired = paired_bootstrap_ci(
        val_ndcg_per_query[winner],
        val_ndcg_per_query[rival],
        group_ids,
        n_boot=n_boot,
    )
    meta: dict[str, Any] = {"paired_ndcg_diff": paired, "validation_means": means}
    if paired["lo"] > 0 and means[winner] >= means[rival]:
        return winner, f"paired bootstrap CI excludes 0 ({winner} vs {rival})", meta
    return rival, f"kept simpler {rival} (no significant gain for {winner})", meta


def _assert_no_test_keys(metrics: dict[str, Any]) -> None:
    for k in metrics:
        if "test" in k.lower():
            raise ValueError("selection must not receive test metrics")
