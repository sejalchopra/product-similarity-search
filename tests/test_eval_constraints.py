"""Weight search constraints and tuned_weights normalization."""

from __future__ import annotations

import numpy as np

from scripts.eval_guardrails import guardrails_pass
from scripts.eval_tune import BLOCK_NAMES, _apply_weight_constraints
from service.scoring import DEFAULT_WEIGHTS, FusionWeights


def test_weight_constraints_cap_text_and_block() -> None:
    start = np.array([DEFAULT_WEIGHTS[b] for b in BLOCK_NAMES], dtype=np.float64)
    huge = start.copy()
    huge[0] = 10.0
    assert _apply_weight_constraints(huge, start, True) is None
    ok = _apply_weight_constraints(start, start, True)
    assert ok is not None


def test_guardrails_reject_large_drop() -> None:
    default = {"same_audience_share10": 0.5, "same_child_share10": 0.5, "same_family_share10": 0.5}
    bad = {"same_audience_share10": 0.4, "same_child_share10": 0.5, "same_family_share10": 0.5}
    ok, reason = guardrails_pass(default, bad)
    assert not ok
    assert reason is not None


def test_normalized_weights_sum_to_one() -> None:
    fw = FusionWeights(
        w={
            "text": 2.0,
            "image": 0.0,
            "category": 0.2,
            "audience": 0.1,
            "brand": 0.1,
            "color": 0.1,
            "price": 0.1,
            "rating": 0.05,
            "weight": 0.05,
        },
        beta=1.0,
    )
    n = fw.normalized()
    assert abs(sum(n.w.values()) - 1.0) < 1e-6
