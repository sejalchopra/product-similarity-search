"""Eval run mode env wiring (no scoring changes)."""

from __future__ import annotations

import os

import pytest

from scripts.evaluate import _eval_run_mode


@pytest.mark.parametrize(
    ("env", "smoke", "capped", "caps"),
    [
        (
            {"EVAL_MEDIUM": "1"},
            False,
            True,
            (1500, 1500, 2000),
        ),
        (
            {
                "EVAL_MEDIUM": "1",
                "EVAL_TUNE_QUERIES": "100",
                "EVAL_VAL_QUERIES": "50",
                "EVAL_TEST_QUERIES": "75",
            },
            False,
            True,
            (100, 50, 75),
        ),
        (
            {"EVAL_SMOKE": "1"},
            True,
            True,
            (400, 200, 200),
        ),
    ],
)
def test_eval_run_mode(
    env: dict[str, str],
    smoke: bool,
    capped: bool,
    caps: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "EVAL_MEDIUM",
        "EVAL_SMOKE",
        "EVAL_MAX_QUERIES",
        "EVAL_TUNE_QUERIES",
        "EVAL_VAL_QUERIES",
        "EVAL_TEST_QUERIES",
        "EVAL_MAX_TUNE",
        "EVAL_MAX_VAL",
        "EVAL_MAX_TEST",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    got_smoke, got_capped, t, v, te = _eval_run_mode()
    assert got_smoke is smoke
    assert got_capped is capped
    assert (t, v, te) == caps
