"""Brand backfill stays off unless asked, and only accepts exact known prefixes."""

from __future__ import annotations

import pytest

from service.brand_backfill import apply_brand_backfill, brand_backfill_enabled, infer_brand


def test_brand_backfill_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAND_BACKFILL", raising=False)
    assert brand_backfill_enabled() is False


def test_infers_longest_exact_prefix_and_rejects_stop_words() -> None:
    brands = ["Nike"] * 20 + ["Shirt"] * 20 + ["Max"] * 20
    titles = ["Nike Air Running Shoe"] + ["Shirt Cotton"] + ["Something Else"]
    brands_in = [""] * 3
    # pad so eligible counts come from the known rows
    all_titles = titles + ["x"] * len(brands)
    all_brands = brands_in + brands
    updated, report = apply_brand_backfill(all_titles, all_brands)
    assert updated[0] == "nike"
    assert updated[1] == ""  # garment word is not eligible
    assert updated[2] == ""
    assert report["rows_backfilled"] == 1


def test_multiple_prefix_rules_keep_longest() -> None:
    brands = ["Acme"] * 20 + ["Acme Home"] * 20
    eligible_rows_titles = ["Acme"] * 20 + ["Acme Home"] * 20
    query_title = "Acme Home Cotton Kurta"
    updated, report = apply_brand_backfill(
        eligible_rows_titles + [query_title],
        brands + [""],
    )
    assert updated[-1] == "acme home"
    assert report["rows_with_multiple_prefix_matches"] == 1
    guessed, matches = infer_brand(query_title, {"acme": 20, "acme home": 20})
    assert guessed == "acme home"
    assert matches == ["acme", "acme home"]
