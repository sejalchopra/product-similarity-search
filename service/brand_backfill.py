"""Infer missing brands from a leading title prefix. Off unless BRAND_BACKFILL=1."""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Any

from service.cleaning import assign_title_groups, normalize_title_for_group, tokenize_title

MIN_BRAND_ROWS = 20
MIN_BRAND_CHARS = 3

_AUDIENCE_WORDS = frozenset(
    {
        "women",
        "woman",
        "womens",
        "men",
        "man",
        "mens",
        "girls",
        "girl",
        "boys",
        "boy",
        "baby",
        "babies",
        "kids",
        "kid",
        "unisex",
        "ladies",
        "lady",
        "female",
        "male",
        "infant",
        "toddler",
        "children",
        "child",
    }
)
_GARMENT_WORDS = frozenset(
    {
        "shirt",
        "tshirt",
        "kurta",
        "dress",
        "jeans",
        "pant",
        "pants",
        "shorts",
        "top",
        "bra",
        "brief",
        "briefs",
        "thong",
        "saree",
        "sari",
        "jacket",
        "shoe",
        "shoes",
        "sock",
        "socks",
        "legging",
        "leggings",
        "skirt",
        "blouse",
        "underwear",
        "boxer",
        "boxers",
        "hoodie",
        "sweater",
        "trouser",
        "trousers",
        "cap",
        "hat",
        "vest",
        "camisole",
        "lingerie",
        "panty",
        "panties",
        "gown",
        "suit",
        "tie",
        "belt",
        "scarf",
        "dupatta",
        "lehenga",
        "nightwear",
        "innerwear",
    }
)
_STOP_WORDS = _AUDIENCE_WORDS | _GARMENT_WORDS


def brand_backfill_enabled() -> bool:
    return os.environ.get("BRAND_BACKFILL", "").strip() in ("1", "true", "yes")


def brand_key(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(tokenize_title(value))


def _rejected_brand(key: str) -> bool:
    if len(key) < MIN_BRAND_CHARS:
        return True
    if key in _STOP_WORDS:
        return True
    tokens = key.split()
    if len(tokens) == 1 and tokens[0] in _STOP_WORDS:
        return True
    return False


def eligible_brand_counts(brands: list[str]) -> dict[str, int]:
    """Brands observed on at least 20 non-missing rows, long enough, not stop words."""
    counts: Counter[str] = Counter()
    for raw in brands:
        key = brand_key(raw)
        if key:
            counts[key] += 1
    return {key: n for key, n in counts.items() if n >= MIN_BRAND_ROWS and not _rejected_brand(key)}


def prefix_collisions(eligible: dict[str, int]) -> list[dict[str, Any]]:
    """Eligible brands where one key is a leading token-prefix of another."""
    keys = sorted(eligible)
    out: list[dict[str, Any]] = []
    for short in keys:
        longer = [other for other in keys if other != short and other.startswith(short + " ")]
        if longer:
            out.append(
                {
                    "brand": short,
                    "also_matched_by": longer,
                    "rows_with_brand": eligible[short],
                }
            )
    return out


def infer_brand(title: str, eligible: dict[str, int]) -> tuple[str | None, list[str]]:
    """Longest leading token prefix that exactly matches an eligible brand.

    The second value lists every eligible prefix of this title (shortest last).
    """
    tokens = tokenize_title(title)
    matches: list[str] = []
    for n in range(1, len(tokens) + 1):
        pref = " ".join(tokens[:n])
        if pref in eligible:
            matches.append(pref)
    if not matches:
        return None, []
    return matches[-1], matches


def apply_brand_backfill(
    titles: list[str],
    brands: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Return updated brand strings and a review report. Does not write artifacts."""
    eligible = eligible_brand_counts(brands)
    inferred_counts: Counter[str] = Counter()
    multi_prefix_rows: list[dict[str, Any]] = []
    review_pool: list[dict[str, str]] = []
    updated = list(brands)
    missing_before = 0
    filled = 0
    for i, (title, raw) in enumerate(zip(titles, brands, strict=True)):
        if brand_key(raw):
            continue
        missing_before += 1
        guessed, matches = infer_brand(title, eligible)
        if guessed is None:
            continue
        updated[i] = guessed
        filled += 1
        inferred_counts[guessed] += 1
        review_pool.append({"title": title, "inferred_brand": guessed})
        if len(matches) > 1:
            multi_prefix_rows.append({"title": title, "prefixes": matches, "chosen": guessed})
    n = max(len(brands), 1)
    missing_after = sum(1 for b in updated if not brand_key(b))
    rng = random.Random(0)
    sample = review_pool if len(review_pool) <= 50 else rng.sample(review_pool, 50)
    report: dict[str, Any] = {
        "enabled_by_env": brand_backfill_enabled(),
        "rows": len(brands),
        "rows_backfilled": filled,
        "missing_brand_rate_before": missing_before / n,
        "missing_brand_rate_after": missing_after / n,
        "eligible_brand_count": len(eligible),
        "top_inferred_brands": [
            {"brand": b, "rows": c} for b, c in inferred_counts.most_common(30)
        ],
        "review_sample": sample,
        "brands_matched_by_multiple_prefix_rules": prefix_collisions(eligible),
        "rows_with_multiple_prefix_matches": len(multi_prefix_rows),
        "multiple_prefix_examples": multi_prefix_rows[:20],
    }
    return updated, report


def _codes_after_backfill(
    brands_before: list[str],
    brands_after: list[str],
    codes_before: list[int],
) -> list[int]:
    key_to_code: dict[str, int] = {}
    for raw, code in zip(brands_before, codes_before, strict=True):
        key = brand_key(raw)
        if key and code >= 0 and key not in key_to_code:
            key_to_code[key] = int(code)
    next_code = (max(codes_before) + 1) if codes_before else 0
    out: list[int] = []
    for raw_before, raw_after, code_before in zip(
        brands_before, brands_after, codes_before, strict=True
    ):
        if brand_key(raw_before):
            out.append(int(code_before))
            continue
        key = brand_key(raw_after)
        if not key:
            out.append(-1)
            continue
        if key not in key_to_code:
            key_to_code[key] = next_code
            next_code += 1
        out.append(key_to_code[key])
    return out


def title_group_effect(
    titles: list[str],
    brands_before: list[str],
    brands_after: list[str],
    *,
    groups_before: list[int] | None = None,
    brand_codes_before: list[int] | None = None,
    normalized_titles: list[str] | None = None,
    watch_groups: tuple[int, ...] = (238, 202, 284),
) -> dict[str, Any]:
    """Compare title groups. Current catalog ids are kept as the before labels."""
    norms = normalized_titles or [normalize_title_for_group(t) for t in titles]
    if brand_codes_before is None:
        brand_codes_before = []
        mapping: dict[str, int] = {}
        for raw in brands_before:
            key = brand_key(raw)
            if not key:
                brand_codes_before.append(-1)
            else:
                if key not in mapping:
                    mapping[key] = len(mapping)
                brand_codes_before.append(mapping[key])
    before = (
        groups_before
        if groups_before is not None
        else assign_title_groups(norms, brand_codes_before)
    )
    after = assign_title_groups(
        norms, _codes_after_backfill(brands_before, brands_after, brand_codes_before)
    )

    def _largest(groups: list[int], k: int = 10) -> list[dict[str, int]]:
        counts = Counter(groups)
        return [{"title_group": g, "row_count": n} for g, n in counts.most_common(k)]

    watched: list[dict[str, Any]] = []
    for gid in watch_groups:
        rows = [i for i, g in enumerate(before) if g == gid]
        landed = Counter(after[i] for i in rows)
        after_sizes = Counter(after)
        watched.append(
            {
                "title_group_before": gid,
                "rows_before": len(rows),
                "sample_titles": [titles[i] for i in rows[:3]],
                "sample_brands_before": [brands_before[i] for i in rows[:3]],
                "sample_brands_after": [brands_after[i] for i in rows[:3]],
                "lands_in": [
                    {
                        "title_group_after": g,
                        "rows_from_this_group": n,
                        "destination_group_rows": after_sizes[g],
                    }
                    for g, n in landed.most_common(5)
                ],
            }
        )
    return {
        "unique_groups_before": len(set(before)),
        "unique_groups_after": len(set(after)),
        "largest_before": _largest(before),
        "largest_after": _largest(after),
        "watched_groups": watched,
    }
