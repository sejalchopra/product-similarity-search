"""Text cleaning, parsing, and title-group dedup."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Final

COLOR_ALIASES: Final[dict[str, str]] = {
    "purpel": "purple",
    "blackk": "black",
    "grey": "gray",
}
COLOR_WORDS: Final[frozenset[str]] = frozenset(
    {
        "beige",
        "black",
        "blue",
        "bronze",
        "brown",
        "burgundy",
        "charcoal",
        "coral",
        "cream",
        "cyan",
        "gold",
        "gray",
        "green",
        "ivory",
        "khaki",
        "lavender",
        "magenta",
        "maroon",
        "mint",
        "multi",
        "multicolor",
        "mustard",
        "navy",
        "nude",
        "olive",
        "orange",
        "peach",
        "pink",
        "purple",
        "red",
        "rust",
        "silver",
        "tan",
        "teal",
        "white",
        "wine",
        "yellow",
    }
)
COLOR_MODIFIERS: Final[frozenset[str]] = frozenset(
    {"matte", "neon", "deep", "metallic", "off", "dark", "bright", "pale", "light"}
)
DISPLAY_PREFIX_TO_LABEL: Final[tuple[tuple[str, str], ...]] = (
    ("Womens", "Women's"),
    ("Mens", "Men's"),
    ("Girls", "Girls"),
    ("Boys", "Boys"),
    ("Baby", "Baby"),
    ("Kids", "Kids"),
)
CAMEL_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")
SIZE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "m",
        "l",
        "2xl",
        "medium",
        "xxl",
        "s",
        "small",
        "xxxl",
        "3xl",
        "xl",
        "xs",
        "large",
        "xxs",
    }
)
CHILD_PREFIX_TO_AUDIENCE: Final[tuple[tuple[str, str], ...]] = (
    ("Womens", "women"),
    ("Mens", "men"),
    ("Girls", "girls"),
    ("Boys", "boys"),
    ("Baby", "baby"),
    ("Kids", "kids"),
)
AUDIENCE_PREFIXES: Final[tuple[str, ...]] = (
    "Womens",
    "Mens",
    "Girls",
    "Boys",
    "Baby",
    "Kids",
)
RE_WOMEN = re.compile(r"\bwomen['']?s?\b|\bwoman\b|\bfemale\b|\bladies\b", re.IGNORECASE)
RE_MEN = re.compile(r"\bmen['']?s?\b|\bman\b|\bmale\b", re.IGNORECASE)
RE_GIRLS = re.compile(r"\bgirls?\b", re.IGNORECASE)
RE_BOYS = re.compile(r"\bboys?\b", re.IGNORECASE)
RE_BABY = re.compile(r"\bbaby\b|\binfant\b|\btoddler\b", re.IGNORECASE)
RE_KIDS = re.compile(r"\bkids?\b|\bchildren\b", re.IGNORECASE)
RE_UNISEX = re.compile(r"\bunisex\b", re.IGNORECASE)
WEIGHT_SENTINEL = "999999999"
GRAM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(g|kg|gram|grams|kilogram|kilograms)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[^a-z0-9]+")


def tokenize_title(title: str) -> list[str]:
    lowered = title.lower()
    tokens = [t for t in TOKEN_RE.split(lowered) if t]
    out: list[str] = []
    for tok in tokens:
        if tok in ("s", "ss") and out:
            continue
        if tok == "s" and not out:
            continue
        out.append(tok)
    return out


def _color_stem(tok: str) -> str | None:
    canonical = COLOR_ALIASES.get(tok, tok)
    if canonical in COLOR_WORDS:
        return canonical
    if tok.endswith("ish") and len(tok) > 4:
        stem = COLOR_ALIASES.get(tok[:-3], tok[:-3])
        if stem in COLOR_WORDS:
            return stem
    return None


def is_color_token(tok: str) -> bool:
    if _color_stem(tok) is not None:
        return True
    return tok in COLOR_MODIFIERS


def normalize_title_for_group(title: str) -> str:
    tokens = tokenize_title(title)
    filtered = [t for t in tokens if t not in SIZE_TOKENS and not is_color_token(t)]
    return " ".join(filtered)


def _title_group_key(norm: str, bcode: int, brands_for_norm: set[int]) -> tuple[str, int]:
    if bcode < 0 and len(brands_for_norm) == 1:
        return (norm, next(iter(brands_for_norm)))
    return (norm, bcode)


def assign_title_groups(
    normalized_titles: list[str],
    brand_codes: list[int] | tuple[int, ...],
) -> list[int]:
    bcodes = [int(b) for b in brand_codes]
    brands_for_norm: dict[str, set[int]] = {}
    for norm, bcode in zip(normalized_titles, bcodes, strict=True):
        if norm and bcode >= 0:
            brands_for_norm.setdefault(norm, set()).add(bcode)

    title_keys: dict[tuple[str, int], int] = {}
    tg: list[int] = []
    for norm, bcode in zip(normalized_titles, bcodes, strict=True):
        key = _title_group_key(norm, bcode, brands_for_norm.get(norm, set()))
        if key not in title_keys:
            title_keys[key] = len(title_keys)
        tg.append(title_keys[key])
    return tg


def _split_camel(segment: str) -> list[str]:
    if not segment:
        return []
    return CAMEL_SPLIT_RE.findall(segment)


def _humanize_category_rest(rest: str) -> str:
    if not rest:
        return ""
    if "_" in rest:
        return "-".join(rest.split("_"))
    return " ".join(_split_camel(rest))


def category_family_key(child_key: str | None) -> str:
    if not child_key:
        return ""
    rest = child_key
    for prefix in AUDIENCE_PREFIXES:
        if child_key.startswith(prefix):
            rest = child_key[len(prefix) :]
            break
    words = _split_camel(rest)
    return words[0] if words else ""


def strip_brand_from_title(brand: str | None, title: str) -> str:
    if not brand or not str(brand).strip():
        return title
    b = str(brand).strip().lower()
    tokens = {t for t in TOKEN_RE.split(b) if t}
    kept = [t for t in tokenize_title(title) if t != b and t not in tokens]
    return " ".join(kept) if kept else title


def humanize_category_key(key: str | None) -> str:
    if not key:
        return ""
    for prefix, label in DISPLAY_PREFIX_TO_LABEL:
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            tail = _humanize_category_rest(rest)
            return f"{label} {tail}".strip()
    return key


def _parse_single_weight(s: str) -> float | None:
    m = GRAM_RE.search(s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("k"):
        return val * 1000.0
    return val


def parse_weight_grams(weight: str | None, details: dict[str, object] | None) -> float | None:
    if weight and weight.strip() != WEIGHT_SENTINEL:
        parsed = _parse_single_weight(weight)
        if parsed is not None:
            return parsed
    if details:
        iw = details.get("Item_Weight") or details.get("Item_weight")
        if iw:
            return _parse_single_weight(str(iw))
    return None


def parse_price(sales_price: object | None) -> float | None:
    if sales_price is None or not str(sales_price).strip():
        return None
    try:
        val = float(str(sales_price).replace(",", "").strip())
    except ValueError:
        return None
    if val <= 0:
        return None
    return val


def parse_rating(rating: object | None) -> float | None:
    if rating is None:
        return None
    try:
        return float(str(rating).strip())
    except ValueError:
        return None


def parse_review_count(
    no_reviews: object | None,
    details: dict[str, object] | None,
) -> int:
    if no_reviews is not None and str(no_reviews).strip():
        try:
            return int(str(no_reviews).replace(",", "").strip())
        except ValueError:
            pass
    if details:
        cr = details.get("Customer_Reviews") or details.get("Customer_reviews")
        if cr:
            m = re.search(r"(\d+)\s+customer", str(cr), re.IGNORECASE)
            if m:
                return int(m.group(1))
            if "first to review" in str(cr).lower():
                return 0
    return 0


def parse_brand(brand: str | None) -> str | None:
    if brand is None:
        return None
    b = brand.strip()
    if not b or b.lower() == "generic":
        return None
    return b.casefold()


def parse_color_tokens(colour: str | None, title: str) -> frozenset[str]:
    tokens: set[str] = set()
    if colour:
        for part in colour.split("|"):
            for tok in tokenize_title(part):
                tokens.add(COLOR_ALIASES.get(tok, tok))
    for tok in tokenize_title(title):
        stem = _color_stem(tok)
        if stem is not None:
            tokens.add(stem)
    return frozenset(tokens)


def _present_rank_keys(rank_map: dict[str, object] | None) -> list[str]:
    if not rank_map:
        return []
    keys: list[str] = []
    for key, value in rank_map.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        keys.append(key)
    return keys


def child_category_key(
    sales_rank_child: dict[str, object] | None,
    parent_child: dict[str, object] | None,
) -> str | None:
    ranked = _present_rank_keys(sales_rank_child)
    if ranked:
        return ranked[0]
    ranked_parent = _present_rank_keys(parent_child)
    if len(ranked_parent) >= 2:
        return ranked_parent[-1]
    if ranked_parent:
        return ranked_parent[0]
    return None


def audience_from_child_key(child_key: str | None) -> str | None:
    if not child_key:
        return None
    for prefix, label in CHILD_PREFIX_TO_AUDIENCE:
        if child_key.startswith(prefix):
            return label
    return None


def audience_from_title(title: str) -> str | None:
    hits: set[str] = set()
    if RE_WOMEN.search(title):
        hits.add("women")
    if RE_MEN.search(title):
        hits.add("men")
    if RE_GIRLS.search(title):
        hits.add("girls")
    if RE_BOYS.search(title):
        hits.add("boys")
    if RE_BABY.search(title):
        hits.add("baby")
    if RE_KIDS.search(title):
        hits.add("kids")
    if RE_UNISEX.search(title):
        hits.add("unisex")
    if "women" in hits and "men" in hits:
        return "unisex"
    if len(hits) == 1:
        return next(iter(hits))
    if len(hits) > 1:
        return "unisex"
    return None


def resolve_audience(child_key: str | None, title: str) -> str:
    from_child = audience_from_child_key(child_key)
    if from_child:
        return from_child
    from_title = audience_from_title(title)
    if from_title:
        return from_title
    return "unknown"


def _strict_title_groups(
    normalized_titles: list[str],
    brand_codes: list[int] | tuple[int, ...],
) -> list[int]:
    bcodes = [int(b) for b in brand_codes]
    keys: dict[tuple[str, int], int] = {}
    out: list[int] = []
    for norm, bcode in zip(normalized_titles, bcodes, strict=True):
        key = (norm, bcode)
        if key not in keys:
            keys[key] = len(keys)
        out.append(keys[key])
    return out


def dedup_merge_stats(
    normalized_titles: list[str],
    brand_codes: list[int] | tuple[int, ...],
    top_k: int = 20,
) -> dict[str, Any]:
    merged = assign_title_groups(normalized_titles, brand_codes)
    bcodes = [int(b) for b in brand_codes]
    brands_for_norm: dict[str, set[int]] = {}
    for norm, bcode in zip(normalized_titles, bcodes, strict=True):
        if norm and bcode >= 0:
            brands_for_norm.setdefault(norm, set()).add(bcode)
    reassigned = 0
    for norm, bcode in zip(normalized_titles, bcodes, strict=True):
        if bcode < 0 and len(brands_for_norm.get(norm, set())) == 1:
            reassigned += 1
    group_sizes: Counter[int] = Counter(int(g) for g in merged)
    largest = [{"title_group": g, "row_count": c} for g, c in group_sizes.most_common(top_k)]
    return {
        "missing_brand_rows_reassigned": reassigned,
        "largest_title_groups": largest,
    }
