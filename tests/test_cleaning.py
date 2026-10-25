from __future__ import annotations

from service.cleaning import (
    assign_title_groups,
    child_category_key,
    dedup_merge_stats,
    humanize_category_key,
    is_color_token,
    normalize_title_for_group,
    parse_color_tokens,
    resolve_audience,
)

GREY_TRACKSUIT = "Monte Carlo Grey Solid Cotton Blend Polo Collar Tracksuit"
BLACKISH_TRACKSUIT = "Monte Carlo Blackish Solid Cotton Blend Polo Collar Tracksuit"


def test_monte_carlo_colorways_share_normalized_title() -> None:
    grey = normalize_title_for_group(GREY_TRACKSUIT)
    blackish = normalize_title_for_group(BLACKISH_TRACKSUIT)
    assert grey == blackish
    assert grey == ("monte carlo solid cotton blend polo collar tracksuit")


def test_monte_carlo_colorways_same_title_group_with_missing_brand() -> None:
    grey = normalize_title_for_group(GREY_TRACKSUIT)
    blackish = normalize_title_for_group(BLACKISH_TRACKSUIT)
    groups = assign_title_groups([grey, blackish], [-1, -1])
    assert groups[0] == groups[1]


def test_missing_brand_joins_single_branded_title_group() -> None:
    norm = "van heusen men formal trousers"
    groups = assign_title_groups([norm, norm, norm], [-1, 5, 5])
    assert groups[0] == groups[1] == groups[2]


def test_missing_brand_stays_split_when_multiple_brands() -> None:
    norm = "ambiguous title"
    groups = assign_title_groups([norm, norm, norm], [-1, 1, 2])
    assert groups[0] != groups[1]
    assert groups[0] != groups[2]
    assert groups[1] != groups[2]


def test_generic_product_title_two_real_brands_not_merged() -> None:
    norm = normalize_title_for_group("Generic Product Title Red Size M")
    groups = assign_title_groups([norm, norm, norm], [-1, 4, 9])
    assert len(set(groups)) == 3


def test_dedup_merge_stats_counts_reassignment() -> None:
    norm = "shared title"
    stats = dedup_merge_stats([norm, norm], [-1, 3])
    assert stats["missing_brand_rows_reassigned"] == 1


def test_color_tokens_strip_grey_gray_blackish() -> None:
    assert is_color_token("grey")
    assert is_color_token("gray")
    assert is_color_token("blackish")
    assert not is_color_token("cotton")
    assert not is_color_token("tracksuit")


def test_size_tokens_still_dropped_from_group_title() -> None:
    title = "Brand Shirt Red Size M XL"
    norm = normalize_title_for_group(title)
    assert "m" not in norm.split()
    assert "xl" not in norm.split()
    assert "red" not in norm.split()


def test_parse_color_tokens_title_ignores_non_colors() -> None:
    tokens = parse_color_tokens(None, "Grey Cotton Tracksuit")
    assert "cotton" not in tokens
    assert "tracksuit" not in tokens
    assert "gray" in tokens


def test_humanize_mens_t_shirts() -> None:
    assert humanize_category_key("MensT_Shirts") == "Men's T-Shirts"


def test_humanize_womens_kurtas() -> None:
    assert humanize_category_key("WomensKurtasKurtis") == "Women's Kurtas Kurtis"


def test_child_category_key_ignores_polars_null_struct_fields() -> None:
    widened = {
        "MensT_Shirts": None,
        "WomensKurtasKurtis": "#91",
        "MensTracksuits": None,
    }
    assert child_category_key(widened, None) == "WomensKurtasKurtis"


def test_child_category_key_lovista_audience() -> None:
    widened = {"MensT_Shirts": None, "WomensKurtasKurtis": "#91"}
    child = child_category_key(widened, None)
    title = (
        "LOVISTA Cotton Gota Patti Tassel Traditional Printed Kurti with Pant,Angrakha Salwar Suit"
    )
    assert child == "WomensKurtasKurtis"
    assert resolve_audience(child, title) == "women"
