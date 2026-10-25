from __future__ import annotations

import numpy as np

from service.cleaning import audience_from_title
from service.config import Settings
from service.scoring import FusionWeights
from service.similarity import find_similar_products
from service.transforms import TextImageTransform, apply_tail_log, midrank_percentile
from tests.conftest import make_synthetic_catalog


def test_query_excluded_from_results() -> None:
    cat = make_synthetic_catalog()
    settings = Settings()
    out = find_similar_products(cat, "id00", 5, settings)
    assert "id00" not in out
    assert len(out) <= 5


def test_womens_title_not_unisex() -> None:
    assert audience_from_title("RJFabrics Women's Cotton Saree") == "women"
    assert audience_from_title("Sf Jeans By Pantaloons Men's T-Shirt") == "men"
    assert audience_from_title("Men and Women Jacket") == "unisex"


def test_tail_log_ceiling() -> None:
    n = 200_000
    p = np.array([1.0 - 1.0 / (2 * n)])
    s = apply_tail_log(p, n)
    assert float(s[0]) > 0.99


def test_missing_image_below_near_duplicate() -> None:
    cat = make_synthetic_catalog()
    q = 0
    cand_with = 1
    cand_missing = 16
    assert cat.image_mask[cand_with]
    assert not cat.image_mask[cand_missing]
    w = FusionWeights(
        w={
            "text": 0,
            "image": 1,
            "category": 0,
            "audience": 0,
            "brand": 0,
            "color": 0,
            "price": 0,
            "rating": 0,
            "weight": 0,
        },
        beta=1.0,
        text_image_transform=TextImageTransform.TAIL_LOG,
    )
    mat_obs = cat.block_matrix(q, np.array([cand_with]), w)
    mat_miss = cat.block_matrix(q, np.array([cand_missing]), w)
    assert float(mat_miss[0, 1]) < float(mat_obs[0, 1])


def test_color_mismatch_below_missing_percentile() -> None:
    sample = np.sort(np.concatenate([np.zeros(900), np.linspace(0.01, 1, 100)]))
    p0 = midrank_percentile(np.array([0.0]), sample)
    assert float(p0[0]) < 0.5
