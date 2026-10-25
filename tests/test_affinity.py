from __future__ import annotations

import numpy as np

from service.block_batch import compute_block_batch
from service.catalog import Catalog, build_category_centroids
from service.scoring import FusionWeights
from service.scoring_profile import ScoringProfile


def _mini_catalog() -> Catalog:
    n = 4
    dim = 8
    text = np.zeros((n, dim), dtype=np.float64)
    text[0] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    text[1] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    text[2] = np.array([0, 1.0, 0, 0, 0, 0, 0, 0])
    text[3] = np.array([0, 0, 1.0, 0, 0, 0, 0, 0])
    from scipy import sparse

    cat_codes = np.array([0, 1, 1, 2], dtype=np.int32)
    centroids = build_category_centroids(text.astype(np.float32), cat_codes, 3)
    cos_sample = np.linspace(0.2, 0.95, 500, dtype=np.float64)
    return Catalog(
        n=n,
        uniq_ids=[f"id{i}" for i in range(n)],
        id_to_row={f"id{i}": i for i in range(n)},
        text=text,
        image=np.zeros((n, dim)),
        image_mask=np.zeros(n, dtype=bool),
        brand_code=np.array([0, 0, 0, 0], dtype=np.int32),
        category_code=cat_codes,
        audience_code=np.array([1, 1, 1, 1], dtype=np.int32),
        title_group=np.arange(n, dtype=np.int32),
        color_csr=sparse.eye(n, format="csr"),
        color_missing=np.ones(n, dtype=bool),
        price=np.full(n, 100.0),
        rating=np.full(n, 4.0),
        rating_n=np.zeros(n, dtype=np.int32),
        weight_g=np.full(n, np.nan),
        brand_freq=np.array([1.0]),
        category_freq=np.array([0.25, 0.5, 0.25]),
        audience_freq=np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        sum_p2_brand=1.0,
        sum_p2_category=0.375,
        sum_p2_audience=1.0,
        catalog_rating_mean=4.0,
        pair_samples={
            "text": cos_sample,
            "image": cos_sample,
            "color": cos_sample,
            "price": cos_sample,
            "rating": cos_sample,
            "weight": cos_sample,
        },
        mu_transform={"tail_log": {"text": 0.05, "image": 0.5}},
        manifest={},
        product_names=[f"P{i}" for i in range(n)],
        brands=["b"] * n,
        categories=[
            "MensUnderwearBriefs",
            "MensUnderwearBriefs",
            "MensT_Shirts",
            "MensUnderwearTrunks",
        ],
        audiences=["men"] * n,
        image_urls=[""] * n,
        names_lower=np.array([f"p{i}" for i in range(n)], dtype=np.str_),
        category_centroids=centroids,
        category_centroid_cos_sample=cos_sample,
        mu_category_affinity=0.5,
    )


def test_category_affinity_symmetric_and_diagonal() -> None:
    cat = _mini_catalog()
    w = FusionWeights(
        w={
            "text": 0,
            "image": 0,
            "category": 1,
            "audience": 0,
            "brand": 0,
            "color": 0,
            "price": 0,
            "rating": 0,
            "weight": 0,
        },
        beta=1.0,
    )
    prof = ScoringProfile(category_mode="affinity")
    b01 = compute_block_batch(cat, 0, np.array([1], dtype=np.intp), w, prof)
    b10 = compute_block_batch(cat, 1, np.array([0], dtype=np.intp), w, prof)
    assert abs(b01.values[0, 2] - b10.values[0, 2]) < 1e-9
    b00 = compute_block_batch(cat, 0, np.array([0], dtype=np.intp), w, prof)
    assert b00.values[0, 2] == 1.0


def test_same_family_brief_beats_same_brand_tshirt() -> None:
    cat = _mini_catalog()
    w = FusionWeights(
        w={
            "text": 0.5,
            "image": 0,
            "category": 0.4,
            "audience": 0,
            "brand": 0.1,
            "color": 0,
            "price": 0,
            "rating": 0,
            "weight": 0,
        },
        beta=1.0,
    )
    prof = ScoringProfile(category_mode="affinity")
    q = 0
    brief = compute_block_batch(cat, q, np.array([1], dtype=np.intp), w, prof)
    shirt = compute_block_batch(cat, q, np.array([2], dtype=np.intp), w, prof)
    s_brief = float((brief.values * w.normalized().vector()).sum())
    s_shirt = float((shirt.values * w.normalized().vector()).sum())
    assert s_brief > s_shirt
