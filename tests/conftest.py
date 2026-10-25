"""Synthetic catalog for unit tests."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from service.catalog import Catalog
from service.transforms import TextImageTransform


def make_synthetic_catalog() -> Catalog:
    n = 20
    rng = np.random.default_rng(0)
    text = rng.standard_normal((n, 8))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    image = rng.standard_normal((n, 8))
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    image_mask = np.array([True] * 15 + [False] * 5)
    uniq_ids = [f"id{i:02d}" for i in range(n)]
    brand_code = np.array([0, 0, 1, 1, -1] + [-1] * 15, dtype=np.int32)
    category_code = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    audience_code = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    title_group = np.array(list(range(10)) + list(range(10)), dtype=np.int32)
    color_csr = sparse.eye(n, format="csr")
    color_missing = np.zeros(n, dtype=bool)
    price = np.linspace(100, 500, n)
    rating = np.full(n, 4.0)
    rating_n = np.array([0] * 18 + [5, 5], dtype=np.int32)
    weight_g = np.full(n, np.nan)
    weight_g[0] = 200.0
    weight_g[1] = 200.0
    pair_samples = {
        "text": rng.uniform(0, 1, 1000),
        "image": rng.uniform(0, 1, 1000),
        "color": rng.uniform(0, 1, 1000),
        "price": rng.uniform(0, 1, 1000),
        "rating": rng.uniform(0, 1, 1000),
        "weight": rng.uniform(0, 1, 1000),
    }
    mu = {
        TextImageTransform.RAW.value: {"text": 0.2, "image": 0.2},
        TextImageTransform.PERCENTILE.value: {"text": 0.5, "image": 0.5},
        TextImageTransform.TAIL_LOG.value: {"text": 0.05, "image": 0.05},
    }
    return Catalog(
        n=n,
        uniq_ids=uniq_ids,
        id_to_row={u: i for i, u in enumerate(uniq_ids)},
        text=text.astype(np.float64),
        image=image.astype(np.float64),
        image_mask=image_mask,
        brand_code=brand_code,
        category_code=category_code,
        audience_code=audience_code,
        title_group=title_group,
        color_csr=color_csr,
        color_missing=color_missing,
        price=price,
        rating=rating,
        rating_n=rating_n,
        weight_g=weight_g,
        brand_freq=np.array([0.5, 0.3, 0.2]),
        category_freq=np.array([0.5, 0.5]),
        audience_freq=np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0.0]),
        sum_p2_brand=0.38,
        sum_p2_category=0.5,
        sum_p2_audience=0.25,
        catalog_rating_mean=4.0,
        pair_samples=pair_samples,
        mu_transform=mu,
        manifest={},
        product_names=[f"Product {i}" for i in range(n)],
        brands=["BrandA", "BrandA", "BrandB", "BrandB", ""] + [""] * 15,
        categories=["Cat0"] * 10 + ["Cat1"] * 10,
        audiences=["women"] * 10 + ["men"] * 10,
        image_urls=[""] * n,
        names_lower=np.array([f"product {i}" for i in range(n)], dtype=np.str_),
        hnsw_index=None,
    )
