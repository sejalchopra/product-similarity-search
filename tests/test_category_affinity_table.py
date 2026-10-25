"""Precomputed category affinity matches the legacy loop."""

from __future__ import annotations

import numpy as np

from service.block_batch import compute_block_batch
from service.catalog import build_category_centroids, sample_category_centroid_cosines
from service.category_affinity import build_affinity_tables, build_centroid_cosine_matrix
from service.scoring import FusionWeights
from service.scoring_profile import ScoringProfile
from service.transforms import TextImageTransform
from tests.conftest import make_synthetic_catalog


def test_affinity_table_matches_loop() -> None:
    catalog = make_synthetic_catalog()
    text = catalog.text.astype(np.float32)
    cat_codes = catalog.category_code
    n_cats = int(cat_codes.max()) + 1
    centroids = build_category_centroids(text, cat_codes, n_cats)
    sample = sample_category_centroid_cosines(centroids, cat_codes, np.random.default_rng(0), 5000)
    tables = build_affinity_tables(centroids, sample)
    cos = build_centroid_cosine_matrix(centroids)
    catalog.category_centroids = centroids
    catalog.category_centroid_cos_sample = sample
    catalog.category_centroid_cosine = cos
    catalog.category_affinity_tables = tables

    qrow = 0
    cand = np.arange(catalog.n, dtype=np.intp)
    for ti in TextImageTransform:
        fw = FusionWeights(
            w={
                "text": 0.3,
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
            text_image_transform=ti,
        )
        fast = compute_block_batch(
            catalog, qrow, cand, fw, ScoringProfile(category_mode="affinity"), metadata=False
        ).values[:, 2]
        catalog.category_affinity_tables = None
        slow = compute_block_batch(
            catalog, qrow, cand, fw, ScoringProfile(category_mode="affinity"), metadata=False
        ).values[:, 2]
        catalog.category_affinity_tables = tables
        np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-9)
