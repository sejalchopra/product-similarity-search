"""Vectorized fusion blocks with raw similarities and imputation metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from service.query_overrides import QueryOverrides
from service.scoring import (
    AUDIENCE_TABLE,
    BLOCK_NAMES,
    FusionWeights,
    audience_expected_impute,
    price_similarity,
    rating_similarity,
    shrink_rating,
    weight_similarity,
)
from service.scoring_profile import ScoringProfile
from service.transforms import midrank_percentile, transform_continuous

if TYPE_CHECKING:
    from service.catalog import Catalog

FloatArray = NDArray[np.float64]


@dataclass
class BlockBatch:
    values: FloatArray
    raw_sim: dict[str, FloatArray | None]
    imputed: NDArray[np.bool_]
    imputed_reason: list[list[str | None]]


def _reason(
    q_ok: bool,
    c_ok: bool,
    field: str,
) -> tuple[bool, str | None]:
    if q_ok and c_ok:
        return False, None
    if not q_ok and not c_ok:
        return True, f"both {field} missing"
    if not q_ok:
        return True, f"query {field} missing"
    return True, f"candidate {field} missing"


def compute_block_batch(
    catalog: Catalog,
    query_row: int,
    candidate_rows: NDArray[np.intp],
    weights: FusionWeights,
    profile: ScoringProfile | None = None,
    overrides: QueryOverrides | None = None,
    *,
    metadata: bool = True,
    block_timings: dict[str, float] | None = None,
) -> BlockBatch:
    prof = profile or ScoringProfile()
    ov = overrides or QueryOverrides()
    cand = candidate_rows
    m = cand.shape[0]
    mat = np.zeros((m, len(BLOCK_NAMES)), dtype=np.float64)
    _clock = time.perf_counter()

    def _lap(name: str) -> None:
        nonlocal _clock
        now = time.perf_counter()
        if block_timings is not None:
            block_timings[name] = block_timings.get(name, 0.0) + (now - _clock)
        _clock = now

    if metadata:
        imputed = np.zeros((m, len(BLOCK_NAMES)), dtype=bool)
        reasons: list[list[str | None]] = [[None] * len(BLOCK_NAMES) for _ in range(m)]
    else:
        imputed = np.zeros((0, 0), dtype=bool)
        reasons = []

    ti = weights.text_image_transform
    beta = weights.beta

    text_mat = catalog.text
    if prof.text_matrix == "nobrand" and catalog.text_nobrand is not None:
        text_mat = catalog.text_nobrand

    text_raw = text_mat[cand] @ text_mat[query_row]
    text_s = transform_continuous(text_raw, catalog.pair_samples["text"], ti)
    mat[:, 0] = text_s
    raw_text: FloatArray = text_raw.astype(np.float64)
    _lap("text")

    q_has_img = bool(catalog.image_mask[query_row]) and not ov.mask_image
    img_raw = np.zeros(m, dtype=np.float64)
    if q_has_img:
        img_raw = catalog.image[cand] @ catalog.image[query_row]
    cand_has = catalog.image_mask[cand]
    both_img = q_has_img & cand_has
    mu_img = catalog.mu_transform[ti.value]["image"]
    img_s = transform_continuous(img_raw, catalog.pair_samples["image"], ti)
    mat[:, 1] = np.where(both_img, img_s, mu_img)
    _lap("image")
    if metadata:
        for i in range(m):
            imp, rs = _reason(q_has_img, bool(cand_has[i]), "image")
            imputed[i, 1] = imp
            reasons[i][1] = rs

    qc = int(catalog.category_code[query_row])
    cc = catalog.category_code[cand]
    both_c = (qc >= 0) & (cc >= 0)
    cat_col = np.zeros(m, dtype=np.float64)
    raw_cat = np.full(m, np.nan, dtype=np.float64)

    if prof.category_mode == "affinity" and catalog.category_centroids is not None:
        mu_aff = float(catalog.mu_category_affinity)
        aff_tables = catalog.category_affinity_tables
        cos_mat = catalog.category_centroid_cosine
        if aff_tables is not None and qc >= 0 and ti.value in aff_tables:
            table = aff_tables[ti.value]
            cat_col = np.full(m, mu_aff, dtype=np.float64)
            if both_c.any():
                cat_col[both_c] = table[qc, cc[both_c]]
            if cos_mat is not None:
                raw_cat = np.where(both_c, cos_mat[qc, cc], np.nan)
            else:
                raw_cat = np.full(m, np.nan)
            if metadata:
                for i in range(m):
                    if both_c[i]:
                        imputed[i, 2] = False
                        reasons[i][2] = None
                    else:
                        imp, rs = _reason(qc >= 0, int(cc[i]) >= 0, "category")
                        imputed[i, 2] = imp
                        reasons[i][2] = rs
        else:
            cent = catalog.category_centroids
            sample = catalog.category_centroid_cos_sample
            if sample is None:
                sample = np.array([0.5], dtype=np.float64)
            for i in range(m):
                c_i = int(cc[i])
                if both_c[i]:
                    if metadata:
                        imputed[i, 2] = False
                        reasons[i][2] = None
                    if c_i == qc:
                        cat_col[i] = 1.0
                        raw_cat[i] = 1.0
                    else:
                        if cos_mat is not None:
                            rv = float(cos_mat[qc, c_i])
                        else:
                            rv = float(cent[qc] @ cent[c_i])
                        raw_cat[i] = rv
                        cat_col[i] = float(
                            transform_continuous(
                                np.array([rv], dtype=np.float64),
                                sample,
                                ti,
                            )[0]
                        )
                else:
                    cat_col[i] = mu_aff
                    if metadata:
                        imp, rs = _reason(qc >= 0, c_i >= 0, "category")
                        imputed[i, 2] = imp
                        reasons[i][2] = rs
    else:
        for i in range(m):
            c_i = int(cc[i])
            if both_c[i]:
                cat_col[i] = 1.0 if c_i == qc else 0.0
                raw_cat[i] = cat_col[i]
            elif qc >= 0:
                cat_col[i] = min(beta * catalog.category_freq[qc], 0.5)
            elif c_i >= 0:
                cat_col[i] = min(beta * catalog.category_freq[c_i], 0.5)
            else:
                cat_col[i] = min(beta * catalog.sum_p2_category, 0.5)
            if metadata:
                imp, rs = _reason(qc >= 0, c_i >= 0, "category")
                imputed[i, 2] = imp
                reasons[i][2] = rs
    mat[:, 2] = cat_col
    _lap("category")

    qa = int(catalog.audience_code[query_row])
    ca = catalog.audience_code[cand]
    aud_col = np.zeros(m, dtype=np.float64)
    raw_aud = np.full(m, np.nan, dtype=np.float64)
    for i in range(m):
        c_a = int(ca[i])
        if qa >= 0 and c_a >= 0:
            aud_col[i] = AUDIENCE_TABLE[qa, c_a]
            raw_aud[i] = aud_col[i]
        elif qa >= 0:
            aud_col[i] = min(
                beta
                * audience_expected_impute(
                    qa,
                    catalog.audience_freq,
                    AUDIENCE_TABLE,
                    catalog.sum_p2_audience,
                ),
                0.5,
            )
        elif c_a >= 0:
            aud_col[i] = min(
                beta
                * audience_expected_impute(
                    c_a,
                    catalog.audience_freq,
                    AUDIENCE_TABLE,
                    catalog.sum_p2_audience,
                ),
                0.5,
            )
        else:
            aud_col[i] = min(beta * catalog.sum_p2_audience, 0.5)
        if metadata:
            imp, rs = _reason(qa >= 0, c_a >= 0, "audience")
            imputed[i, 3] = imp
            reasons[i][3] = rs
    mat[:, 3] = aud_col
    _lap("audience")

    qb = int(catalog.brand_code[query_row])
    if ov.mask_brand:
        qb = -1
    cb = catalog.brand_code[cand]
    brand_col = np.zeros(m, dtype=np.float64)
    raw_brand = np.full(m, np.nan, dtype=np.float64)
    for i in range(m):
        c_b = int(cb[i])
        if qb >= 0 and c_b >= 0:
            brand_col[i] = 1.0 if c_b == qb else 0.0
            raw_brand[i] = brand_col[i]
        elif qb >= 0:
            brand_col[i] = min(beta * catalog.brand_freq[qb], 0.5)
        elif c_b >= 0:
            brand_col[i] = min(beta * catalog.brand_freq[c_b], 0.5)
        else:
            brand_col[i] = min(beta * catalog.sum_p2_brand, 0.5)
        if metadata:
            imp, rs = _reason(qb >= 0, c_b >= 0, "brand")
            imputed[i, 4] = imp
            reasons[i][4] = rs
    mat[:, 4] = brand_col
    _lap("brand")

    col_dots = (
        (catalog.color_csr[cand] @ catalog.color_csr[query_row].T)
        .toarray()
        .astype(np.float64)
        .ravel()
    )
    q_col_miss = bool(catalog.color_missing[query_row])
    q_col_ok = not q_col_miss
    both_col = (~catalog.color_missing[cand]) & q_col_ok
    mu_col = 0.5
    col_p = midrank_percentile(col_dots, catalog.pair_samples["color"])
    mat[:, 5] = np.where(both_col, col_p, mu_col)
    _lap("color")
    if metadata:
        for i in range(m):
            imp, rs = _reason(not q_col_miss, not bool(catalog.color_missing[cand[i]]), "color")
            imputed[i, 5] = imp
            reasons[i][5] = rs

    qp = catalog.price[query_row]
    cp = catalog.price[cand]
    both_p = np.isfinite(qp) & np.isfinite(cp)
    raw_p = np.array(
        [price_similarity(qp, cp[i]) if both_p[i] else 0.0 for i in range(m)],
        dtype=np.float64,
    )
    raw_p_store = np.where(both_p, raw_p, np.nan)
    p_s = midrank_percentile(np.nan_to_num(raw_p, nan=0.0), catalog.pair_samples["price"])
    mat[:, 6] = np.where(both_p, p_s, 0.5)
    _lap("price")
    if metadata:
        for i in range(m):
            imp, rs = _reason(np.isfinite(qp), np.isfinite(cp[i]), "price")
            imputed[i, 6] = imp
            reasons[i][6] = rs

    qn = int(catalog.rating_n[query_row])
    cn = catalog.rating_n[cand]
    qr = catalog.rating[query_row]
    cr = catalog.rating[cand]
    both_r = (qn > 0) & (cn > 0) & np.isfinite(qr) & np.isfinite(cr)
    rs_arr = np.zeros(m, dtype=np.float64)
    for i in range(m):
        if both_r[i]:
            r1 = shrink_rating(float(qr), qn, catalog.catalog_rating_mean)
            r2 = shrink_rating(float(cr[i]), int(cn[i]), catalog.catalog_rating_mean)
            rs_arr[i] = rating_similarity(r1, r2)
    r_s = midrank_percentile(rs_arr, catalog.pair_samples["rating"])
    mat[:, 7] = np.where(both_r, r_s, 0.5)
    _lap("rating")
    raw_rating = np.where(both_r, rs_arr, np.nan)
    if metadata:
        for i in range(m):
            imp, rs = _reason(
                qn > 0 and np.isfinite(qr),
                int(cn[i]) > 0 and np.isfinite(cr[i]),
                "rating",
            )
            imputed[i, 7] = imp
            reasons[i][7] = rs

    qw = catalog.weight_g[query_row]
    cw = catalog.weight_g[cand]
    both_w = np.isfinite(qw) & np.isfinite(cw)
    ws = np.zeros(m, dtype=np.float64)
    for i in range(m):
        if both_w[i]:
            ws[i] = weight_similarity(float(qw), float(cw[i]))
    w_s = midrank_percentile(ws, catalog.pair_samples["weight"])
    mat[:, 8] = np.where(both_w, w_s, 0.5)
    _lap("weight")
    raw_weight = np.where(both_w, ws, np.nan)
    if metadata:
        for i in range(m):
            imp, rs = _reason(np.isfinite(qw), np.isfinite(cw[i]), "weight")
            imputed[i, 8] = imp
            reasons[i][8] = rs

    if not metadata:
        return BlockBatch(values=mat, raw_sim={}, imputed=imputed, imputed_reason=reasons)

    raw_map: dict[str, FloatArray | None] = {
        "text": raw_text,
        "image": np.where(both_img, img_raw, np.nan),
        "category": raw_cat,
        "audience": raw_aud,
        "brand": raw_brand,
        "color": np.where(both_col, col_dots, np.nan),
        "price": raw_p_store,
        "rating": raw_rating,
        "weight": raw_weight,
    }
    imputed[:, 0] = False
    return BlockBatch(values=mat, raw_sim=raw_map, imputed=imputed, imputed_reason=reasons)
