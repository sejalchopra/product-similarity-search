"""Load artifacts and pre-encoded arrays for vectorized scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import sparse

from service.block_batch import BlockBatch, compute_block_batch
from service.cleaning import assign_title_groups, normalize_title_for_group
from service.query_overrides import QueryOverrides
from service.scoring import FusionWeights, fuse_scores
from service.scoring_profile import ScoringProfile
from service.transforms import TextImageTransform, median_imputed

FloatArray = NDArray[np.float64]


@dataclass
class Catalog:
    n: int
    uniq_ids: list[str]
    id_to_row: dict[str, int]
    text: FloatArray
    image: FloatArray
    image_mask: NDArray[np.bool_]
    brand_code: NDArray[np.int32]
    category_code: NDArray[np.int32]
    audience_code: NDArray[np.int32]
    title_group: NDArray[np.int32]
    color_csr: sparse.csr_matrix
    color_missing: NDArray[np.bool_]
    price: FloatArray
    rating: FloatArray
    rating_n: NDArray[np.int32]
    weight_g: FloatArray
    brand_freq: FloatArray
    category_freq: FloatArray
    audience_freq: FloatArray
    sum_p2_brand: float
    sum_p2_category: float
    sum_p2_audience: float
    catalog_rating_mean: float
    pair_samples: dict[str, FloatArray]
    mu_transform: dict[str, dict[str, float]]
    manifest: dict[str, Any]
    product_names: list[str]
    brands: list[str]
    categories: list[str]
    audiences: list[str]
    image_urls: list[str]
    names_lower: NDArray[np.str_]
    hnsw_index: Any | None = None
    text_nobrand: FloatArray | None = None
    category_centroids: FloatArray | None = None
    category_centroid_cos_sample: FloatArray | None = None
    category_centroid_cosine: FloatArray | None = None
    category_affinity_tables: dict[str, FloatArray] | None = None
    mu_category_affinity: float = 0.5
    active_profile: ScoringProfile = field(default_factory=ScoringProfile)

    def row_for(self, product_id: str) -> int:
        try:
            return self.id_to_row[product_id]
        except KeyError:
            raise KeyError(product_id) from None

    def block_matrix(
        self,
        query_row: int,
        candidate_rows: NDArray[np.intp] | None,
        weights: FusionWeights,
        profile: ScoringProfile | None = None,
        overrides: QueryOverrides | None = None,
    ) -> FloatArray:
        if candidate_rows is None:
            cand = np.arange(self.n, dtype=np.intp)
        else:
            cand = candidate_rows
        prof = profile or self.active_profile
        return compute_block_batch(
            self, query_row, cand, weights, prof, overrides, metadata=False
        ).values

    def block_details(
        self,
        query_row: int,
        candidate_rows: NDArray[np.intp],
        weights: FusionWeights,
        profile: ScoringProfile | None = None,
        overrides: QueryOverrides | None = None,
    ) -> BlockBatch:
        prof = profile or self.active_profile
        return compute_block_batch(self, query_row, candidate_rows, weights, prof, overrides)

    def scores_for_query(
        self,
        query_row: int,
        weights: FusionWeights,
        candidate_rows: NDArray[np.intp] | None = None,
        profile: ScoringProfile | None = None,
        overrides: QueryOverrides | None = None,
    ) -> FloatArray:
        mat = self.block_matrix(
            query_row, candidate_rows, weights, profile=profile, overrides=overrides
        )
        return fuse_scores(mat, weights)

    def ann_candidate_rows(self, query_row: int, settings: Any) -> NDArray[np.intp]:
        k = settings.ann_pool_per_source
        text_scores = self.text @ self.text[query_row]
        text_top = np.argpartition(-text_scores, min(k, self.n - 1))[:k]

        qb = self.brand_code[query_row]
        if qb >= 0:
            brand_rows = np.where(self.brand_code == qb)[0]
            if brand_rows.size > k:
                sub = text_scores[brand_rows]
                brand_top = brand_rows[np.argpartition(-sub, k)[:k]]
            else:
                brand_top = brand_rows
        else:
            brand_top = np.array([], dtype=np.intp)

        qc = self.category_code[query_row]
        if qc >= 0:
            cat_rows = np.where(self.category_code == qc)[0]
            if cat_rows.size > k:
                sub = text_scores[cat_rows]
                cat_top = cat_rows[np.argpartition(-sub, k)[:k]]
            else:
                cat_top = cat_rows
        else:
            cat_top = np.array([], dtype=np.intp)

        if self.hnsw_index is not None:
            qv = self.text[query_row : query_row + 1].astype(np.float32)
            _, ids = self.hnsw_index.search(qv, k)
            hnsw_top = ids[0]
            hnsw_top = hnsw_top[hnsw_top >= 0]
        else:
            hnsw_top = text_top

        union = np.unique(np.concatenate([hnsw_top, brand_top, cat_top, text_top])).astype(np.intp)
        return cast(NDArray[np.intp], union.astype(np.intp, copy=False))


def build_category_centroids(
    text: NDArray[np.float32],
    category_code: NDArray[np.int32],
    n_categories: int,
) -> FloatArray:
    dim = text.shape[1]
    cent = np.zeros((n_categories, dim), dtype=np.float64)
    counts = np.zeros(n_categories, dtype=np.float64)
    for i in range(text.shape[0]):
        c = int(category_code[i])
        if c < 0:
            continue
        cent[c] += text[i].astype(np.float64)
        counts[c] += 1.0
    for c in range(n_categories):
        if counts[c] > 0:
            cent[c] /= counts[c]
            norm = float(np.linalg.norm(cent[c]))
            if norm > 0:
                cent[c] /= norm
    return cent


def sample_category_centroid_cosines(
    centroids: FloatArray,
    category_code: NDArray[np.int32],
    rng: np.random.Generator,
    size: int = 200_000,
) -> FloatArray:
    valid = np.flatnonzero(category_code >= 0)
    if valid.size == 0:
        return np.array([0.5], dtype=np.float64)
    out: list[float] = []
    while len(out) < size:
        i = int(rng.integers(0, valid.size))
        j = int(rng.integers(0, valid.size))
        ci = int(category_code[valid[i]])
        cj = int(category_code[valid[j]])
        out.append(float(centroids[ci] @ centroids[cj]))
    return np.array(out[:size], dtype=np.float64)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_catalog(artifacts_dir: Path, build_hnsw: bool = False) -> Catalog:
    artifacts_dir = Path(artifacts_dir)
    manifest_path = artifacts_dir / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    text_path = artifacts_dir / "text_embeddings.npy"
    image_path = artifacts_dir / "image_embeddings.npy"
    mask_path = artifacts_dir / "image_mask.npy"
    meta_path = artifacts_dir / "metadata.parquet"
    cdf_path = artifacts_dir / "cdf.npz"

    for p, key in [
        (text_path, "text_sha256"),
        (image_path, "image_sha256"),
    ]:
        if key in manifest and p.exists():
            if _sha256_file(p) != manifest[key]:
                raise RuntimeError(f"sha256 mismatch for {p.name}")

    text = np.load(text_path).astype(np.float32)
    image = np.load(image_path).astype(np.float32)
    image_mask = np.load(mask_path).astype(bool)
    meta = pl.read_parquet(meta_path)

    if text.shape[0] != manifest["row_count"]:
        raise RuntimeError("row count mismatch")
    if text.shape[1] != 384:
        raise RuntimeError("text dim must be 384")
    if image.shape[1] != 512:
        raise RuntimeError("image dim must be 512")

    cdf = np.load(cdf_path)
    pair_samples = {k: cdf[k].astype(np.float64) for k in cdf.files}

    mu_transform = manifest.get("mu_transform", {})

    n = text.shape[0]
    uniq_ids = meta["uniq_id"].to_list()
    id_to_row = {u: i for i, u in enumerate(uniq_ids)}

    brand_codes = meta["brand_code"].to_numpy().astype(np.int32)
    category_codes = meta["category_code"].to_numpy().astype(np.int32)
    audience_codes = meta["audience_code"].to_numpy().astype(np.int32)
    color_missing = meta["color_missing"].to_numpy().astype(bool)
    price = meta["price"].to_numpy().astype(np.float64)
    rating = meta["rating"].to_numpy().astype(np.float64)
    rating_n = meta["rating_n"].to_numpy().astype(np.int32)
    weight_g = meta["weight_g"].to_numpy().astype(np.float64)

    color_ind = meta["color_indices"].to_list()
    color_data = meta["color_values"].to_list()
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, (inds, vals) in enumerate(zip(color_ind, color_data, strict=False)):
        for j, v in zip(inds, vals, strict=False):
            rows.append(i)
            cols.append(int(j))
            data.append(float(v))
    color_csr = sparse.csr_matrix((data, (rows, cols)), shape=(n, manifest["color_vocab_size"]))
    color_csr = sparse.csr_matrix(color_csr)
    norms = np.sqrt(color_csr.multiply(color_csr).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    color_csr = sparse.diags(1.0 / norms) @ color_csr

    brand_freq = np.array(manifest["brand_freq"], dtype=np.float64)
    category_freq = np.array(manifest["category_freq"], dtype=np.float64)
    audience_freq = np.array(manifest["audience_freq"], dtype=np.float64)

    hnsw_index = None
    if build_hnsw:
        import faiss

        d = text.shape[1]
        index = faiss.IndexHNSWFlat(d, 16)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index.add(text.astype(np.float32))
        hnsw_index = index

    def _str_col(name: str) -> list[str]:
        return ["" if v is None else str(v) for v in meta[name].to_list()]

    product_names = _str_col("product_name")
    brands = _str_col("brand")
    categories = _str_col("category_child")
    audiences = _str_col("audience")
    image_urls = _str_col("medium_url")
    names_lower = np.array([n.lower() for n in product_names], dtype=np.str_)

    normalized_titles = [normalize_title_for_group(name) for name in product_names]
    title_groups = np.array(
        assign_title_groups(normalized_titles, brand_codes.tolist()),
        dtype=np.int32,
    )

    n_cats = len(manifest.get("category_freq", []))
    centroids = build_category_centroids(text, category_codes, n_cats)
    cent_path = artifacts_dir / "category_centroids.npy"
    if cent_path.exists():
        if "category_centroids_sha256" in manifest:
            if _sha256_file(cent_path) != manifest["category_centroids_sha256"]:
                raise RuntimeError("sha256 mismatch for category_centroids.npy")
        loaded = np.load(cent_path).astype(np.float64)
        if loaded.shape == centroids.shape:
            centroids = loaded
    cos_sample = pair_samples.get("category_centroid_cos")
    if cos_sample is None:
        cos_sample = sample_category_centroid_cosines(
            centroids, category_codes, np.random.default_rng(42)
        )
    mu_aff = float(manifest.get("mu_category_affinity", 0.5))
    if "mu_category_affinity" not in manifest:
        from service.transforms import TextImageTransform

        ti = TextImageTransform.TAIL_LOG
        mu_aff = float(median_imputed(cos_sample, ti))

    from service.category_affinity import (
        build_affinity_tables,
        build_centroid_cosine_matrix,
    )

    cos_mat_path = artifacts_dir / "category_centroid_cosine.npy"
    aff_path = artifacts_dir / "category_affinity_calibrated.npz"
    category_centroid_cosine: FloatArray | None = None
    category_affinity_tables: dict[str, FloatArray] | None = None
    if aff_path.exists() and "category_affinity_calibrated_sha256" in manifest:
        if _sha256_file(aff_path) != manifest["category_affinity_calibrated_sha256"]:
            raise RuntimeError("sha256 mismatch for category_affinity_calibrated.npz")
        with np.load(aff_path) as z:
            category_affinity_tables = {k: z[k].astype(np.float64) for k in z.files}
    else:
        category_affinity_tables = build_affinity_tables(centroids, cos_sample)
    if cos_mat_path.exists() and "category_centroid_cosine_sha256" in manifest:
        if _sha256_file(cos_mat_path) != manifest["category_centroid_cosine_sha256"]:
            raise RuntimeError("sha256 mismatch for category_centroid_cosine.npy")
        category_centroid_cosine = np.load(cos_mat_path).astype(np.float64)
    else:
        category_centroid_cosine = build_centroid_cosine_matrix(centroids)

    text_nobrand: FloatArray | None = None
    nobrand_path = artifacts_dir / "text_embeddings_nobrand.npy"
    if nobrand_path.exists():
        if "text_nobrand_sha256" in manifest:
            if _sha256_file(nobrand_path) != manifest["text_nobrand_sha256"]:
                raise RuntimeError("sha256 mismatch for text_embeddings_nobrand.npy")
        text_nobrand = np.load(nobrand_path).astype(np.float64)

    variant = str(manifest.get("selected_variant", "binary"))
    active_profile = ScoringProfile.from_variant(variant)

    return Catalog(
        n=n,
        uniq_ids=uniq_ids,
        id_to_row=id_to_row,
        text=text.astype(np.float64),
        image=image.astype(np.float64),
        image_mask=image_mask,
        brand_code=brand_codes,
        category_code=category_codes,
        audience_code=audience_codes,
        title_group=title_groups,
        color_csr=color_csr,
        color_missing=color_missing,
        price=price,
        rating=rating,
        rating_n=rating_n,
        weight_g=weight_g,
        brand_freq=brand_freq,
        category_freq=category_freq,
        audience_freq=audience_freq,
        sum_p2_brand=float(manifest["sum_p2_brand"]),
        sum_p2_category=float(manifest["sum_p2_category"]),
        sum_p2_audience=float(manifest["sum_p2_audience"]),
        catalog_rating_mean=float(manifest["catalog_rating_mean"]),
        pair_samples=pair_samples,
        mu_transform=mu_transform,
        manifest=manifest,
        product_names=product_names,
        brands=brands,
        categories=categories,
        audiences=audiences,
        image_urls=image_urls,
        names_lower=names_lower,
        hnsw_index=hnsw_index,
        text_nobrand=text_nobrand,
        category_centroids=centroids,
        category_centroid_cos_sample=cos_sample,
        category_centroid_cosine=category_centroid_cosine,
        category_affinity_tables=category_affinity_tables,
        mu_category_affinity=mu_aff,
        active_profile=active_profile,
    )


def compute_mu_transform(pair_samples: dict[str, FloatArray]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for ti in TextImageTransform:
        out[ti.value] = {
            "text": median_imputed(pair_samples["text"], ti),
            "image": median_imputed(pair_samples["image"], ti),
        }
    return out
