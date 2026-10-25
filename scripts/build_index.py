#!/usr/bin/env python3
"""Build artifacts from the Amazon fashion ldjson catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from service.brand_backfill import apply_brand_backfill, brand_backfill_enabled
from service.catalog import (
    build_category_centroids,
    compute_mu_transform,
    sample_category_centroid_cosines,
)
from service.category_affinity import build_affinity_tables, build_centroid_cosine_matrix
from service.cleaning import (
    assign_title_groups,
    child_category_key,
    normalize_title_for_group,
    parse_brand,
    parse_color_tokens,
    parse_price,
    parse_rating,
    parse_review_count,
    parse_weight_grams,
    resolve_audience,
)
from service.scoring import AUDIENCE_CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
LDJSON_NAME = (
    "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
)
PAIR_SAMPLE_SIZE = 200_000
IMAGE_TIMEOUT = 15


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_ldjson() -> Path:
    path = DATA_DIR / LDJSON_NAME
    if path.exists():
        return path
    zpath = DATA_DIR / "archive.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(DATA_DIR)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return path


def load_frame(path: Path) -> pl.DataFrame:
    return pl.read_ndjson(path, infer_schema_length=None)


def build_color_vocab(df: pl.DataFrame) -> frozenset[str]:
    vocab: set[str] = set()
    for row in df.iter_rows(named=True):
        tokens = parse_color_tokens(row.get("colour"), row.get("product_name") or "")
        vocab.update(tokens)
    return frozenset(vocab)


def encode_metadata(
    df: pl.DataFrame, color_vocab: frozenset[str]
) -> tuple[pl.DataFrame, list[str], dict[str, int], dict[str, int]]:
    vocab_list = sorted(color_vocab)
    vocab_index = {c: i for i, c in enumerate(vocab_list)}
    rows: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        details = row.get("product_details__k_v_pairs") or {}
        if not isinstance(details, dict):
            details = {}
        child = child_category_key(
            row.get("sales_rank_in_child_category"),
            row.get("parent___child_category__all"),
        )
        aud = resolve_audience(child, row.get("product_name") or "")
        colors = parse_color_tokens(row.get("colour"), row.get("product_name") or "")
        inds = [vocab_index[c] for c in colors if c in vocab_index]
        vals = [1.0] * len(inds)
        norm_title = normalize_title_for_group(row.get("product_name") or "")
        rows.append(
            {
                "uniq_id": row["uniq_id"],
                "product_name": row.get("product_name") or "",
                "brand": parse_brand(row.get("brand")),
                "category_child": child,
                "audience": aud,
                "normalized_title": norm_title,
                "color_indices": inds,
                "color_values": vals,
                "color_missing": len(inds) == 0,
                "price": parse_price(row.get("sales_price")),
                "rating": parse_rating(row.get("rating")),
                "rating_n": parse_review_count(row.get("no__of_reviews"), details),
                "weight_g": parse_weight_grams(row.get("weight"), details),
                "medium_url": (row.get("medium") or "").split("|")[0].strip()
                if row.get("medium")
                else "",
            }
        )
    meta = pl.DataFrame(rows)
    if brand_backfill_enabled():
        filled, _report = apply_brand_backfill(
            meta["product_name"].to_list(), meta["brand"].to_list()
        )
        meta = meta.with_columns(pl.Series("brand", filled))
    brands = sorted({b for b in meta["brand"].to_list() if b})
    brand_map = {b: i for i, b in enumerate(brands)}
    cats = sorted({c for c in meta["category_child"].to_list() if c})
    cat_map = {c: i for i, c in enumerate(cats)}
    aud_map = {c: i for i, c in enumerate(AUDIENCE_CLASSES)}
    brand_codes = [brand_map.get(b, -1) if b else -1 for b in meta["brand"].to_list()]
    tg = assign_title_groups(meta["normalized_title"].to_list(), brand_codes)
    cat_codes = [cat_map.get(c, -1) if c else -1 for c in meta["category_child"].to_list()]
    aud_codes = [aud_map.get(a, 7) for a in meta["audience"].to_list()]
    meta = meta.with_columns(
        pl.Series("brand_code", brand_codes, dtype=pl.Int32),
        pl.Series("category_code", cat_codes, dtype=pl.Int32),
        pl.Series("audience_code", aud_codes, dtype=pl.Int32),
        pl.Series("title_group", tg, dtype=pl.Int32),
        pl.col("price").cast(pl.Float64),
        pl.col("rating").cast(pl.Float64),
        pl.col("weight_g").cast(pl.Float64),
    )
    return meta, vocab_list, brand_map, cat_map


def sample_pair_sims(
    text: np.ndarray,
    image: np.ndarray,
    image_mask: np.ndarray,
    meta: pl.DataFrame,
    color_csr: Any,
    rng: random.Random,
) -> dict[str, np.ndarray]:
    n = text.shape[0]
    samples: dict[str, list[float]] = {
        "text": [],
        "image": [],
        "color": [],
        "price": [],
        "rating": [],
        "weight": [],
    }
    price = meta["price"].to_numpy()
    rating = meta["rating"].to_numpy()
    rating_n = meta["rating_n"].to_numpy()
    weight_g = meta["weight_g"].to_numpy()
    color_missing = meta["color_missing"].to_numpy()
    catalog_mean = float(np.nanmean(rating[np.isfinite(rating)]))

    while len(samples["text"]) < PAIR_SAMPLE_SIZE:
        i = rng.randrange(n)
        j = rng.randrange(n)
        samples["text"].append(float(text[i] @ text[j]))
        if image_mask[i] and image_mask[j]:
            samples["image"].append(float(image[i] @ image[j]))
        if not color_missing[i] and not color_missing[j]:
            dot = float((color_csr[i] @ color_csr[j].T).toarray()[0, 0])
            samples["color"].append(dot)
        pi, pj = price[i], price[j]
        if np.isfinite(pi) and np.isfinite(pj) and pi > 0 and pj > 0:
            samples["price"].append(float(np.exp(-abs(np.log(pi) - np.log(pj)) / 0.5)))
        if rating_n[i] > 0 and rating_n[j] > 0:
            from service.scoring import rating_similarity, shrink_rating

            r1 = shrink_rating(float(rating[i]), int(rating_n[i]), catalog_mean)
            r2 = shrink_rating(float(rating[j]), int(rating_n[j]), catalog_mean)
            samples["rating"].append(rating_similarity(r1, r2))
        wi, wj = weight_g[i], weight_g[j]
        if np.isfinite(wi) and np.isfinite(wj) and wi > 0 and wj > 0:
            from service.scoring import weight_similarity

            samples["weight"].append(weight_similarity(float(wi), float(wj)))

    return {k: np.array(v[:PAIR_SAMPLE_SIZE], dtype=np.float64) for k, v in samples.items()}


def _image_cache_file(cache_dir: Path, url: str) -> Path:
    return cache_dir / hashlib.sha256(url.encode()).hexdigest()


def _valid_cached_image(path: Path) -> bool:
    from PIL import Image

    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def crawl_image_urls(
    urls: list[str],
    cache_dir: Path,
    *,
    workers: int = 16,
    timeout_sec: float = 8.0,
    time_limit_sec: float = 90 * 60,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Download images only. Resumes from cache. Does not encode embeddings."""
    import threading
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests
    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_path or (ARTIFACTS / "image_crawl.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    status_path = ARTIFACTS / "image_crawl_status.json"
    reasons: Counter[str] = Counter()
    http_status: Counter[str] = Counter()
    lock = threading.Lock()
    done = 0
    n = len(urls)
    deadline = time.monotonic() + time_limit_sec
    started = time.monotonic()

    def _log(msg: str) -> None:
        line = f"[image-crawl {time.monotonic() - started:7.1f}s] {msg}"
        print(line, flush=True)
        with log_file.open("a") as fh:
            fh.write(line + "\n")

    def _one(i: int, url: str) -> tuple[int, str]:
        if time.monotonic() > deadline:
            return i, "time_limit"
        if not url:
            return i, "empty_url"
        cache_file = _image_cache_file(cache_dir, url)
        if _valid_cached_image(cache_file):
            return i, "cache_hit"
        if cache_file.exists():
            cache_file.unlink(missing_ok=True)
        try:
            resp = requests.get(url, timeout=timeout_sec)
        except requests.Timeout:
            return i, "timeout"
        except requests.RequestException as exc:
            return i, f"request_{type(exc).__name__}"
        code = str(resp.status_code)
        with lock:
            http_status[code] += 1
        if resp.status_code != 200:
            return i, f"http_{resp.status_code}"
        try:
            img = Image.open(__import__("io").BytesIO(resp.content)).convert("RGB")
            img.save(cache_file, format="JPEG")
        except Exception as exc:
            cache_file.unlink(missing_ok=True)
            return i, f"invalid_image_{type(exc).__name__}"
        if not _valid_cached_image(cache_file):
            cache_file.unlink(missing_ok=True)
            return i, "invalid_image"
        return i, "downloaded"

    _log(f"start n={n} workers={workers} timeout={timeout_sec}s limit={time_limit_sec:.0f}s")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, i, url) for i, url in enumerate(urls)]
        for fut in as_completed(futures):
            _i, reason = fut.result()
            with lock:
                reasons[reason] += 1
                done += 1
                snapshot_done = done
            if snapshot_done % 200 == 0 or snapshot_done == n:
                cov = (reasons["downloaded"] + reasons["cache_hit"]) / max(n, 1)
                _log(
                    f"{snapshot_done}/{n} cached_or_downloaded="
                    f"{reasons['downloaded'] + reasons['cache_hit']} coverage={cov:.3f} "
                    f"http={dict(http_status)}"
                )
                status_path.write_text(
                    json.dumps(
                        {
                            "completed": snapshot_done,
                            "rows": n,
                            "reasons": dict(reasons),
                            "http_status": dict(http_status),
                            "elapsed_sec": time.monotonic() - started,
                        },
                        indent=2,
                    )
                )
    elapsed = time.monotonic() - started
    usable = reasons["downloaded"] + reasons["cache_hit"]
    report = {
        "rows": n,
        "usable_images": usable,
        "image_coverage": usable / max(n, 1),
        "reasons": dict(reasons),
        "http_status": dict(http_status),
        "elapsed_sec": elapsed,
        "time_limit_sec": time_limit_sec,
        "workers": workers,
        "timeout_sec": timeout_sec,
    }
    status_path.write_text(json.dumps(report, indent=2))
    _log(f"done usable={usable} coverage={report['image_coverage']:.3f} reasons={dict(reasons)}")
    return report


def encode_images_from_cache(
    urls: list[str],
    cache_dir: Path,
    *,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """CLIP-encode cached JPEGs. Rows without a valid cache file stay masked off."""
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    n = len(urls)
    embeddings = np.zeros((n, 512), dtype=np.float16)
    mask = np.zeros(n, dtype=bool)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    import torch

    batch_imgs: list[Any] = []
    batch_idx: list[int] = []

    def _flush() -> None:
        if not batch_imgs:
            return
        inputs = processor(images=batch_imgs, return_tensors="pt")
        with torch.inference_mode():
            feats = model.get_image_features(**inputs)
            if not torch.is_tensor(feats):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().numpy().astype(np.float16)
        for row_i, vec in zip(batch_idx, arr, strict=True):
            embeddings[row_i] = vec
            mask[row_i] = True
        for img in batch_imgs:
            img.close()
        batch_imgs.clear()
        batch_idx.clear()

    for i, url in enumerate(urls):
        if not url:
            continue
        path = _image_cache_file(cache_dir, url)
        if not _valid_cached_image(path):
            continue
        try:
            batch_imgs.append(Image.open(path).convert("RGB"))
            batch_idx.append(i)
        except Exception:
            continue
        if len(batch_imgs) >= batch_size:
            _flush()
        if (i + 1) % 500 == 0:
            print(f"image encode {i + 1}/{n} coverage {mask.mean():.3f}", flush=True)
    _flush()
    print(f"image encode done coverage {float(mask.mean()):.3f}", flush=True)
    return embeddings, mask


def crawl_images(
    urls: list[str],
    out_path: Path,
    mask_path: Path,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy helper: download then encode. Prefer --images-only for the timed crawl."""
    crawl_image_urls(urls, cache_dir, timeout_sec=IMAGE_TIMEOUT, time_limit_sec=10**9)
    image, mask = encode_images_from_cache(urls, cache_dir)
    np.save(out_path, image)
    np.save(mask_path, mask)
    return image, mask


def encode_text(names: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(names, normalize_embeddings=True, show_progress_bar=True)
    return emb.astype(np.float16)


def _build_color_csr(meta: pl.DataFrame, vocab_size: int) -> Any:
    from scipy import sparse

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, row in enumerate(meta.iter_rows(named=True)):
        for j, v in zip(row["color_indices"], row["color_values"], strict=False):
            rows.append(i)
            cols.append(int(j))
            data.append(float(v))
    n = meta.height
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, vocab_size))


def _write_artifacts(
    meta: pl.DataFrame,
    vocab_list: list[str],
    brand_map: dict[str, int],
    cat_map: dict[str, int],
    text_f32: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    text_path: Path,
    image_path: Path,
) -> None:
    names = meta["product_name"].to_list()
    uniq_ids = meta["uniq_id"].to_list()
    n = meta.height

    color_csr = _build_color_csr(meta, len(vocab_list))
    image_f32 = image.astype(np.float32)
    pair_samples = sample_pair_sims(text_f32, image_f32, mask, meta, color_csr, random.Random(42))
    mu_transform = compute_mu_transform(pair_samples)
    np.savez(ARTIFACTS / "cdf.npz", **pair_samples)

    brand_freq = np.zeros(len(brand_map), dtype=np.float64)
    for bc in meta["brand_code"].to_list():
        if bc >= 0:
            brand_freq[bc] += 1
    brand_freq /= max(n, 1)
    cat_freq = np.zeros(len(cat_map), dtype=np.float64)
    for cc in meta["category_code"].to_list():
        if cc >= 0:
            cat_freq[cc] += 1
    cat_freq /= max(n, 1)
    aud_freq = np.zeros(len(AUDIENCE_CLASSES), dtype=np.float64)
    for ac in meta["audience_code"].to_list():
        aud_freq[ac] += 1
    aud_freq /= max(n, 1)

    meta.write_parquet(ARTIFACTS / "metadata.parquet")
    cat_codes = meta["category_code"].to_numpy().astype(np.int32)
    centroids = build_category_centroids(text_f32.astype(np.float64), cat_codes, len(cat_map))
    cent_path = ARTIFACTS / "category_centroids.npy"
    np.save(cent_path, centroids.astype(np.float32))
    cos_sample = pair_samples.get("category_centroid_cos")
    if cos_sample is None:
        cos_sample = sample_category_centroid_cosines(
            centroids, cat_codes, np.random.default_rng(42)
        )
    cos_mat = build_centroid_cosine_matrix(centroids)
    cos_mat_path = ARTIFACTS / "category_centroid_cosine.npy"
    np.save(cos_mat_path, cos_mat.astype(np.float32))
    aff_tables = build_affinity_tables(centroids, cos_sample)
    aff_path = ARTIFACTS / "category_affinity_calibrated.npz"
    np.savez(aff_path, **aff_tables)
    manifest = {
        "text_model": "all-MiniLM-L6-v2",
        "image_model": "openai/clip-vit-base-patch32",
        "text_dim": 384,
        "image_dim": 512,
        "row_count": n,
        "text_sha256": _sha256_file(text_path),
        "image_sha256": _sha256_file(image_path),
        "image_coverage": float(mask.mean()),
        "color_vocab_size": len(vocab_list),
        "brand_freq": brand_freq.tolist(),
        "category_freq": cat_freq.tolist(),
        "audience_freq": aud_freq.tolist(),
        "sum_p2_brand": float(np.sum(brand_freq**2)),
        "sum_p2_category": float(np.sum(cat_freq**2)),
        "sum_p2_audience": float(np.sum(aud_freq**2)),
        "catalog_rating_mean": float(np.nanmean(meta["rating"].to_numpy().astype(float))),
        "mu_transform": mu_transform,
        "category_centroids_sha256": _sha256_file(cent_path),
        "category_centroid_cosine_sha256": _sha256_file(cos_mat_path),
        "category_affinity_calibrated_sha256": _sha256_file(aff_path),
        "names": {uid: name for uid, name in zip(uniq_ids, names, strict=False)},
    }
    with (ARTIFACTS / "manifest.json").open("w") as f:
        json.dump(manifest, f)
    print(f"Wrote artifacts for {n} rows, image coverage {manifest['image_coverage']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--max-image", type=int, default=0)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Rebuild metadata.parquet, cdf, and manifest; keep existing embeddings",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Crawl image URLs, then encode CLIP embeddings and refresh the manifest",
    )
    parser.add_argument("--image-workers", type=int, default=16)
    parser.add_argument("--image-timeout", type=float, default=8.0)
    parser.add_argument("--image-time-limit-sec", type=float, default=90 * 60)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.images_only:
        meta = pl.read_parquet(ARTIFACTS / "metadata.parquet")
        urls = meta["medium_url"].to_list()
        cache_dir = ARTIFACTS / "image_cache"
        crawl_image_urls(
            urls,
            cache_dir,
            workers=args.image_workers,
            timeout_sec=args.image_timeout,
            time_limit_sec=args.image_time_limit_sec,
        )
        image, mask = encode_images_from_cache(urls, cache_dir)
        image_path = ARTIFACTS / "image_embeddings.npy"
        mask_path = ARTIFACTS / "image_mask.npy"
        text_path = ARTIFACTS / "text_embeddings.npy"
        np.save(image_path, image)
        np.save(mask_path, mask)
        text_f32 = np.load(text_path).astype(np.float32)
        manifest_path = ARTIFACTS / "manifest.json"
        with manifest_path.open() as f:
            old_manifest = json.load(f)
        vocab_size = int(old_manifest.get("color_vocab_size", 1))
        brand_n = max(int(meta["brand_code"].max()) + 1, 1)
        cat_n = max(int(meta["category_code"].max()) + 1, 1)
        _write_artifacts(
            meta,
            [str(i) for i in range(vocab_size)],
            {str(i): i for i in range(brand_n)},
            {str(i): i for i in range(cat_n)},
            text_f32,
            image,
            mask,
            text_path,
            image_path,
        )
        return

    ldjson = ensure_ldjson()
    df = load_frame(ldjson)
    color_vocab = build_color_vocab(df)
    meta, vocab_list, brand_map, cat_map = encode_metadata(df, color_vocab)
    names = meta["product_name"].to_list()
    uniq_ids = meta["uniq_id"].to_list()
    n = len(uniq_ids)

    text_path = ARTIFACTS / "text_embeddings.npy"
    image_path = ARTIFACTS / "image_embeddings.npy"
    mask_path = ARTIFACTS / "image_mask.npy"

    if args.metadata_only:
        if not text_path.exists() or not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(
                "metadata-only rebuild requires text_embeddings.npy, "
                "image_embeddings.npy, and image_mask.npy in artifacts/"
            )
        old_meta_path = ARTIFACTS / "metadata.parquet"
        if old_meta_path.exists():
            old_ids = pl.read_parquet(old_meta_path)["uniq_id"].to_list()
            if old_ids != uniq_ids:
                raise RuntimeError("uniq_id order changed in ldjson; run a full index build")
        text_f32 = np.load(text_path).astype(np.float32)
        image = np.load(image_path)
        mask = np.load(mask_path).astype(bool)
        if text_f32.shape[0] != n:
            raise RuntimeError("embedding row count does not match catalog")
        _write_artifacts(
            meta, vocab_list, brand_map, cat_map, text_f32, image, mask, text_path, image_path
        )
        return

    text_f32 = encode_text(names).astype(np.float32)
    np.save(text_path, text_f32.astype(np.float16))

    urls = meta["medium_url"].to_list()
    if args.max_image > 0:
        urls = urls[: args.max_image]
        meta = meta.head(args.max_image)
        n = meta.height
        text_f32 = text_f32[:n]

    if args.skip_images:
        image = np.zeros((n, 512), dtype=np.float16)
        mask = np.zeros(n, dtype=bool)
    else:
        image, mask = crawl_images(
            urls,
            image_path,
            mask_path,
            ARTIFACTS / "image_cache",
        )
    np.save(image_path, image)
    np.save(mask_path, mask)

    _write_artifacts(
        meta, vocab_list, brand_map, cat_map, text_f32, image, mask, text_path, image_path
    )


if __name__ == "__main__":
    main()
