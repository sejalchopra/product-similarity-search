#!/usr/bin/env python3
"""1M-vector HNSW benchmark (pure HNSW, not capped union)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "bench_hnsw.json"


def main() -> None:
    import os

    import faiss

    smoke = os.environ.get("SMOKE_BENCH", "").lower() in ("1", "true", "yes")
    n_index = 5_000 if smoke else 1_000_000
    n_query = 100 if smoke else 1_000
    d = 384
    rng = np.random.default_rng(42)
    text_path = ROOT / "artifacts" / "text_embeddings.npy"
    if text_path.exists():
        base = np.load(text_path).astype(np.float32)
        pool = base[rng.integers(0, base.shape[0], size=n_index)]
    else:
        pool = rng.standard_normal((n_index, d)).astype(np.float32)
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    noise_scale = 0.01 / np.sqrt(d)
    noise = rng.standard_normal(pool.shape).astype(np.float32) * noise_scale
    vectors = pool + noise
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    if text_path.exists():
        holdout = base[rng.choice(base.shape[0], size=n_query, replace=False)].astype(np.float32)
    else:
        holdout = vectors[:n_query].copy()

    flat = faiss.IndexFlatIP(d)
    flat.add(vectors)
    hnsw = faiss.IndexHNSWFlat(d, 16)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(vectors)

    results: list[dict[str, float]] = []
    for ef in (16, 32, 64, 128):
        hnsw.hnsw.efSearch = ef
        t0 = time.perf_counter()
        _, ann_ids = hnsw.search(holdout, 10)
        elapsed = time.perf_counter() - t0
        _, exact_ids = flat.search(holdout, 10)
        hits = 0
        for a_row, e_row in zip(ann_ids, exact_ids, strict=False):
            hits += len(set(a_row.tolist()) & set(e_row.tolist()))
        recall = hits / (n_query * 10)
        results.append(
            {
                "efSearch": ef,
                "recall_at_10": recall,
                "p50_ms": elapsed / n_query * 1000,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"results": results, "n_index": n_index}, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
