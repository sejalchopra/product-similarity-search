#!/usr/bin/env python3
"""Warm demo endpoints and sample /find_similar_products latency."""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.error
import urllib.request

WARM_PATHS = ("/", "/meta", "/eval", "/health", "/ready")
WARM_QUERY = "26d41bdc1495de290bc8e6062d927729"
WARMED_IDS = [WARM_QUERY] * 5
COLD_IDS = [
    "ad8a5a196d515ef09dfdaf082bdc37c4",
    "72c61c06a56440e0d018cd61d5bc28c9",
    "12d33c127417b634899aa4c7cbe747f1",
    "b3a208523ad80bc669497d5a2d050cd1",
    "26d41bdc1495de290bc8e6062d927729",
]


def _get(base: str, path: str) -> float:
    url = base.rstrip("/") + path
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=120) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def _find_similar(base: str, product_id: str, n: int = 10) -> float:
    path = f"/find_similar_products?product_id={product_id}&num_similar={n}"
    return _get(base, path)


def main() -> None:
    p = argparse.ArgumentParser(description="Warm UI/API and time find_similar_products")
    p.add_argument("--base", default="http://127.0.0.1:8000", help="Service base URL")
    args = p.parse_args()
    base = args.base

    print(f"Warming {base} …")
    for path in WARM_PATHS:
        try:
            ms = _get(base, path)
            print(f"  {path}: {ms:.1f} ms")
        except urllib.error.URLError as e:
            raise SystemExit(f"warm failed {path}: {e}") from e
    _find_similar(base, WARM_QUERY)

    warmed_ms: list[float] = []
    for pid in WARMED_IDS:
        warmed_ms.append(_find_similar(base, pid))
    cold_ms: list[float] = []
    for pid in COLD_IDS:
        cold_ms.append(_find_similar(base, pid))

    print("\n/find_similar_products (num_similar=10), ms:")
    wvals = [round(x, 1) for x in warmed_ms]
    cvals = [round(x, 1) for x in cold_ms]
    print(f"  warmed (repeat):  mean={statistics.mean(warmed_ms):.1f}  values={wvals}")
    print(f"  cold (distinct):  mean={statistics.mean(cold_ms):.1f}  values={cvals}")


if __name__ == "__main__":
    main()
