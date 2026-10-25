# Design note

All metrics below are copied from [`eval.json`](eval.json) (`make eval`). Do not hand-edit numbers in the eval region.

**Eval tiers:** `make eval-smoke` (400/200/200 queries) and `make eval-medium` (default 1500/1500/2000, capped, writes `docs/eval.medium.json` only) use the same pipeline as `make eval` but do not overwrite `docs/eval.json`. Pool stages resume from `artifacts/eval_cache/`. Only `make eval` refreshes this document.

<!-- current_state:start -->
## Current state (from eval.json)

_Run `make eval` to populate this block from `docs/eval.json`._
<!-- current_state:end -->

<!-- eval:start -->
## Problem and data

See `eval.json` after `make eval`.
<!-- eval:end -->

## Latency (exact path)

From `eval.json` → `latency_exact` (500-query benchmark on loaded artifacts). Reports lean `/find_similar_products` vs full-catalog scoring with per-candidate metadata.

In-process LRU cache backs `/find_similar_products`.

## ANN

- **Online:** `SEARCH_MODE=ann` uses capped union (HNSW/text top-500 ∪ same-brand ∪ same-category), then the same fusion scorer.
- **Bench:** `make bench` runs pure HNSW on 1M bootstrap MiniLM vectors (`docs/bench_hnsw.json`). Union-at-1M is not measured.
- Reference: Malkov & Yashunin, HNSW (arXiv:1603.09320).

## Deploy

- `docker build` copies `artifacts/` (no runtime fetch).
- [`k8s.yaml`](../k8s.yaml): Deployment + Service, probes, `imagePullPolicy: IfNotPresent`.
- **Stretch:** `docker-compose.yml` + `scripts/load_qdrant.py` (Qdrant named vectors).

## Limitations

- Image crawl coverage is 0.995 (29,861 / 30,000). That is above the 5% floor, so tuning does not force the image weight to 0. The weight stays 0 only while coverage is under 5%.
- Co-purchase labels are title-string matched (noisier than ASIN joins).
- Low pair-active rate on weight limits that block’s discriminative power.
