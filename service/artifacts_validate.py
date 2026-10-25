"""Validate tuned variant vs manifest artifact shas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from service.scoring_profile import ScoringProfile


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def variant_needs_nobrand(variant: str) -> bool:
    return variant in ("B", "A+B")


def variant_needs_centroids(variant: str) -> bool:
    return variant in ("A", "A+B")


def validate_variant_artifacts(
    artifacts_dir: Path,
    manifest: dict[str, object],
    variant: str,
) -> None:
    prof = ScoringProfile.from_variant(variant)
    if prof.text_matrix == "nobrand" or variant_needs_nobrand(variant):
        path = artifacts_dir / "text_embeddings_nobrand.npy"
        key = "text_nobrand_sha256"
        if not path.is_file():
            raise RuntimeError(f"tuned_weights selected_variant={variant} requires {path.name}")
        expected = manifest.get(key)
        if not isinstance(expected, str):
            raise RuntimeError(f"manifest missing {key} for nobrand variant")
        if _sha256_file(path) != expected:
            raise RuntimeError(f"sha256 mismatch for {path.name}")
    if prof.category_mode == "affinity" or variant_needs_centroids(variant):
        path = artifacts_dir / "category_centroids.npy"
        key = "category_centroids_sha256"
        if not path.is_file():
            raise RuntimeError(f"tuned_weights selected_variant={variant} requires {path.name}")
        expected = manifest.get(key)
        if not isinstance(expected, str):
            raise RuntimeError(f"manifest missing {key} for affinity variant")
        if _sha256_file(path) != expected:
            raise RuntimeError(f"sha256 mismatch for {path.name}")


def validate_tuned_weights_file(artifacts_dir: Path) -> None:
    tuned = artifacts_dir / "tuned_weights.json"
    manifest_path = artifacts_dir / "manifest.json"
    if not tuned.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    tw = json.loads(tuned.read_text())
    variant = str(tw.get("selected_variant", manifest.get("selected_variant", "binary")))
    validate_variant_artifacts(artifacts_dir, manifest, variant)
