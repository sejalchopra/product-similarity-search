from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from service.artifacts_validate import validate_variant_artifacts


def test_validate_centroids_requires_sha_match(tmp_path: Path) -> None:
    art = tmp_path / "artifacts"
    art.mkdir()
    cent = art / "category_centroids.npy"
    np.save(cent, np.zeros((2, 4), dtype=np.float32))
    import hashlib

    h = hashlib.sha256(cent.read_bytes()).hexdigest()
    manifest = {"category_centroids_sha256": h}
    validate_variant_artifacts(art, manifest, "A")
    manifest["category_centroids_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        validate_variant_artifacts(art, manifest, "A")


def test_binary_skips_nobrand(tmp_path: Path) -> None:
    validate_variant_artifacts(tmp_path, {}, "binary")
