"""Disk cache for eval stages keyed by config hash."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "artifacts" / "eval_cache"


def force_refresh() -> bool:
    return os.environ.get("EVAL_FORCE", "").strip() in ("1", "true", "yes")


def config_hash(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_path(stage: str, cfg_hash: str) -> Path:
    return CACHE_DIR / f"{stage}_{cfg_hash}.pkl"


def load_stage(stage: str, cfg_hash: str) -> Any | None:
    if force_refresh():
        return None
    path = cache_path(stage, cfg_hash)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def save_stage(stage: str, cfg_hash: str, obj: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(stage, cfg_hash)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
