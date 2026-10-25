"""Optional query-field masks for explore (read-only scoring)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryOverrides:
    mask_brand: bool = False
    mask_image: bool = False
