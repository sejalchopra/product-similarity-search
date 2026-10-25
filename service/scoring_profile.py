"""Which text matrix and category block variant to use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CategoryMode = Literal["binary", "affinity"]
TextMatrix = Literal["default", "nobrand"]


@dataclass(frozen=True)
class ScoringProfile:
    category_mode: CategoryMode = "binary"
    text_matrix: TextMatrix = "default"

    @staticmethod
    def from_variant(name: str) -> ScoringProfile:
        if name == "A":
            return ScoringProfile(category_mode="affinity", text_matrix="default")
        if name == "B":
            return ScoringProfile(category_mode="binary", text_matrix="nobrand")
        if name == "A+B":
            return ScoringProfile(category_mode="affinity", text_matrix="nobrand")
        return ScoringProfile(category_mode="binary", text_matrix="default")
