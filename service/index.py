"""FAISS HNSW helper (typed adapter)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import faiss


def build_hnsw_index(vectors: NDArray[np.float32]) -> faiss.Index:
    import faiss

    d = vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, 16)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(vectors)
    return index
