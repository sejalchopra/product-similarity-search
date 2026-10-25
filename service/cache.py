"""In-process LRU cache of top-50 results per query."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def lru_cache_topk(maxsize: int) -> Callable[[Callable[..., list[str]]], Callable[..., list[str]]]:
    def decorator(fn: Callable[..., list[str]]) -> Callable[..., list[str]]:
        @functools.lru_cache(maxsize=maxsize)
        def cached(
            product_id: str,
            mode: str,
            num_similar_cap: int,
        ) -> tuple[str, ...]:
            return tuple(fn(product_id, num_similar_cap))

        def wrapper(product_id: str, num_similar: int, mode: str = "exact") -> list[str]:
            top = cached(product_id, mode, 50)
            return list(top[:num_similar])

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator
