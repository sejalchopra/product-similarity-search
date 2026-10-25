#!/usr/bin/env python3
"""Stretch: upsert artifacts into Qdrant (optional)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.parse_args()
    raise SystemExit(
        "Qdrant loader is stretch-only; implement after single-container path is green."
    )


if __name__ == "__main__":
    main()
