#!/usr/bin/env python3
"""Copy smoke tuned weights to production path after confirmation."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "artifacts" / "tuned_weights.smoke.json"
PROD = ROOT / "artifacts" / "tuned_weights.json"


def main() -> None:
    if not SMOKE.exists():
        raise SystemExit(f"Missing {SMOKE}; run make eval-smoke first")
    data = json.loads(SMOKE.read_text())
    print("Source:", SMOKE)
    print("  source:", data.get("source"))
    print("  split:", json.dumps(data.get("split"), indent=2))
    print(
        "  variant:", data.get("selected_variant"), "transform:", data.get("text_image_transform")
    )
    if os.environ.get("CONFIRM", "").strip() != "1":
        raise SystemExit("Refusing to copy without CONFIRM=1")
    PROD.write_text(SMOKE.read_text())
    print(f"Wrote {PROD}")


if __name__ == "__main__":
    main()
