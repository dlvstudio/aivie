#!/usr/bin/env python3
"""Chuẩn hóa mọi nguồn → data/text_corpus/*.jsonl (mỗi dòng một JSON string)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.training.normalize_corpus import OUT_DIR, normalize_all


def main() -> int:
    stats = normalize_all()
    print(f"✓ text_corpus → {OUT_DIR}")
    for name, n in stats.items():
        print(f"  {name}: {n} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
