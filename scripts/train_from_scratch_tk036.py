#!/usr/bin/env python3
"""DEPRECATED — dùng scripts/train.py."""
import sys
print("DEPRECATED: dùng PYTHONPATH=src python scripts/train.py", file=sys.stderr)
print("  normalize: scripts/normalize_corpus.py", file=sys.stderr)
print("  full:      scripts/train.py --manifest config/train_v10.yaml --out ...", file=sys.stderr)
raise SystemExit(1)
