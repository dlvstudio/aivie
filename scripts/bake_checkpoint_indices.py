#!/usr/bin/env python3
"""
Bake skip spans + seq index (+ FRAME_BRACKET nếu thiếu) vào checkpoint.

Chạy một lần trên checkpoint production → lần sau chỉ load, không mine lại (~15 phút).

  PYTHONPATH=src python scripts/bake_checkpoint_indices.py \\
    checkpoints/network_v10_histwiki_paragraph.pkl

  # ghi đè tại chỗ (atomic rename):
  PYTHONPATH=src python scripts/bake_checkpoint_indices.py ckpt.pkl --in-place

  # re-mine skip spans dù đã có:
  PYTHONPATH=src python scripts/bake_checkpoint_indices.py ckpt.pkl --force-skip

  # re-mine FRAME_BRACKET dù đã có (chỉ cần khi muốn ép; checkpoint thiếu
  # hlink_endpoints — bake trước TK_037 — nay tự re-mine, không cần cờ này):
  PYTHONPATH=src python scripts/bake_checkpoint_indices.py ckpt.pkl --force-frame
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.training.checkpoint_bake import bake_derived_indices
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Bake derived indices into checkpoint .pkl")
    ap.add_argument("checkpoint", help="đường dẫn .pkl nguồn")
    ap.add_argument(
        "--out",
        default=None,
        help="đích (mặc định: <stem>_baked.pkl hoặc --in-place)",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="ghi đè checkpoint nguồn (qua file .baking tạm)",
    )
    ap.add_argument(
        "--force-skip",
        action="store_true",
        help="re-mine skip spans kể cả checkpoint đã có",
    )
    ap.add_argument(
        "--force-frame",
        action="store_true",
        help="re-mine FRAME_BRACKET kể cả checkpoint đã có layer + meta + endpoints",
    )
    ap.add_argument("--no-frame", action="store_true", help="bỏ mine FRAME_BRACKET")
    ap.add_argument("--no-skip", action="store_true", help="bỏ mine skip spans")
    ap.add_argument("--no-seq", action="store_true", help="bỏ build seq index")
    args = ap.parse_args()

    src = Path(args.checkpoint)
    if not src.is_absolute():
        src = _ROOT / src
    if not src.exists():
        print(f"Không tìm thấy: {src}", file=sys.stderr)
        return 2

    if args.in_place:
        dst = src.with_suffix(src.suffix + ".baking")
        final = src
    elif args.out:
        dst = Path(args.out)
        if not dst.is_absolute():
            dst = _ROOT / dst
        final = dst
    else:
        dst = src.with_name(src.stem + "_baked.pkl")
        final = dst

    log = get_run_logger(f"bake_{src.stem}")
    t0 = time.time()
    state, cache = load_checkpoint(src, logger=log)

    with log.phase("bake derived indices"):
        stats = bake_derived_indices(
            state,
            logger=log,
            skip_spans=not args.no_skip,
            seq_index=not args.no_seq,
            frame_brackets=not args.no_frame,
            force_skip_spans=args.force_skip,
            force_frame_brackets=args.force_frame,
        )
        log.log(f"stats: {stats}")

    with log.phase(f"save → {final.name}"):
        save_checkpoint(state, dst, cache)
        if args.in_place:
            os.replace(dst, final)
    log.log(f"✓ XONG ({time.time() - t0:.1f}s) → {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
