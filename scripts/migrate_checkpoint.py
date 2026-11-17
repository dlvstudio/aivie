#!/usr/bin/env python3
"""
TK_011 — Migrate checkpoint sang format serialize MẢNG NUMPY (load nhanh).

Re-save adjacency (cạnh) + seg-graph (link ISG) dưới dạng cột numpy thay key-chuỗi/list-dict.
LOSSLESS: đồ thị + output token-identical (đã kiểm chứng). Ghi ra file tạm rồi ĐỔI TÊN nguyên
tử → không hỏng checkpoint nếu gián đoạn. Checkpoint format cũ vẫn load được (back-compat), nên
migrate là tùy chọn — chỉ để nhanh hơn lúc load.

  python scripts/migrate_checkpoint.py checkpoints/grade_v3_big_a2_t3_isg.pkl [thêm...]
  python scripts/migrate_checkpoint.py --all           # mọi checkpoints/*.pkl (bỏ .content)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.util.progress import get_run_logger

ROOT = Path(__file__).resolve().parents[1]


def migrate(path: Path, log) -> None:
    size0 = path.stat().st_size / 1e6
    t = time.time()
    state, cache = load_checkpoint(str(path), logger=log)
    load_s = time.time() - t
    tmp = path.with_suffix(path.suffix + ".migrating")
    t = time.time()
    save_checkpoint(state, str(tmp), cache, bake_derived=True, logger=log)
    save_s = time.time() - t
    # đổi tên nguyên tử (an toàn nếu gián đoạn)
    os.replace(tmp, path)
    size1 = path.stat().st_size / 1e6
    log.log(f"{path.name}: load {load_s:.1f}s save {save_s:.1f}s  {size0:.0f}→{size1:.0f}MB")


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate checkpoint → numpy-array format (fast load)")
    ap.add_argument("checkpoints", nargs="*", help="đường dẫn .pkl")
    ap.add_argument("--all", action="store_true", help="mọi checkpoints/*.pkl")
    args = ap.parse_args()

    if args.all:
        paths = sorted(
            p for p in (ROOT / "checkpoints").glob("*.pkl")
            if not p.name.endswith(".content.pkl")
        )
    else:
        paths = [Path(p) for p in args.checkpoints]
    if not paths:
        print("Không có checkpoint nào — truyền đường dẫn hoặc --all", file=sys.stderr)
        return 2

    log = get_run_logger("migrate_checkpoint")
    for p in paths:
        if not p.exists():
            log.log(f"BỎ QUA (không tồn tại): {p}")
            continue
        migrate(p, log)
    log.log(f"Xong {len(paths)} checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
