#!/usr/bin/env python3
"""
DUMP slot-hub đã induce — để chọn KHUNG probe P3.1 từ hub THẬT (TK_044).

    ⚠️ CHẠY M4. PYTHONPATH=src python -u scripts/dump_slot_hubs.py \
        --checkpoint checkpoints/network_v11.pkl --top 60 --log-file out/slot_hubs_dump.log

## Vì sao (đọc RUNBOOK_TK_044_P3.1.md §1)

SlotHub = slot-signature của POC ĐÃ CÓ SẴN: `signature=(left,right)` + `fillers` (thực thể thay slot)
+ `right` = chính `after_role` (token cấu-trúc sau slot). P3.1 nối 2 hub đồng-chỉ. NHƯNG binding chỉ
có chỗ neo nếu KHUNG probe TRÙNG một hub thật (≥2 filler phân biệt). Dump này để M4:
  1. Xem graph có hub nào (khung + filler + #filler);
  2. DỰNG LẠI `bind_probe.jsonl` từ khung hub THẬT (điền thực-thể MỚI vào slot) — không bịa khung
     mà graph chưa học (P3.0 dùng khung tuỳ ý → không hub → không thể bind, đó là lý do nền=0 kép:
     vừa thiếu kênh copy, vừa thiếu cả slot).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--top", type=int, default=60, help="số hub in (xếp theo #filler giảm dần)")
    ap.add_argument("--min-fillers", type=int, default=3)
    ap.add_argument("--log-file", default="out/slot_hubs_dump.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True); logf.write(str(m) + "\n"); logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    ps, vocab = state.parents, state.vocab
    sur = vocab.token_of
    hubs = list(getattr(ps, "hubs", []) or [])
    log(f"[{time.time()-t0:.0f}s] ✓ loaded · hubs induced = {len(hubs)}")
    if not hubs:
        log("⚠ CHƯA có hub (induce_slot_hubs chưa chạy / chưa persist). Gọi ps.induce_slot_hubs() "
            "trong build rồi lưu lại checkpoint, hoặc induce tại đây trước khi dump.")
        return 1

    def s(ids):
        return " ".join(sur(t) for t in ids) or "∅"

    hubs.sort(key=lambda h: len(h.fillers), reverse=True)
    shown = 0
    log(f"\n═══ TOP hub (≥{args.min_fillers} filler) — khung `left [SLOT] right` · filler mẫu ═══")
    for h in hubs:
        if len(h.fillers) < args.min_fillers:
            continue
        top_f = sorted(h.fillers.items(), key=lambda kv: -kv[1])[:6]
        fillers = ", ".join(f"{sur(f)!r}×{p}" for f, p in top_f)
        log(f"\n  «{s(h.left)}» [SLOT] «{s(h.right)}»   ({len(h.fillers)} filler)")
        log(f"      filler: {fillers}")
        shown += 1
        if shown >= args.top:
            break
    log(f"\n[{time.time()-t0:.0f}s] in {shown} hub. → chọn khung có filler=THỰC-THỂ (tên/danh-từ) "
        "làm bind_probe; điền thực-thể MỚI vào [SLOT]; `right` = role-backoff sẵn.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
