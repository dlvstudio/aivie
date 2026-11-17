#!/usr/bin/env python3
"""
TK_010 — build t1 THỐNG NHẤT: mỗi bài vi-alpaca = MỘT nơ-ron cha QA dài (không chia câu).

Quy trình đúng một kiểu train top-down (chỉ đạo tác giả 2026-07-02):
  1. Cả bài (Hỏi=instruction, Đáp=toàn văn) → MỘT ParentRecord qua ingest_qa (G1+W_seq trên
     chuỗi đầy đủ → token nối ngang theo index).
  2. decompose_repair chạy CHUNG → trung gian của bài tự nổi (token↔trung gian).
  3. bind_fact_links: từ list token tìm fact ĐÃ CÓ (cộng hưởng) → sắp theo vị trí xuất hiện
     → tầng link phụ t1→fact (fact↔fact ngang suy từ thứ tự).

Build từ `_t1.pkl` (TRƯỚC khi có 3 680 cha-câu TK_004 — tự loại đường cũ; wiki episode giữ
làm nguồn phrase_assoc) → `grade_v3_big_a2_t3.pkl`.

  python scripts/build_t1_unified.py [--max 500]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.training.ingest import ingest_qa

ROOT = Path(__file__).resolve().parents[1]
ESSAY_PRIORITY = 0.5  # ngoài miền — dưới curated_priority (0.9), two-tier không đổi


def main() -> int:
    ap = argparse.ArgumentParser(description="TK_010: build t1 unified → _t3.pkl")
    ap.add_argument("--checkpoint", default="checkpoints/grade_v3_big_a2_t1.pkl")
    ap.add_argument("--essays", default="data/real_vi/essays.jsonl")
    ap.add_argument("--out", default="checkpoints/grade_v3_big_a2_t3.pkl")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument(
        "--legacy-repair",
        action="store_true",
        help="Chạy decompose_repair batch thay vì ISG (TK_013)",
    )
    args = ap.parse_args()

    from nn_aivie.util.progress import PerfSummary, get_run_logger

    log = get_run_logger("build_t1_unified")
    perf = PerfSummary()

    state, cache = load_checkpoint(str(ROOT / args.checkpoint))
    ps = state.parents
    n_before = len(ps)
    log.log(f"Base: {args.checkpoint}  parents={n_before}  episodes(wiki)={len(ps.episodes)}")

    essays = [json.loads(l) for l in (ROOT / args.essays).open(encoding="utf-8")]
    if args.max:
        essays = essays[: args.max]
    log.log(f"essays={len(essays)} (max={args.max or 'all'})")

    # (1) mỗi bài = MỘT cha QA dài — KHÔNG chia câu
    with log.phase("ingest_essays", summary=perf):
        pairs = [(e["instruction"], " ".join(e["sentences"])) for e in essays]
        ingest_qa(state, pairs, priority=ESSAY_PRIORITY, split_punct=True, logger=log, log_every=100)
        new_idxs = list(range(n_before, len(ps)))
        # TK_012: tag nguồn = nhãn DEBUG → content-cache (KHÔNG vào nơ-ron)
        for e, pi in zip(essays, new_idxs):
            cache.set_parent_tag(pi, e["doc_id"])
        log.log(f"  nạp {len(new_idxs)} cha t1 nguyên vẹn")

    # (2) phân rã — ISG online (TK_013) hoặc Re-Pair batch (--legacy-repair)
    with log.phase("decompose", summary=perf):
        if args.legacy_repair:
            res = ps.decompose_repair(state.vocab, min_frequency=2, max_merges=40000, max_span=6)
            log.log(f"  decompose_repair: {res}")
        else:
            ps.seg_min_frequency = 2
            res = ps.rebuild_seg_graph(state.vocab, dap_id=state.vocab.id_of("Đáp:"))
            log.log(f"  ISG rebuild: {res}")

    # (3) tầng link phụ: fact_links theo vị trí xuất hiện (chỉ nối tới cha CÓ TRƯỚC)
    with log.phase("fact_links", summary=perf):
        old_only = frozenset(new_idxs)
        total = 0
        n = len(new_idxs)
        for k, pi in enumerate(new_idxs):
            total += ps.bind_fact_links(pi, state.vocab, exclude=old_only)
            if (k + 1) % 100 == 0:
                log.progress("fact_links", k + 1, n, extra=f"links={total}")
        log.log(f"  fact_links: {total} link (TB {total/max(1,len(new_idxs)):.1f}/bài)")

    with log.phase("save", summary=perf):
        save_checkpoint(state, str(ROOT / args.out), cache)
    log.log(f"Saved: {args.out} (+ content sidecar)  parents={len(ps)}")
    perf.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
