#!/usr/bin/env python3
"""
Thước LM không bão hoà (Trụ 1 / bước 0 — docs/study/08, TK_015).

Đo khả năng dự đoán next-token của SUBSTRATE (base-LM: W_seq ⊙ readout, KHÔNG agenda) trên
corpus held-out THẬT → log2(rank) + MRR + top-k. Không bão hoà (rank trải 1..vocab), nhạy với
cải tiến substrate (D1 từ điển, D2 q·k, D3 khung xa). THẤP bits = tốt.

Held-out mặc định: câu trong `data/real_vi/essays.jsonl` (vi-alpaca) — KHÔNG nằm trong
`grade_v3_big_a2.pkl` (build TRƯỚC essay _t2) → held-out sạch tương đối. Sổ nợ leakage: để có
số TUYỆT ĐỐI sạch cần reserved-shard lúc train (TK_011 perf) — số hiện tại dùng để theo DELTA
qua D1–D4 trên corpus cố định (đủ cho ruler tiến hoá).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.eval.metrics import next_token_lm

_ROOT = Path(__file__).resolve().parents[1]
_HELDOUT = _ROOT / "data/eval_v4/heldout_lm.jsonl"


def _build_heldout(n: int, seed: int) -> list[str]:
    """Sample cố định (seed) các câu essay → persist để ruler ổn định qua các lần chạy."""
    if _HELDOUT.exists():
        return [json.loads(l)["text"] for l in _HELDOUT.read_text("utf-8").splitlines() if l.strip()]
    src = _ROOT / "data/real_vi/essays.jsonl"
    sents: list[str] = []
    for line in src.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        for s in json.loads(line).get("sentences", []):
            s = s.strip()
            if 5 <= len(s.split()) <= 40:  # câu vừa phải, đủ vị trí chấm
                sents.append(s)
    random.Random(seed).shuffle(sents)
    chosen = sents[:n]
    _HELDOUT.parent.mkdir(parents=True, exist_ok=True)
    _HELDOUT.write_text("\n".join(json.dumps({"text": s}, ensure_ascii=False) for s in chosen), "utf-8")
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4_convpool.pkl")
    ap.add_argument("--n", type=int, default=200, help="số câu held-out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    texts = _build_heldout(args.n, args.seed)
    print(f"Held-out: {len(texts)} câu ({_HELDOUT.relative_to(_ROOT)})")
    print(f"Checkpoint: {args.checkpoint}  (đang load…)")
    state, _cache = load_checkpoint(args.checkpoint)
    print(f"vocab={len(state.vocab)}  parents={len(state.parents.parents) if state.parents else 0}")

    r = next_token_lm(state, texts)
    print("\n=== LM ruler (base-LM, không agenda) ===")
    print(f"  bits (mean log2 rank) : {r.bits:.3f}   ← THẤP = tốt, KHÔNG bão hoà")
    print(f"  MRR                   : {r.mrr:.3f}")
    print(f"  top-1 / top-5 / top-10: {r.top1:.1%} / {r.top5:.1%} / {r.top10:.1%}")
    print(f"  vị trí chấm           : {r.n_pos}  (bỏ qua OOV/non-surface: {r.n_oov})")


if __name__ == "__main__":
    main()
