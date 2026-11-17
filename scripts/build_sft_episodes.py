#!/usr/bin/env python3
"""
Build t1 SFT (Trụ 7 T2 — TK_004) từ essays.jsonl (vi-alpaca) trên checkpoint có sẵn.

Mỗi bài: câu → statement parent (priority 0.5, NGOÀI miền — không lọt pool curated two-tier)
+ G1/W_seq (fluency cho văn essay); t1 = member xuất xứ theo thứ tự + member CỘNG HƯỞNG từ
tri thức cũ (cơ chế tác giả: np1/np2 "rung" thì nối); match_ids = token instruction (+input).

Mặc định build trên grade_v3_big_a2_t1.pkl (giữ episode wiki) → grade_v3_big_a2_t2.pkl.

  python scripts/build_sft_episodes.py
  python scripts/build_sft_episodes.py --max 200 --resonance-top 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.training.ingest import ingest_statements

ROOT = Path(__file__).resolve().parents[1]

# ngoài miền: dưới curated_priority (0.9) để two-tier không đổi; trên mức "nhiễu" để
# completion member vẫn dùng được khi essay là đích đúng.
ESSAY_PRIORITY = 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description="Build t1 SFT từ essays.jsonl (TK_004)")
    ap.add_argument("--checkpoint", default="checkpoints/grade_v3_big_a2_t1.pkl")
    ap.add_argument("--essays", default="data/real_vi/essays.jsonl")
    ap.add_argument("--out", default="checkpoints/grade_v3_big_a2_t2.pkl")
    ap.add_argument("--max", type=int, default=0, help="0 = tất cả")
    ap.add_argument("--resonance-top", type=int, default=8,
                    help="số member cộng hưởng tối đa nối từ tri thức CŨ (0 = tắt)")
    ap.add_argument("--resonance-min", type=float, default=0.05)
    args = ap.parse_args()

    state, cache = load_checkpoint(str(ROOT / args.checkpoint))
    ps = state.parents
    n_parents_before = len(ps)
    print(f"Checkpoint: {args.checkpoint}  parents={n_parents_before}  "
          f"episodes={len(getattr(ps, 'episodes', []))}")

    essays = [json.loads(l) for l in (ROOT / args.essays).open(encoding="utf-8")]
    if args.max:
        essays = essays[: args.max]

    made = res_total = 0
    for e in essays:
        # 1) câu essay → parent + G1/W_seq (train-time, R0 split_punct như dữ liệu HF khác)
        ingest_statements(state, e["sentences"], priority=ESSAY_PRIORITY, split_punct=True)
        member_idxs: list[int] = []
        for s in e["sentences"]:
            ids = tuple(state.vocab.encode_tokens(tokenize_punct(s)))
            pi = ps._key_to_idx.get(ids)
            if pi is not None and pi not in member_idxs:
                member_idxs.append(pi)
        if len(member_idxs) < 2:
            continue
        own = set(member_idxs)

        # 2) cộng hưởng với tri thức CŨ (trước khi bài này nạp): kích hoạt bag instruction+bài
        resonance: list[tuple[int, float]] = []
        if args.resonance_top > 0:
            bag: list[int] = []
            for s in [e["instruction"], e.get("input", "")] + e["sentences"]:
                if s:
                    bag.extend(state.vocab.encode_tokens(tokenize_punct(s)))
            ranked = ps.rank_matches(bag, state.vocab, None,
                                     top_n=args.resonance_top + len(own), fuzzy_topk=0)
            resonance = [
                (pi, float(sc)) for pi, sc in ranked
                if pi not in own and pi < n_parents_before and sc >= args.resonance_min
            ][: args.resonance_top]
            res_total += len(resonance)

        # 3) chìa khớp = instruction (+input nếu có)
        key_text = e["instruction"] + (" " + e["input"] if e.get("input") else "")
        match_ids = tuple(state.vocab.encode_tokens(tokenize_punct(key_text)))
        ps.add_sft_episode(
            key=e["doc_id"], title=e["instruction"][:80],
            match_ids=match_ids, member_idxs=member_idxs, resonance=resonance,
            cache=cache,
        )
        made += 1

    print(f"→ {made} episode SFT; parents {n_parents_before}→{len(ps)}; "
          f"member cộng hưởng: {res_total} (TB {res_total/max(1, made):.1f}/bài)")
    save_checkpoint(state, str(ROOT / args.out), cache)
    print(f"Đã lưu: {args.out} (+ content sidecar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
