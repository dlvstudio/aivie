#!/usr/bin/env python3
"""
Build tầng nơ-ron trừu tượng t1 (episode — v4 Trụ 7) cho checkpoint ĐÃ CÓ, không train lại.

Đọc data/real_vi/statements_doc.jsonl (câu wiki kèm doc_id + para + title), nhóm câu theo
đoạn văn, tra nơ-ron cha qua _key_to_idx (đúng chuỗi token đã ingest) → nối member theo
XUẤT XỨ, giữ thứ tự. Lưu checkpoint mới (mặc định: thêm hậu tố _t1 — giữ baseline nguyên vẹn).

Ví dụ:
  python scripts/build_episodes.py --checkpoint checkpoints/grade_v3_big_a2.pkl
  python scripts/build_episodes.py --checkpoint ... --out checkpoints/grade_v3_big_a2.pkl  # ghi đè
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.teaching.align import tokenize_punct

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build t1 episode layer từ statements_doc.jsonl")
    ap.add_argument("--checkpoint", default="checkpoints/grade_v3_big_a2.pkl")
    ap.add_argument("--docs", default="data/real_vi/statements_doc.jsonl")
    ap.add_argument("--out", default="", help="đường dẫn lưu (mặc định: <checkpoint>_t1.pkl)")
    ap.add_argument("--min-members", type=int, default=2)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else ckpt.with_name(ckpt.stem + "_t1.pkl")
    state, cache = load_checkpoint(str(ckpt))
    ps = state.parents
    if ps is None:
        raise SystemExit("checkpoint không có parent store")
    print(f"Checkpoint: {ckpt}  parents={len(ps)}  episodes hiện có={len(getattr(ps, 'episodes', []))}")

    # nhóm câu theo (doc_id, para) — GIỮ THỨ TỰ xuất hiện
    groups: dict[str, list[str]] = defaultdict(list)
    titles: dict[str, str] = {}
    with (ROOT / args.docs).open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            did = d.get("doc_id")
            if not did:
                continue
            key = f"wiki:{did}#{d.get('para', 0)}"
            groups[key].append(d["text"])
            titles.setdefault(key, d.get("title", ""))

    print(f"Đoạn văn (episode ứng viên): {len(groups)}")
    payload = [
        (key, titles.get(key, ""),
         [tuple(state.vocab.encode_tokens(tokenize_punct(s))) for s in sents])
        for key, sents in groups.items()
    ]
    made, matched = ps.build_episodes_from_provenance(
        payload, min_members=args.min_members, cache=cache)
    total_sents = sum(len(s) for s in groups.values())
    print(f"→ tạo {made} episode t1 (≥{args.min_members} member); "
          f"câu match cha: {matched}/{total_sents} ({matched/total_sents:.0%})")
    in_ep = len({pi for e in ps.episodes for pi, _ in e.members})
    print(f"→ cha thuộc ≥1 episode: {in_ep}/{len(ps)}")

    save_checkpoint(state, str(out), cache)
    print(f"Đã lưu: {out} (+ content sidecar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
