#!/usr/bin/env python3
"""
Probe KẾT NỐI NGANG (tác giả hỏi 2026-07-06): đếm từng lớp + power + phân tích KHOẢNG CÁCH.

Các lớp ngang trong mạng:
  1. W_seq (SEQUENTIAL)      — ngang KỀ (gap=1), weight = eta×count (power ẩn trong weight)
  2. ISG seg_h (TK_013)      — ngang trong-cha giữa slot; bigram_count = POWER tường minh
  3. FRAME_BRACKET (D3)      — ngang XA có hướng (nếu→thì); weight = PMI (KHÔNG power, KHÔNG gap!)
  4. fact_links (TK_010)     — ngang mức FACT trong t1
  5. REINFORCED (G2)         — ngang có hướng q→a trong đáp

KHOẢNG CÁCH: quét source_ids mọi cha → histogram gap opener→closer cho các cặp bracket top.
Nếu phân bố NHỌN (mode rõ) → gap là tín hiệu dùng được cho gate (R2 TK_017 §11).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.util.progress import get_run_logger

PAIRS = [("nếu", "thì"), ("tuy", "nhưng"), ("vì", "nên"), ("càng", "càng"),
         ("không", "mà"), ("khi", "thì")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4_full.pkl")
    args = ap.parse_args()
    log = get_run_logger("probe_hlinks")
    state, _ = load_checkpoint(args.checkpoint, logger=log)
    ps, v = state.parents, state.vocab

    print("\n=== 1. ĐẾM CÁC LỚP KẾT NỐI NGANG ===")
    for kind in [EdgeKind.SEQUENTIAL, EdgeKind.FRAME_BRACKET, EdgeKind.REINFORCED,
                 EdgeKind.CO_ACTIVATION]:
        print(f"  {kind.value:16s}: {len(state.adjacency.layer(kind)):>9,} cạnh")
    seg = ps.seg
    n_seg_links = sum(len(l) for l in seg._links_by_parent.values())
    print(f"  {'ISG seg_h':16s}: {n_seg_links:>9,} cạnh ngang trong-cha "
          f"({len(seg.clusters):,} cluster; bigram_count phân biệt: {len(seg.bigram_count):,})")
    n_fact = sum(len(rec.fact_links) for rec in ps.parents)
    print(f"  {'fact_links (t1)':16s}: {n_fact:>9,} cạnh ngang mức fact")

    print("\n=== 2. POWER của cạnh ngang ===")
    top_bg = sorted(seg.bigram_count.items(), key=lambda kv: -kv[1])[:8]
    print("  ISG bigram_count (POWER tường minh) top:")
    for (a, b), c in top_bg:
        print(f"    {v.token_of(a)!r}→{v.token_of(b)!r}: power={c}")
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    print(f"  FRAME_BRACKET: weight=PMI, KHÔNG lưu power/count, KHÔNG lưu gap (bị vứt lúc mine)")

    print("\n=== 3. KHOẢNG CÁCH opener→closer (quét source_ids mọi cha) ===")
    ids_of = {}
    for a, b in PAIRS:
        ia, ib = v._token_to_id.get(a), v._token_to_id.get(b)
        if ia is not None and ib is not None:
            ids_of[(a, b)] = (ia, ib)
    hist: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for rec in ps.parents:
        src = rec.source_ids
        pos: dict[int, list[int]] = defaultdict(list)
        for i, t in enumerate(src):
            pos[t].append(i)
        for (a, b), (ia, ib) in ids_of.items():
            for i in pos.get(ia, []):
                # closer GẦN NHẤT sau opener (gap≥1)
                nxt = [j for j in pos.get(ib, []) if j > i]
                if nxt:
                    hist[(a, b)][nxt[0] - i] += 1
    for pair, h in hist.items():
        total = sum(h.values())
        if total < 5:
            continue
        mode_gap, mode_n = h.most_common(1)[0]
        top5 = h.most_common(5)
        in_w = sum(n for g, n in h.items() if abs(g - mode_gap) <= 2) / total
        print(f"  {pair[0]!r}→{pair[1]!r}: n={total}  mode_gap={mode_gap} "
              f"({mode_n/total:.0%})  P(|gap-mode|≤2)={in_w:.0%}  top5={top5}")


if __name__ == "__main__":
    main()
