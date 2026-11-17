#!/usr/bin/env python3
"""TK_042 §0.1b — Đo phân bố next-token THẬT: có gì để lấy mẫu không?

    PYTHONPATH=src python scripts/probe_branching.py checkpoints/network_v10_histwiki.pkl

⚠️ KHÔNG chạy trên `network_v10.pkl` (1.85GB đĩa → ~20GB RAM, TK_040 M0) — sẽ swap máy.

⚠️ Số này là H(next|w0) TB với mỗi w0 tính ĐỀU NHAU — KHÔNG phải cross-entropy trên văn
thật (TK_031: 4.155 bits). Hai đại lượng khác nhau, đừng trộn (TK_042 §0.2).

Câu hỏi: mạng có một PHÂN BỐ xác suất để sinh, hay một chuỗi gần-xác-định để chép?
Nếu entropy ≈ 0 → không có gì để lấy mẫu → temperature vô nghĩa.
"""
import contextlib
import io
import sys

import numpy as np

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind

ckpt = sys.argv[1]
print(f"load {ckpt} …", flush=True)
with contextlib.redirect_stdout(io.StringIO()):
    state, _ = load_checkpoint(ckpt)
v, ps = state.vocab, state.parents
print(f"vocab={len(v):,} cha={len(ps.parents):,}", flush=True)

W = state.adjacency.layer(EdgeKind.SEQUENTIAL)
top = [t for t, _ in sorted(ps._gdf.items(), key=lambda kv: -kv[1])[:400]]
brs, ents, p1s = [], [], []
for t in top:
    try:
        row = W.row(t)
    except Exception:
        continue
    if not row or len(row) < 1:
        continue
    w = np.array(list(row.values()), dtype=float)
    if w.sum() <= 0:
        continue
    p = w / w.sum()
    brs.append(len(row))
    ents.append(float(-(p * np.log2(np.maximum(p, 1e-12))).sum()))
    p1s.append(float(p.max()))

brs, ents, p1s = np.array(brs), np.array(ents), np.array(p1s)
print(f"\n=== P(next|w0), {len(brs)} token tần cao nhất ===")
print(f"  phân nhánh : TB {brs.mean():8.1f}  median {np.median(brs):6.0f}  successor/token")
print(f"  entropy    : TB {ents.mean():8.2f}  median {np.median(ents):6.2f}  bits")
print(f"  P(top-1)   : TB {p1s.mean():8.3f}  median {np.median(p1s):6.3f}")
print(f"  perplexity hữu hiệu (2^H): TB {2**ents.mean():.1f} lựa chọn thật sự")
print(f"\n  % token entropy < 1 bit (gần XÁC ĐỊNH): {100*(ents < 1).mean():.1f}%")
print(f"  % token P(top-1) > 0.5 (một đường áp đảo): {100*(p1s > 0.5).mean():.1f}%")
print(f"  % token P(top-1) > 0.9 : {100*(p1s > 0.9).mean():.1f}%")
