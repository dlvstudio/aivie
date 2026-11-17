#!/usr/bin/env python3
"""
TK_042 G9 — BIỂU DIỄN ngữ cảnh: cái túi decay=0.7 tự bóp chết token cuối. Sửa được không?

    PYTHONPATH=src python -u scripts/probe_ctx_repr.py --checkpoint checkpoints/network_v10_histwiki.pkl

## Vì sao (đọc tay G8 §7.8h — 4 ca thật)

```
ngữ cảnh: "…cẩn thận quan sát để tìm ra tất"   GOLD: 'cả'
cosine chọn: …"ra cho"→'nhà'  …"tạo ra cho"→'nhân'  …"khám phá ra"→'dòng'
hàng xóm ĐÚNG: HẠNG 255/256
```

`"tất cả"` là từ ghép CỐ ĐỊNH — token cuối quyết định gần như tuyệt đối. Mà `decay=0.7` trên
L=8 cho trọng số `1, 0.7, 0.49, 0.34…` **tổng ≈ 3.3** ⇒ `E[tất]` chỉ chiếm **~30%** vector ⇒
**7 token còn lại DÌM CHẾT tín hiệu mạnh nhất**. Cosine đi tìm hàng xóm kết thúc bằng `cho`/`ra`.

⇒ Đó KHÔNG phải lỗi hàm chấm. Biểu diễn **tự bóp chết** thứ nó cầm trong tay. Và `decay=0.7`
là con số tác giả probe **chọn tuỳ tiện, không đo**.

## Thước: LIFT@1, không phải recall@1

G8 §7.8i đã học: **recall một mình VÔ NGHĨA** — với k lớn + Zipf, gold lọt vào do MAY
(recall@256=57.3% mà ngẫu nhiên đã 26.5%). Phải so với ĐỐI CHỨNG NGẪU NHIÊN.

`lift@1 = recall@1 / recall@1(ngẫu nhiên)` = **hàng xóm GẦN NHẤT đúng token kế, hơn may bao
nhiêu lần**. Đo trên `decay=0.7, L=8`: **98×** (15.0% vs 0.2%) — tín hiệu THẬT và MẠNH, kể cả
với cái túi thô. Câu hỏi: sửa biểu diễn thì lên bao nhiêu?

`decay=0.0` ⇒ **CHỈ token cuối** ⇒ bigram khớp-mờ. Nếu nó thắng `decay=0.7` thì cái túi 8 token
đang **HẠI** chứ không giúp — và đó là kết luận đắt giá, vì nó nói ngữ cảnh dài **không lấy được
bằng cách cộng vector**.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

from nn_aivie.core.checkpoint_io import load_checkpoint

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from eval_bits import build_counts, load_heldout  # noqa: E402
from probe_knn_lm import ctx_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--holdout", default="data/real_vi/essays.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datastore", type=int, default=400_000)
    ap.add_argument("--configs", default="0.0:1,0.0:8,0.3:8,0.5:8,0.7:8,0.9:8,0.5:4,0.5:2")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"load {args.checkpoint} …", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        state, _ = load_checkpoint(args.checkpoint)
    E = state.embedding.matrix.astype(np.float32)
    V = len(state.vocab)
    seqs = [list(p.source_ids) for p in state.parents.parents if len(p.source_ids) > 2]
    rng.shuffle(seqs)
    texts = load_heldout(_ROOT / args.holdout, args.n, args.seed)
    enc = [state.vocab.encode_prompt(t) for t in texts]
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    _bi, uni, total = build_counts(state)
    p_uni = (uni + 1.0) / (total + V)
    print(f"vocab={V:,} · cha={len(seqs):,} · E{E.shape}", flush=True)

    print(
        f"\n{'decay':>6} {'L':>3} {'recall@1':>9} {'NGẪU':>7} {'LIFT@1':>8} "
        f"{'recall@k':>9} {'bits/BYTE':>10}"
    )
    print(f"{'':>6} {'':>3} {'':>9} {'':>7} {'':>8} {'':>9} {'(k=%d)' % args.k:>10}")
    best = None
    for cfg in args.configs.split(","):
        d_s, l_s = cfg.split(":")
        decay, L = float(d_s), int(l_s)
        X, Y, _ = ctx_matrix(seqs, E, L, decay, args.datastore, rng, keep_ctx=True)
        Q, G, _ = ctx_matrix(enc, E, L, decay, 10**9, rng, keep_ctx=True)

        CH = 256
        top = np.zeros((len(Q), args.k), dtype=np.int32)
        topsim = np.zeros((len(Q), args.k), dtype=np.float32)
        for a in range(0, len(Q), CH):
            b = min(a + CH, len(Q))
            s_blk = Q[a:b] @ X.T
            idx = np.argpartition(-s_blk, args.k, axis=1)[:, : args.k]
            sm = np.take_along_axis(s_blk, idx, 1)
            order = np.argsort(-sm, axis=1)
            top[a:b] = np.take_along_axis(idx, order, 1)
            topsim[a:b] = np.take_along_axis(sm, order, 1)
            del s_blk

        rnd = np.random.default_rng(0).integers(0, len(X), size=top.shape)
        r1 = float((Y[top[:, 0]] == G).mean())
        rr1 = float((Y[rnd[:, 0]] == G).mean())
        rk = float((Y[top] == G[:, None]).any(axis=1).mean())

        sb = 0.0
        for lam in (0.5, 0.7):
            t = 0.0
            for r in range(len(Q)):
                w = np.exp((topsim[r] - topsim[r].max()) / args.tau)
                w /= w.sum()
                hit = Y[top[r]] == G[r]
                p_knn = float(w[hit].sum()) if hit.any() else 0.0
                p = lam * p_knn + (1 - lam) * float(p_uni[G[r]])
                t += -math.log2(max(p, 1e-12))
            sb = min(sb, t / nbytes) if sb else t / nbytes
        lift = r1 / max(rr1, 1e-9)
        print(
            f"{decay:>6.1f} {L:>3} {100*r1:>8.1f}% {100*rr1:>6.2f}% {lift:>7.1f}× "
            f"{100*rk:>8.1f}% {sb:>10.3f}"
        )
        if best is None or lift > best[0]:
            best = (lift, decay, L, sb)

    print(f"\n  ⇒ LIFT@1 tốt nhất: {best[0]:.1f}× tại decay={best[1]}, L={best[2]} "
          f"({best[3]:.3f} bits/byte)")
    print("  Đối chiếu: decay=0.7,L=8 (probe G8 dùng) = 98× · ĐẾM baseline = 1.288 bits/byte")
    print("\n  decay=0.0,L=1 = CHỈ token cuối = bigram khớp-mờ. Nếu nó THẮNG túi 8 token")
    print("  ⇒ cộng vector KHÔNG lấy được ngữ cảnh dài (kết luận đắt: về G7).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
