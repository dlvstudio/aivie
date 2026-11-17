#!/usr/bin/env python3
"""
C0 probe (TK_015, docs/study/08) — XÁC NHẬN 2 bức tường mà từ điển (D1) + khung xa (D3) nhắm,
CÓ THẬT và thấy được trên checkpoint hiện tại. Read-only, không rebuild (rẻ).

Tường 1 — ĐA NGHĨA BLUR (D1 sense-neuron nhắm): token đa nghĩa = 1 vector → hàng xóm embedding
  TRỘN nghĩa (hoặc chỉ 1 nghĩa trội nuốt phần còn lại). Nếu tách sense-neuron thì mỗi nghĩa có
  hồ sơ hàng xóm riêng.
Tường 2 — THIẾU CẠNH XA (D3 FRAME_BRACKET nhắm): cặp khung `nếu…thì`, `Hỏi:…Đáp:` KHÔNG kề
  nhau → cạnh SEQUENTIAL (bigram) gần 0 dù chúng thuộc về nhau; substrate không có link biểu
  diễn ràng buộc này.
"""

from __future__ import annotations

import argparse

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind

POLYSEMOUS = ["đông", "chân", "đá", "là", "chạy", "bàn", "cân", "cổ"]
FRAME_PAIRS = [("nếu", "thì"), ("tuy", "nhưng"), ("vì", "nên"),
               ("bởi", "nên"), ("càng", "càng"), ("Hỏi:", "Đáp:")]


def _id(vocab, tok: str):
    return vocab._token_to_id.get(tok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/grade_v3_big_a2_t3_isg.pkl")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    print(f"load {args.checkpoint} …")
    state, _ = load_checkpoint(args.checkpoint)
    vocab, emb, adj = state.vocab, state.embedding, state.adjacency
    seq = adj.layer(EdgeKind.SEQUENTIAL)
    co = adj.layer(EdgeKind.CO_ACTIVATION)
    print(f"vocab={len(vocab)}\n")

    print("=" * 70)
    print("TƯỜNG 1 — ĐA NGHĨA BLUR (hàng xóm embedding của token đa nghĩa)")
    print("=" * 70)
    for w in POLYSEMOUS:
        wid = _id(vocab, w)
        if wid is None:
            print(f"  '{w}': OOV (không có trong vocab)")
            continue
        nb = emb.nearest(wid, topk=args.topk)
        toks = [f"{vocab.token_of(i)}·{s:.2f}" for i, s in nb]
        print(f"  '{w}'  →  {', '.join(toks) if toks else '(rỗng)'}")

    print("\n" + "=" * 70)
    print("TƯỜNG 2 — THIẾU CẠNH XA (cặp khung không-kề)")
    print("  seq(a→b) = trọng số bigram LIỀN KỀ ;  co(a,b) = đồng-hiện cửa sổ")
    print("=" * 70)
    for a, b in FRAME_PAIRS:
        ai, bi = _id(vocab, a), _id(vocab, b)
        if ai is None or bi is None:
            print(f"  '{a}'→'{b}': OOV ({a if ai is None else b})")
            continue
        s = seq.row(ai).get(bi, 0.0)
        c = co.row(ai).get(bi, 0.0)
        sim = emb.similarity(ai, bi)
        flag = "  ← cạnh kề ~0 dù thuộc về nhau" if s < 1.0 else ""
        print(f"  '{a}'→'{b}':  seq={s:.2f}  co={c:.2f}  emb_sim={sim:.2f}{flag}")


if __name__ == "__main__":
    main()
