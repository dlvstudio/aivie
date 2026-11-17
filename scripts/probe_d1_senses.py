#!/usr/bin/env python3
"""
TK_015 D1 — đo TÁCH NGHĨA: sense-neuron mỗi nghĩa có hồ sơ hàng xóm RIÊNG không (vs vector
token đơn bị trộn ở C0)? So token blur ↔ các sense-neuron; Jaccard neighbor-set giữa sense
(THẤP = tách rõ). Read-only trên checkpoint đã có substrate từ điển.
"""

from __future__ import annotations

import argparse

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.vocab import is_surface_token

WORDS = ["đá", "đông", "chân", "chạy", "bàn", "cân", "cổ"]


def _surface_neighbors(state, nid, topk):
    out = []
    for i, s in state.embedding.nearest(nid, topk=topk * 3):
        tok = state.vocab.token_of(i)
        if is_surface_token(tok):
            out.append((tok, s))
        if len(out) >= topk:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4.pkl")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()
    print(f"load {args.checkpoint} …")
    state, _ = load_checkpoint(args.checkpoint)
    v = state.vocab
    print(f"vocab={len(v)}\n")

    for w in WORDS:
        tid = v._token_to_id.get(w)
        print("=" * 66)
        if tid is None:
            print(f"'{w}': token OOV"); continue
        blur = _surface_neighbors(state, tid, args.topk)
        print(f"'{w}' (token đơn — C0 blur): {', '.join(f'{t}·{s:.2f}' for t,s in blur)}")
        # thu các sense-neuron
        senses = []
        k = 1
        while (sid := v._token_to_id.get(f"Sense:{w}#{k}")) is not None:
            senses.append((k, sid))
            k += 1
        if not senses:
            print("  (chưa có sense-neuron — từ không trong từ điển đã nạp)"); continue
        nbsets = []
        for k, sid in senses:
            nb = _surface_neighbors(state, sid, args.topk)
            nbsets.append({t for t, _ in nb})
            print(f"  Sense:{w}#{k}: {', '.join(f'{t}·{s:.2f}' for t,s in nb)}")
        # Jaccard trung bình giữa các cặp sense (thấp = tách)
        if len(nbsets) >= 2:
            js = []
            for a in range(len(nbsets)):
                for b in range(a + 1, len(nbsets)):
                    u = nbsets[a] | nbsets[b]
                    js.append(len(nbsets[a] & nbsets[b]) / len(u) if u else 0.0)
            print(f"  → Jaccard TB giữa nghĩa = {sum(js)/len(js):.2f} (THẤP = tách nghĩa rõ)")


if __name__ == "__main__":
    main()
