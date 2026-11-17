#!/usr/bin/env python
"""PHA 1 install (§12.114/§12.115) — tái dựng E_skip vào checkpoint (re-derive, KHÔNG train).

Embedding tự nhận là "view dẫn xuất của A — tính lại offline mỗi epoch" (`core/embedding.py:133`).
Cài E_skip = tính lại closed-form từ `emit_ids` đã có; không dữ liệu mới, không backprop.

Hai chế độ:
  --neighbors "Lý,phở,Hà Nội,vua"   cổng G-emb(c): in hàng xóm E_old vs E_skip để đọc tay (v12).
  --save PATH                       cổng đã đạt ⇒ ghi checkpoint mới với state.embedding = E_skip.

KHÔNG ghi đè checkpoint gốc trừ khi --save trỏ thẳng vào nó (cố ý). Mặc định chỉ đo.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402
from nn_aivie.hca.embedding_skip import build_eskip  # noqa: E402


def _id(v, w):
    return v._token_to_id.get(w) or v._token_to_id.get("Ġ" + w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--emb-dmax", type=int, default=8)
    ap.add_argument("--emb-lam", type=float, default=0.85)
    ap.add_argument("--emb-dim", type=int, default=96)
    ap.add_argument("--neighbors", default="", help="danh sách từ, phẩy phân cách — dump G-emb(c)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--save", default="", help="đường dẫn ghi checkpoint mới (E_skip). Trống = chỉ đo.")
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    E_old = state.embedding
    print(f"[{time.time()-t0:.0f}s] loaded {len(state.parents):,} cha · "
          f"E_old {None if E_old is None else E_old.matrix.shape}", flush=True)
    E_new = build_eskip(state, dim=args.emb_dim, d_max=args.emb_dmax, lam=args.emb_lam)
    print(f"[{time.time()-t0:.0f}s] E_skip {E_new.matrix.shape}", flush=True)

    if args.neighbors:
        def nn(emb, w):
            tid = _id(v, w)
            if tid is None or emb is None:
                return f"(không id: {w})"
            return ", ".join(v.token_of(i).lstrip("Ġ")
                             for i, _ in emb.nearest(tid, topk=args.topk, min_sim=-1.0))
        print("\n=== G-emb(c) hàng xóm ngữ nghĩa (đọc tay) ===")
        for w in [x.strip() for x in args.neighbors.split(",") if x.strip()]:
            print(f"  «{w}»")
            print(f"    E_old : {nn(E_old, w)}")
            print(f"    E_skip: {nn(E_new, w)}")

    if args.save:
        state.embedding = E_new
        out = str(_ROOT / args.save) if not Path(args.save).is_absolute() else args.save
        save_checkpoint(state, out)
        print(f"\n[{time.time()-t0:.0f}s] ✓ ghi checkpoint E_skip → {out}", flush=True)
    else:
        print(f"\n[{time.time()-t0:.0f}s] (chỉ đo — không --save)", flush=True)


if __name__ == "__main__":
    main()
