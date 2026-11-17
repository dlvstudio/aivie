#!/usr/bin/env python3
"""
PROBE TẦNG KHUÔN (sequence-level) — chuỗi HẠNG-MỤC có cấu trúc + khái quát không? (TK_042).

    PYTHONPATH=src python -u scripts/probe_template_layer.py

## Vì sao cần probe NÀY (sau khi bigram ra ~trung tính)

`probe_category_layer` đo BIGRAM (1 bước): đủ data → category ≈ surface (surface phủ hết).
NHƯNG khuôn ngữ pháp là về CHUỖI DÀI ([DET][NOUN][VERB]…), thứ bigram KHÔNG đo. Điểm mấu chốt:
khi context DÀI ra (n=1→3), **surface n-gram SỤP phủ** (4-gram bề mặt hầu hết chưa-thấy trên
held-out) trong khi **category n-gram GIỮ phủ** (ít lớp → chuỗi hạng-mục tái xuất). Nếu category
vẫn dự đoán tốt ở n cao mà surface đã sụp ⇒ KHUÔN mang cấu trúc dài khái quát ⇒ Tier build đáng.

## Ba đo (chỉ cần E_v11.npy + corpus — KHÔNG cần graph 2GB)

1. **Đọc tay:** khuôn = cat-trigram tần suất cao + 1 hiện thực bề mặt — trông như frame ngữ pháp?
2. **Cấu trúc khuôn:** bits dự đoán HẠNG-MỤC kế theo context k=0..3. Tụt mạnh ⇒ chuỗi hạng mục
   CÓ cấu trúc (không ngẫu nhiên) = có khuôn.
3. **Khái quát (chốt):** mỗi order k=1..3, surface vs category → **coverage%** (context đã-thấy) +
   **bits** (add-k) trên held-out. Surface phủ sụp / category giữ + bits category < surface ở n cao
   ⇒ khuôn khái quát vượt chép.

## Điều kiện phân định (GHI TRƯỚC)

* category coverage GIỮ cao + bits TỤT theo n, surface coverage SỤP ⇒ khuôn có juice ⇒ build Tier 3.
* category coverage/bits cũng SỤP hoặc không hơn surface ⇒ khuôn ≈ vô ích ⇒ chốt "LM lo ngữ
  pháp + graph lo nhớ" (đường b), đóng nhánh dứt khoát (KHÔNG đắp thêm núm).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from eval_bits import load_heldout  # noqa: E402
from train_neural_lm import load_corpus  # noqa: E402
from probe_category_layer import kmeans_cosine  # noqa: E402 — CÙNG clustering (deterministic)

from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--e-path", default="out/E_v11.npy")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--manifest", default="config/train_v10.yaml")
    ap.add_argument("--holdout", default="data/sft/qa_heldout.jsonl")
    ap.add_argument("--n-holdout", type=int, default=300)
    ap.add_argument("--k", type=int, default=200, help="số hạng mục (khớp probe_category_layer)")
    ap.add_argument("--kmeans-iters", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=10_000_000, help="cắt corpus (surface n-gram tốn RAM)")
    ap.add_argument("--max-order", type=int, default=3, help="context tối đa (n-gram = order+1)")
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok.tokens)
    E = np.load(_ROOT / args.e_path).astype(np.float32)
    if E.shape[0] != V:
        raise SystemExit(f"✗ E rows {E.shape[0]} ≠ vocab {V} (TK_041 §5.2)")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    log(f"E {E.shape} · vocab {V:,}")

    ids = load_corpus(_ROOT / args.manifest, tok, None, log)[: args.max_tokens]
    log(f"k-means K={args.k} …")
    lab = kmeans_cosine(En, args.k, args.kmeans_iters, args.seed, log)
    catv = lab[ids]  # nhãn hạng mục theo dòng token

    K, mo = args.k, args.max_order
    # đếm order 1..mo: surface ctx→token, category ctx→token, category ctx→category-kế
    s_ctx = [defaultdict(lambda: defaultdict(int)) for _ in range(mo + 1)]
    c_ctx = [defaultdict(lambda: defaultdict(int)) for _ in range(mo + 1)]
    cc_ctx = [defaultdict(lambda: defaultdict(int)) for _ in range(mo + 1)]
    log(f"đếm n-gram (order 1..{mo}, corpus {len(ids)/1e6:.0f}M) …")
    idl = ids.tolist()
    cal = catv.tolist()
    for i in range(mo, len(idl)):
        w, cw = idl[i], cal[i]
        for k in range(1, mo + 1):
            s_ctx[k][tuple(idl[i - k:i])][w] += 1
            cc = tuple(cal[i - k:i])
            c_ctx[k][cc][w] += 1
            cc_ctx[k][cc][cw] += 1

    def bits(ctx_tab, ctx_key, nxt, support):
        d = ctx_tab.get(ctx_key)
        tot = sum(d.values()) if d else 0
        p = ((d.get(nxt, 0) if d else 0) + args.smooth) / (tot + args.smooth * support)
        return -math.log2(max(1e-12, p)), (d is not None and nxt in d)

    # held-out streams
    texts = load_heldout(_ROOT / args.holdout, args.n_holdout, args.seed)
    streams = [tok.encode(t) for t in texts if len(tok.encode(t)) > mo]
    log(f"held-out {Path(args.holdout).name}: {len(streams)} chuỗi")

    # ── PART 1: đọc tay khuôn (cat-trigram tần cao) ──
    log("═══ PART 1: KHUÔN cat-trigram tần cao + 1 hiện thực bề mặt ═══")
    tri = cc_ctx[min(3, mo)] if mo >= 3 else cc_ctx[mo]
    top_ctx = sorted(tri, key=lambda c: -sum(tri[c].values()))[:10]
    for cc in top_ctx:
        # tìm 1 chuỗi surface thật khớp khuôn này trong corpus
        realiz = ""
        for i in range(mo, min(len(idl), 2_000_000)):
            if tuple(cal[i - len(cc):i]) == cc:
                realiz = tok.decode(idl[i - len(cc):i + 1])
                break
        log(f"  {cc} (n={sum(tri[cc].values())}): {realiz!r}")

    # ── PART 2: cấu trúc khuôn — bits dự đoán HẠNG-MỤC kế theo context ──
    log("═══ PART 2: bits dự đoán HẠNG-MỤC kế (tụt theo k ⇒ khuôn có cấu trúc) ═══")
    for k in range(1, mo + 1):
        tot = 0.0
        n = 0
        for e in streams:
            ce = [int(lab[t]) for t in e]
            for i in range(k, len(ce)):
                b, _ = bits(cc_ctx[k], tuple(ce[i - k:i]), ce[i], K)
                tot += b
                n += 1
        log(f"  order {k} (context {k} hạng-mục): {tot/max(1,n):.3f} bits/hạng-mục")

    # ── PART 3 (CHỐT): khái quát — coverage + bits token-kế, surface vs category ──
    log("═══ PART 3: dự đoán TOKEN kế — coverage% + bits, surface vs category theo order ═══")
    for k in range(1, mo + 1):
        sb = cb = 0.0
        scov = ccov = 0
        n = 0
        for e in streams:
            ce = [int(lab[t]) for t in e]
            for i in range(k, len(e)):
                b1, h1 = bits(s_ctx[k], tuple(e[i - k:i]), e[i], V)
                b2, h2 = bits(c_ctx[k], tuple(ce[i - k:i]), e[i], V)
                sb += b1
                cb += b2
                scov += h1
                ccov += h2
                n += 1
        n = max(1, n)
        log(f"  order {k}: surface cov {scov/n*100:4.1f}% bits {sb/n:6.3f}  ·  "
            f"category cov {ccov/n*100:4.1f}% bits {cb/n:6.3f}")

    log("═══ ĐỌC: PART3 — surface cov SỤP theo order? category cov GIỮ + bits < surface ở order cao "
        "⇒ KHUÔN khái quát ⇒ build Tier 3. Cả hai sụp/không hơn ⇒ chốt LM-lo-ngữ-pháp (đóng nhánh) ═══")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
