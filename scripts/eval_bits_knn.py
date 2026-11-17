#!/usr/bin/env python3
"""
TK_042 G16 — bits/byte HELD-OUT: kNN có cải thiện PERPLEXITY thật không?

    PYTHONPATH=src python -u scripts/eval_bits_knn.py

## Vì sao đây là thước CHỐT (§7.17c/d)

G13→G15 đo bằng đọc tay + novelty + rep4 — mọi thước đó hoặc mù ("bịa mạch lạc") hoặc chủ quan.
bits/byte = cross-entropy thật: **-log2 P(token đúng)** cộng dồn, chuẩn hoá theo byte. Nó phạt
CẢ hai lỗi cùng lúc:
* LẶP  → gán xác suất cao cho token lặp, thấp cho token đúng ở chỗ khác ⇒ bits cao
* BỊA  → gán xác suất cao cho token sai ⇒ bits cao ở token đúng

Không đọc tay, không novelty. Con số thấp hơn = mô hình tốt hơn, hết cãi.

## Câu hỏi DUY NHẤT script trả lời

    bits/byte(LM+kNN) < bits/byte(LM thuần) ?

* CÓ  ⇒ kNN cải thiện perplexity THẬT — đáng giữ dù sinh chưa hoàn hảo (§7.17d còn cần LM lớn
        hơn để SINH tốt, nhưng kNN đã chứng minh giá trị đo được).
* KHÔNG ⇒ kNN không giúp density — bỏ, và xác nhận LM 11.4M là trần bằng CON SỐ, không phải
        đọc tay.

## Vì sao phải TÁCH held-out khỏi datastore (leakage = kết quả giả)

Nếu đánh giá trên chính bài đã nạp vào datastore, kNN tìm thấy CHÍNH token đang đo (cos≈1.0)
⇒ bits≈0 giả tạo. Chia theo bài: index % `--holdout-every` == 0 → held-out, còn lại → datastore.
Không bài nào vừa ở hai phía.

## Vì sao λ phải > 0

P_kNN(token đúng) có thể = 0 (token đúng không nằm trong k hàng xóm) ⇒ -log2(0) = ∞. kNN-LM
LUÔN nội suy với LM (λ>0) để P_LM>0 mọi token làm sàn. Vì vậy KHÔNG đo λ=0 (kNN thuần) —
không phải chọn tuỳ tiện, là ràng buộc toán học.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from nn_aivie.core.neural_lm import NeuralLM  # noqa: E402
from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402
from probe_knn_scale import KnnStoreScale  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="out/g7_best_model.pt")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--corpus", default="data/text_corpus/history_articles.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--knn-temp", type=float, default=0.05)
    ap.add_argument("--holdout-every", type=int, default=10, help="bài i%%N==0 → held-out")
    ap.add_argument("--max-tokens", type=int, default=2500, help="giới hạn token đánh giá (CPU)")
    ap.add_argument("--lams", default="1.0,0.7,0.5,0.4,0.25")
    args = ap.parse_args()

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok)
    lm = NeuralLM.load(_ROOT / args.lm, V, device="cpu")
    lams = [float(x) for x in args.lams.split(",")]
    print(f"LM {sum(p.numel() for p in lm.model.parameters())/1e6:.1f}M · vocab {V:,} · ctx={lm.ctx}")

    # chia bài: datastore vs held-out — KHÔNG bài nào ở cả hai phía
    ds_docs, ho_docs = [], []
    for i, line in enumerate(open(_ROOT / args.corpus, encoding="utf-8")):
        (ho_docs if i % args.holdout_every == 0 else ds_docs).append(json.loads(line))
    print(f"datastore {len(ds_docs)} bài · held-out {len(ho_docs)} bài (tách, không leakage)")

    store = KnnStoreScale(lm, tok)
    for text in ds_docs:
        store.add_document(text)
    store.finalize()
    print(f"✓ datastore {len(store.Y):,} cặp (h,next)\n")

    bits = {lam: 0.0 for lam in lams}
    n_tok = n_byte = 0
    ctx = lm.ctx
    for text in ho_docs:
        ids = tok.encode(text)
        for i in range(1, len(ids)):
            if n_tok >= args.max_tokens:
                break
            prefix = ids[:i]
            target = ids[i]
            with torch.no_grad():
                lg = lm.model(torch.tensor(prefix[-ctx:])[None])[0, -1]
                p_lm = torch.softmax(lg, -1).numpy().astype(np.float64)
            p_kn, _ = store.probs(prefix, V, args.k, args.knn_temp)
            for lam in lams:
                p_mix = lam * p_lm[target] + (1 - lam) * p_kn[target]
                bits[lam] += -np.log2(max(p_mix, 1e-12))
            n_tok += 1
            n_byte += len(tok.decode([target]).encode("utf-8"))
        if n_tok >= args.max_tokens:
            break

    print(f"đánh giá {n_tok:,} token · {n_byte:,} byte held-out\n")
    print(f"  {'λ':>5} {'bits/token':>11} {'bits/byte':>10}   ghi chú")
    base_bt = bits[1.0] / max(n_tok, 1)
    best_lam, best_bb = None, float("inf")
    for lam in lams:
        bpt = bits[lam] / max(n_tok, 1)
        bpb = bits[lam] / max(n_byte, 1)
        note = "← LM THUẦN (baseline)" if lam == 1.0 else (
            f"cải thiện {100*(base_bt-bpt)/base_bt:+.1f}% vs LM thuần"
        )
        print(f"  {lam:>5} {bpt:>11.4f} {bpb:>10.4f}   {note}")
        if bpb < best_bb:
            best_bb, best_lam = bpb, lam

    print()
    if best_lam == 1.0:
        print("⇒ LM THUẦN thắng — kNN KHÔNG cải thiện perplexity. Xác nhận bằng CON SỐ: kNN không")
        print("  giúp density ở quy mô/LM này. Củng cố §7.17d 'LM 11.4M là trần'. Bỏ kNN, hoặc")
        print("  chỉ giữ cho SINH (phá lặp) chứ không cho density.")
    else:
        imp = 100 * (base_bt - bits[best_lam] / max(n_tok, 1)) / base_bt
        print(f"⇒ kNN CẢI THIỆN: λ={best_lam} tốt nhất, {imp:+.1f}% bits/token vs LM thuần.")
        print("  kNN có giá trị ĐO ĐƯỢC (không chỉ phá lặp). Đáng giữ. Việc SINH tốt vẫn cần LM")
        print("  lớn hơn (§7.17d) nhưng kNN đã chứng minh không phải đồ trang trí.")
    print("\n⚠️ Chỉ đo density (perplexity), KHÔNG đo 'sinh có đúng sự thật' — hai câu khác nhau.")
    print("  kNN cải thiện bits/byte mà SINH vẫn trôi chủ đề (§7.17d) là hoàn toàn có thể xảy ra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
