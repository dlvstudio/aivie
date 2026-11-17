#!/usr/bin/env python3
"""
PROBE TẦNG HẠNG MỤC — E phân bố có chứa lớp ngữ pháp/nghĩa để học VĂN PHONG không? (TK_042).

    PYTHONPATH=src python -u scripts/probe_category_layer.py

## Câu hỏi (tác giả 2026-07-20): "mạng học được văn phong/ngữ pháp không? tầng Hebbian khác nhau?"

Chép verbatim = mạng lưu chuỗi BỀ MẶT. Muốn SINH câu mới đúng văn phạm cần tầng TRỪU TƯỢNG:
nơ-ron HẠNG MỤC (loại-từ/vai-trò) + KHUÔN (chuỗi hạng-mục). Nguyên liệu = **E (PPMI+SVD)**:
từ đồng-ngữ-cảnh có E gần nhau ⇒ cluster E = proto-loại-từ TỰ NỔI (Luật 0 — không gán nhãn).

Probe RẺ (không train lại; chỉ cần `out/E_v11.npy` 12MB, KHÔNG cần graph 2GB) — chặn 1 đợt
build tầng vô ích (tiền lệ R-slot/order-2/E-episode). Hai câu hỏi:

1. **Cluster E → ĐỌC TAY:** các cụm có trông như loại-từ/lớp-nghĩa (danh/động/số/địa danh)
   không? Ngẫu nhiên ⇒ không có trừu tượng ở corpus này ⇒ tầng khuôn vô vọng, DỪNG.
2. **category-bigram vs surface-bigram trên held-out:** dự đoán token kế TỪ HẠNG MỤC của token
   hiện tại có khái quát tốt hơn bigram bề mặt không — NHẤT LÀ trên bigram CHƯA THẤY lúc train
   (bề mặt = thuộc lòng, hạng mục = khái quát). Đây là "ngữ pháp khái quát vs bề mặt chép".

## Điều kiện phân định (GHI TRƯỚC)

* Interp `λ·P_surface + (1-λ)·P_cat`: nếu λ<1 HẠ bits held-out so λ=1 (surface thuần) ⇒ hạng mục
  mang cấu trúc khái quát ⇒ tầng khuôn ĐÁNG build. λ=1 thấp nhất ⇒ không juice, xác nhận trần.
* Trên bigram CHƯA THẤY: category PHẢI thắng surface rõ nếu nó khái quát thật.
* (cùng phương pháp λ-sweep với kNN §7.17e — thước quen, phạt cả copy lẫn bịa.)
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

from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402


def kmeans_cosine(X: np.ndarray, K: int, iters: int, seed: int, log) -> np.ndarray:
    """K-means cosine (X đã chuẩn hoá L2 ⇒ cosine = dot). numpy thuần — sklearn KHÔNG có trong
    venv dự án. Trả nhãn cluster mỗi hàng."""
    rng = np.random.default_rng(seed)
    N, D = X.shape
    C = X[rng.choice(N, K, replace=False)].copy()
    lab = np.zeros(N, dtype=np.int32)
    for it in range(iters):
        lab = (X @ C.T).argmax(1).astype(np.int32)  # gán: centroid gần nhất
        sums = np.zeros((K, D), dtype=np.float64)
        cnt = np.zeros(K, dtype=np.float64)
        np.add.at(sums, lab, X)
        np.add.at(cnt, lab, 1.0)
        C = sums / np.maximum(cnt[:, None], 1.0)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        if it % 5 == 0:
            log(f"  kmeans iter {it}/{iters}")
    return lab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--e-path", default="out/E_v11.npy")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--manifest", default="config/train_v10.yaml", help="corpus đếm bigram (v10 nhanh)")
    ap.add_argument("--holdout", default="data/sft/qa_heldout.jsonl")
    ap.add_argument("--n-holdout", type=int, default=300)
    ap.add_argument("--k", type=int, default=200, help="số hạng mục (cluster E)")
    ap.add_argument("--kmeans-iters", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=20_000_000, help="cắt corpus đếm bigram")
    ap.add_argument("--show-clusters", type=int, default=25)
    ap.add_argument("--members", type=int, default=12)
    ap.add_argument("--smooth", type=float, default=0.1, help="add-k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok.tokens)
    E = np.load(_ROOT / args.e_path).astype(np.float32)
    log(f"E {E.shape} · vocab {V:,} · fingerprint {tok.fingerprint()}")
    if E.shape[0] != V:
        raise SystemExit(f"✗ E rows {E.shape[0]} ≠ vocab {V} — E và tokenizer KHÁC (TK_041 §5.2)")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    log(f"corpus (đếm bigram, cắt {args.max_tokens/1e6:.0f}M) …")
    ids = load_corpus(_ROOT / args.manifest, tok, None, log)[: args.max_tokens]
    freq = np.bincount(ids, minlength=V)

    log(f"k-means K={args.k} trên E …")
    lab = kmeans_cosine(En, args.k, args.kmeans_iters, args.seed, log)

    # ── PART 1: đọc tay các cụm (top thành viên theo tần suất corpus) ──
    log(f"═══ PART 1: {args.show_clusters} CỤM (top {args.members} theo tần suất) — ĐỌC TAY ═══")
    order = sorted(range(args.k), key=lambda k: -freq[lab == k].sum())  # cụm nội-dung-nặng trước
    for k in order[: args.show_clusters]:
        members = np.where(lab == k)[0]
        top = members[np.argsort(-freq[members])][: args.members]
        surfs = [repr(tok.decode([int(i)])) for i in top if freq[i] > 0]
        log(f"  c{k:3d} (n={len(members)}): {' '.join(surfs)}")

    # ── PART 2: category-bigram vs surface-bigram trên held-out ──
    log("═══ PART 2: dự đoán token kế — surface vs category (held-out) ═══")
    s_next: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    s_tot = np.zeros(V, dtype=np.float64)
    c_next: list[dict[int, int]] = [defaultdict(int) for _ in range(args.k)]
    c_tot = np.zeros(args.k, dtype=np.float64)
    for a, b in zip(ids[:-1], ids[1:]):
        a, b = int(a), int(b)
        s_next[a][b] += 1
        s_tot[a] += 1
        ca = int(lab[a])
        c_next[ca][b] += 1
        c_tot[ca] += 1

    def p_surface(a, b):
        return (s_next[a].get(b, 0) + args.smooth) / (s_tot[a] + args.smooth * V)

    def p_cat(a, b):
        ca = int(lab[a])
        return (c_next[ca].get(b, 0) + args.smooth) / (c_tot[ca] + args.smooth * V)

    texts = load_heldout(_ROOT / args.holdout, args.n_holdout, args.seed)
    pairs = []
    for t in texts:
        e = tok.encode(t)
        pairs.extend(zip(e[:-1], e[1:]))
    log(f"held-out {Path(args.holdout).name}: {len(texts)} đơn vị · {len(pairs):,} bigram")

    lams = [1.0, 0.7, 0.5, 0.3, 0.0]
    for lam in lams:
        s = sum(-math.log2(max(1e-12, lam * p_surface(a, b) + (1 - lam) * p_cat(a, b)))
                for a, b in pairs)
        tag = ("surface thuần" if lam == 1.0 else "category thuần" if lam == 0.0 else "")
        log(f"  λ={lam:.1f} {tag:14s} {s/max(1,len(pairs)):.3f} bits/token")

    # chia theo bigram ĐÃ THẤY / CHƯA THẤY lúc train — test khái quát
    seen = [(a, b) for a, b in pairs if s_next[int(a)].get(int(b), 0) > 0]
    unseen = [(a, b) for a, b in pairs if s_next[int(a)].get(int(b), 0) == 0]

    def avg_bits(ps, fn):
        return sum(-math.log2(max(1e-12, fn(int(a), int(b)))) for a, b in ps) / max(1, len(ps))

    log(f"  ── CHƯA THẤY lúc train (n={len(unseen):,}) — test khái quát ──")
    log(f"     surface {avg_bits(unseen, p_surface):.3f} · category {avg_bits(unseen, p_cat):.3f} bits")
    log(f"  ── ĐÃ THẤY (n={len(seen):,}) — surface thuộc lòng ──")
    log(f"     surface {avg_bits(seen, p_surface):.3f} · category {avg_bits(seen, p_cat):.3f} bits")

    log("═══ ĐỌC: (1) cụm PART1 có ra loại-từ/lớp-nghĩa? (2) λ<1 HẠ bits ⇒ tầng khuôn đáng; "
        "category thắng trên CHƯA-THẤY ⇒ khái quát thật ═══")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
