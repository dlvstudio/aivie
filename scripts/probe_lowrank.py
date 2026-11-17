#!/usr/bin/env python3
"""
TK_042 G2 — THÍ NGHIỆM RẺ: hạng-thấp có ĐÁNH BẠI đếm-đầy-đủ trên held-out không?

    PYTHONPATH=src python -u scripts/probe_lowrank.py \
        --checkpoint checkpoints/network_v10_histwiki.pkl --n 200

## Giả thuyết (TK_042 §7.0)

`W_seq` là bảng V×V **99.99% rỗng**, mỗi ô ĐỘC LẬP: `W[chó][sủa]` không nói gì về
`W[cún][sủa]`. Đây là bài toán **Netflix**. Lời giải: **phân tích hạng thấp** — `W ≈ A·Bᵀ` với
`r ≪ V` **không đủ chỗ nhớ từng ô** ⇒ buộc phát hiện hàng nào giống hàng nào ⇒ **ĐIỀN VÀO các
số không** ⇒ `P(cún|"con ___") > 0` dù chưa từng thấy.

## Toán (Levy & Goldberg 2014: word2vec = phân tích ngầm ma trận PPMI)

    M[i,j] = max(0, ln( c(i,j)·N / (c(i)·c(j)) ))        # PPMI CÓ HƯỚNG (trái≠phải)
    M ≈ U_r Σ_r V_rᵀ  →  A = U_r√Σ_r ,  B = V_r√Σ_r      # M_r[i,j] = ⟨A[i], B[j]⟩

    PMI(i,j) = ln P(j|i) − ln P(j)   ⇒   ln P(j|i) = ln P(j) + PMI(i,j)
    ⇒ s(j|i) = ln P_uni(j) + M_r[i,j] ;  P(j|i) = softmax_j s(j|i)

Nội suy unigram **giống hệt baseline** để so CÔNG BẰNG:
    P_final = lam·P_lowrank + (1−lam)·P_uni

## Đọc kết quả

**Nhị phân.** hạng-r **< baseline bits/byte** ⇒ khái quát hoá THẬT (điền số 0 có ích).
Không ⇒ **TK_042 §7.4: ghi âm tính, DỪNG, quay về A hoặc C.** KHÔNG đắp G4 lên G2 âm tính.

Kỳ vọng ghi TRƯỚC (§7.1): corpus 19.6M quan sát vs ~21M cần cho r=64 ⇒ **NGAY RANH GIỚI** ⇒
**50/50**; điểm ngọt dự đoán r=32–64; r≥128 under-determined.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from eval_bits import load_heldout  # noqa: E402


def build_matrix(state):
    """W_seq → COO thưa (i=trái, j=phải). CÓ HƯỚNG — không đối xứng hoá."""
    W = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    V = len(state.vocab)
    I, J, D = [], [], []
    for i, j, w in W.items():
        if 0 <= i < V and 0 <= j < V and w > 0:
            I.append(i); J.append(j); D.append(float(w))
    C = sp.coo_matrix((D, (I, J)), shape=(V, V)).tocsr()
    return C, V


def ppmi(C):
    """PPMI có hướng, giữ THƯA (chỉ tính trên ô khác 0 — ô 0 có PPMI=0 theo định nghĩa max)."""
    N = C.sum()
    row = np.asarray(C.sum(axis=1)).ravel()
    col = np.asarray(C.sum(axis=0)).ravel()
    Coo = C.tocoo()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((Coo.data * N) / (row[Coo.row] * col[Coo.col]))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi = np.maximum(pmi, 0.0)
    keep = pmi > 0
    return sp.coo_matrix(
        (pmi[keep], (Coo.row[keep], Coo.col[keep])), shape=C.shape
    ).tocsr(), row, col, N


def eval_model(encoded, V, score_row_fn, p_uni, lam):
    """CE trên MỌI vị trí. `score_row_fn(w0) -> ln P(·|w0) chưa chuẩn hoá` hoặc None."""
    sum_bits = 0.0
    n_pos = 0
    n_bytes = 0
    for ids, nb in encoded:
        if len(ids) < 2:
            continue
        n_bytes += nb
        for i in range(1, len(ids)):
            w0, gold = ids[i - 1], ids[i]
            if gold >= V:
                p = 1e-12
            else:
                s = score_row_fn(w0)
                if s is None:
                    p_m = float(p_uni[gold])
                else:
                    s = s - s.max()
                    e = np.exp(s)
                    p_m = float(e[gold] / e.sum())
                p = lam * p_m + (1.0 - lam) * float(p_uni[gold])
            sum_bits += -math.log2(max(p, 1e-12))
            n_pos += 1
    return sum_bits / max(1, n_pos), sum_bits / max(1, n_bytes), n_pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--holdout", default="data/real_vi/essays.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranks", default="16,32,64,128")
    ap.add_argument("--lams", default="0.7,0.85,0.95")
    args = ap.parse_args()

    texts = load_heldout(_ROOT / args.holdout, args.n, args.seed)
    print(f"held-out: {len(texts)} đơn vị · {args.holdout}", flush=True)
    print(f"load {args.checkpoint} …", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        state, _ = load_checkpoint(args.checkpoint)
    encoded = [(state.vocab.encode_prompt(t), len(t.encode("utf-8"))) for t in texts]
    C, V = build_matrix(state)
    print(f"vocab={V:,} · cạnh W_seq={C.nnz:,} · độ đầy={100*C.nnz/(V*V):.4f}%", flush=True)

    uni = np.asarray(C.sum(axis=0)).ravel()
    p_uni = (uni + 1.0) / (uni.sum() + V)
    ln_p_uni = np.log(p_uni)
    lams = [float(x) for x in args.lams.split(",")]

    # ---- BASELINE: đếm đầy đủ (bigram chuẩn hoá) ----
    rowsum = np.asarray(C.sum(axis=1)).ravel()

    def base_row(w0):
        if w0 >= V or rowsum[w0] <= 0:
            return None
        r = np.asarray(C[w0].todense()).ravel()
        return np.log(np.maximum(r / rowsum[w0], 1e-12))

    print(f"\n{'mô hình':>16} {'lam':>5} {'bits/token':>11} {'bits/BYTE':>10}")
    best_base = None
    for lam in lams:
        bt, bb, n = eval_model(encoded, V, base_row, p_uni, lam)
        print(f"{'ĐẾM đầy đủ':>16} {lam:>5.2f} {bt:>11.3f} {bb:>10.3f}")
        if best_base is None or bb < best_base[1]:
            best_base = (lam, bb, bt)
    print(f"  → BASELINE tốt nhất: {best_base[1]:.3f} bits/byte (lam={best_base[0]})")

    # ---- HẠNG THẤP ----
    print("\nPPMI có hướng …", flush=True)
    M, _row, _col, _N = ppmi(C)
    print(f"  PPMI nnz={M.nnz:,}", flush=True)

    results = []
    for r in [int(x) for x in args.ranks.split(",")]:
        print(f"\nSVD r={r} …", flush=True)
        U, S, Vt = svds(M, k=r)
        idx = np.argsort(-S)
        U, S, Vt = U[:, idx], S[idx], Vt[idx]
        A = (U * np.sqrt(S)).astype(np.float32)
        B = (Vt.T * np.sqrt(S)).astype(np.float32)
        print(f"  A{A.shape} B{B.shape} · Σ: {S[0]:.2f} … {S[-1]:.2f}", flush=True)

        def lr_row(w0, A=A, B=B):
            if w0 >= V:
                return None
            return ln_p_uni + (B @ A[w0]).astype(np.float64)

        best_r = None
        for lam in lams:
            bt, bb, n = eval_model(encoded, V, lr_row, p_uni, lam)
            print(f"{'HẠNG r=%d' % r:>16} {lam:>5.2f} {bt:>11.3f} {bb:>10.3f}")
            if best_r is None or bb < best_r[1]:
                best_r = (lam, bb, bt)
        results.append((r, best_r))

    print("\n" + "=" * 62)
    print(f"BASELINE (đếm đầy đủ) : {best_base[1]:.3f} bits/byte")
    win = False
    for r, (lam, bb, bt) in results:
        d = bb - best_base[1]
        mark = "✅ THẮNG" if d < 0 else "❌ thua"
        win = win or d < 0
        print(f"  hạng r={r:<4} lam={lam:<5.2f}: {bb:.3f} bits/byte  (Δ {d:+.3f})  {mark}")
    print("=" * 62)
    print(
        "\n→ KHÁI QUÁT HOÁ THẬT: hạng-thấp điền số 0 CÓ ÍCH. Tiếp G3/G4."
        if win
        else "\n→ ÂM TÍNH. TK_042 §7.4: GHI ÂM TÍNH, DỪNG. KHÔNG đắp G4 lên.\n"
        "   (corpus 19.6M vs ~21M cần cho r=64 — đúng dự đoán ranh giới §7.1)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
