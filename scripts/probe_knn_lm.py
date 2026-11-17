#!/usr/bin/env python3
"""
TK_042 G8 — k=1 (CHÉP) vs k=64 (TRỘN): khác nhau bao nhiêu bits?

    PYTHONPATH=src python -u scripts/probe_knn_lm.py --checkpoint checkpoints/network_mini_bpe.pkl

## Ý tưởng (trao đổi với tác giả 2026-07-17)

> *"Có cách nào nén theo ngữ cảnh, cùng ngữ cảnh thì kéo lại gần nhau — lúc RUNTIME,
> chứ không phải train từ đầu?"*

Cái đó có tên: **similarity-based smoothing** (Dagan/Lee/Pereira 1999), **cache model**
(Grave 2017), **kNN-LM** (Khandelwal 2020). Và phát hiện đáng nói:

> **nn-aivie ĐÃ LÀ một kNN-LM** — `ParentStore` = datastore, `rank_matches` = tìm láng giềng,
> agenda = phần tiếp theo của láng giềng. Chỉ là một cái **k=1**: nó lấy cha giống nhất rồi
> **dội nguyên chuỗi con** ra làm agenda.
>
> ⇒ **"verbatim copy" chính là triệu chứng của k=1.** Một kNN-LM lấy 1 láng giềng và chép nó
> thì đúng là máy chép. Lấy k=64 và trộn theo độ giống thì ra **PHÂN BỐ**.

Probe này hỏi đúng một câu chưa ai hỏi: **k=1 vs k=64 khác nhau bao nhiêu bits?**
KHÔNG train gì — chỉ dùng `E` (PPMI+SVD, d=96) và `parents` đã có.

## Cách làm

    ctx_vec(ids, i) = Σ_{j=1..L} decay^(j-1) · E[ids[i-j]]        (chuẩn hoá L2)
    datastore       = {(ctx_vec tại vị trí i, ids[i])}  lấy mẫu từ parents
    P(next|ctx)     ∝ Σ_{k láng giềng} softmax(sim/τ) · 1[next_i]
    P_final         = lam·P_knn + (1−lam)·P_uni          (giống baseline → so CÔNG BẰNG)

## Đọc kết quả (ghi TRƯỚC)

Trần đã đo (TK_042 §7.5b): mọi mô hình hiệu quả bằng ~1 token ngữ cảnh chặn ở **6.42
bits/token**. Biểu diễn ngữ cảnh ở đây là **túi có trọng số vị trí** — thô, KHÔNG phải
attention (attention cần trọng số chiếu HỌC được). Nên **kỳ vọng: k=64 > k=1 nhưng KHÔNG
phá trần**. Similarity-based smoothing 1999 cho ~20% giảm perplexity so KN ≈ ~0.3 bit.

Giá trị: nếu k=64 kéo được đáng kể mà không train gì ⇒ đổi cách nhìn "agenda là miếng vá".
Nếu bất động ⇒ biết chắc hơn rằng nút thắt là **biểu diễn ngữ cảnh**, không phải cách trộn.
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


def ctx_matrix(seqs, E, L, decay, max_rows, rng, keep_ctx=False):
    """(ctx_vec, next_token[, ctx_ids]) từ các chuỗi. ctx = túi có trọng số vị trí giảm dần.

    `keep_ctx`: giữ L token ngữ cảnh để ĐỌC TAY (§7.8h) — 3.22 bits/token là con số trừu
    tượng cho tới khi nhìn thấy cosine chọn SAI cái gì.
    """
    d = E.shape[1]
    X = np.zeros((max_rows if max_rows < 10**8 else 200000, d), dtype=np.float32)
    Y = np.zeros(len(X), dtype=np.int32)
    C = np.zeros((len(X), L), dtype=np.int32) if keep_ctx else None
    n = 0
    w = np.array([decay**j for j in range(L)], dtype=np.float32)
    for ids in seqs:
        for i in range(1, len(ids)):
            if n >= len(X):
                break
            v = np.zeros(d, dtype=np.float32)
            for j in range(min(L, i)):
                t = ids[i - 1 - j]
                if t < len(E):  # vocab legacy mọc lúc encode → id ngoài E → vector 0
                    v += w[j] * E[t]
            X[n] = v
            Y[n] = ids[i]
            if C is not None:
                for j in range(min(L, i)):
                    C[n, j] = ids[i - 1 - j]
            n += 1
        if n >= len(X):
            break
    X, Y = X[:n], Y[:n]
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X /= np.maximum(nrm, 1e-8)
    return (X, Y, C[:n]) if keep_ctx else (X, Y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--holdout", default="data/real_vi/essays.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--L", type=int, default=8, help="số token ngữ cảnh")
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--datastore", type=int, default=400_000)
    ap.add_argument("--ks", default="1,4,16,64,256")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lams", default="0.7,0.9")
    ap.add_argument("--show-cases", type=int, default=0, help="in N ca ĐỌC TAY")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"load {args.checkpoint} …", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        state, _ = load_checkpoint(args.checkpoint)
    if state.embedding is None:
        print("✗ checkpoint không có E — cần embedding (PPMI+SVD)")
        return 1
    E = state.embedding.matrix.astype(np.float32)
    V = len(state.vocab)
    print(f"vocab={V:,} · cha={len(state.parents.parents):,} · E{E.shape}", flush=True)

    # datastore từ parents
    seqs = [list(p.source_ids) for p in state.parents.parents if len(p.source_ids) > 2]
    rng.shuffle(seqs)
    print(f"datastore từ {len(seqs):,} cha …", flush=True)
    X, Y, Xc = ctx_matrix(seqs, E, args.L, args.decay, args.datastore, rng, keep_ctx=True)
    print(f"  {len(X):,} cặp (ctx, next)", flush=True)

    # held-out
    texts = load_heldout(_ROOT / args.holdout, args.n, args.seed)
    enc = [state.vocab.encode_prompt(t) for t in texts]
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    Q, G, Qc = ctx_matrix(enc, E, args.L, args.decay, 10**9, rng, keep_ctx=True)
    print(f"held-out: {len(Q):,} vị trí · {nbytes:,} byte", flush=True)

    # unigram để nội suy (giống baseline)
    _bi, uni, total = build_counts(state)
    p_uni = (uni + 1.0) / (total + V)

    ks = [int(x) for x in args.ks.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    kmax = max(ks)
    # CHIA KHỐI: S đầy đủ = len(Q)×len(X)×4B — với 3.3k query × 400k datastore = 5.2GB,
    # thừa sức giết máy 8GB (TK_042 §7.7a). Chỉ giữ top-kmax mỗi query.
    print(f"\ntìm láng giềng ({len(Q):,} × {len(X):,}, chia khối) …", flush=True)
    CH = 256
    top = np.zeros((len(Q), kmax), dtype=np.int32)
    topsim = np.zeros((len(Q), kmax), dtype=np.float32)
    for a in range(0, len(Q), CH):
        b = min(a + CH, len(Q))
        s_blk = Q[a:b] @ X.T
        idx = np.argpartition(-s_blk, kmax, axis=1)[:, :kmax]
        sm = np.take_along_axis(s_blk, idx, 1)
        order = np.argsort(-sm, axis=1)  # sắp giảm dần → top[:, :k] đúng là k gần nhất
        top[a:b] = np.take_along_axis(idx, order, 1)
        topsim[a:b] = np.take_along_axis(sm, order, 1)
        del s_blk

    # RECALL@k — tách RETRIEVAL khỏi CALIBRATION.
    #
    # k=1 là mô hình TỰ TIN CỰC ĐOAN: P(gold)=1 nếu hàng xóm duy nhất đúng, =0 nếu sai →
    # sai thì trả giá khủng khiếp bằng bits. Nên bits giảm khi k↑ CÓ THỂ chỉ là "bớt tự tin"
    # (calibration), KHÔNG phải "tìm được nhiều hơn" (retrieval). Recall@k tách hai cái:
    #   recall@64 ≫ recall@1  →  trộn tìm được THẬT
    #   recall@64 ≈ recall@1  →  chỉ là hedging
    # (Thước CŨ đo log2(rank) nên MÙ với calibration — đúng bẫy đã giấu chuyện KN smoothing.)
    # ⚠️ ĐỐI CHỨNG NGẪU NHIÊN — BẮT BUỘC, và bản đầu THIẾU.
    #
    # `recall@k` một mình VÔ NGHĨA: với k=256 hàng xóm và phân bố token kiểu Zipf, gold (nhất
    # là token phổ biến 'của','là','cả') lọt vào top-256 **DO MAY**, không do liên quan. Đọc
    # tay xác nhận (§7.8h ca "Hãy"): hàng xóm "đúng" ở hạng 255 có sim=0.819 — nó KHÔNG phải
    # ngữ cảnh "Hãy", nó lọt vào do trùng token phổ biến.
    # ⇒ Phải so với recall của k hàng xóm NGẪU NHIÊN. Chênh lệch mới là tín hiệu THẬT.
    rnd = np.random.default_rng(0).integers(0, len(X), size=top.shape)
    print(f"\n{'k':>5} {'recall@k':>9} {'NGẪU NHIÊN':>11} {'lift':>7}  ← lift≈1 ⇒ recall là MAY")
    recall = {}
    for k in ks:
        hits = (Y[top[:, :k]] == G[:, None]).any(axis=1)
        rhits = (Y[rnd[:, :k]] == G[:, None]).any(axis=1)
        recall[k] = float(hits.mean())
        rr = float(rhits.mean())
        print(f"{k:>5} {100*recall[k]:>8.1f}% {100*rr:>10.1f}% {recall[k]/max(rr,1e-9):>7.2f}×")

    print(f"\n{'k':>5} {'lam':>5} {'bits/token':>11} {'bits/BYTE':>10}")
    res = {}
    for k in ks:
        idx = top[:, :k]
        sims = topsim[:, :k]
        for lam in lams:
            sb = 0.0
            for r in range(len(Q)):
                w = np.exp((sims[r] - sims[r].max()) / args.tau)
                w /= w.sum()
                p_knn = 0.0
                gold = G[r]
                hit = Y[idx[r]] == gold
                if hit.any():
                    p_knn = float(w[hit].sum())
                p = lam * p_knn + (1 - lam) * float(p_uni[gold])
                sb += -math.log2(max(p, 1e-12))
            bt, bb = sb / len(Q), sb / nbytes
            print(f"{k:>5} {lam:>5.2f} {bt:>11.3f} {bb:>10.3f}")
            res.setdefault(k, []).append(bb)

    # ---- ORACLE: TRẦN TRÊN của HÀM CHẤM ----
    #
    # recall@256=57.3% ⇒ token đúng CÓ MẶT trong 256 hàng xóm ở 57.3% vị trí, mà bits vẫn thua
    # đếm ⇒ THÔNG TIN CÓ, MÔ HÌNH KHÔNG DÙNG ĐƯỢC (softmax(sim/τ) trải khối lượng SAI).
    # Oracle = chấm HOÀN HẢO: dồn hết khối lượng vào hàng xóm có token ĐÚNG, nếu nó có mặt.
    # Tách dư địa "sửa HÀM CHẤM" khỏi dư địa "sửa BIỂU DIỄN":
    #   oracle ≪ ĐẾM  → hàng xóm ĐỦ TỐT; hàm chấm là đòn bẩy THẬT (học 1 scorer = yêu cầu NHỎ)
    #   oracle ≈ ĐẾM  → chấm hoàn hảo cũng vô ích ⇒ gốc là BIỂU DIỄN ⇒ về G7, không có lối tắt
    #
    # ⚠️ Oracle KHÔNG ĐẠT ĐƯỢC — nó nhìn trộm đáp án. Số thật của một scorer HỌC ĐƯỢC nằm GIỮA
    # (cosine hiện tại) và (oracle). Oracle chỉ nói DƯ ĐỊA TỐI ĐA, không nói lấy được bao nhiêu.
    print(f"\n{'k':>5} {'lam':>5} {'ORACLE bits/tok':>16} {'bits/BYTE':>10}  ← chấm HOÀN HẢO")
    oracle = {}
    for k in ks:
        hit = (Y[top[:, :k]] == G[:, None]).any(axis=1)
        for lam in sorted(set(lams + [0.95, 0.99])):
            sb = 0.0
            for r in range(len(Q)):
                p_knn = 1.0 if hit[r] else 0.0
                p = lam * p_knn + (1 - lam) * float(p_uni[G[r]])
                sb += -math.log2(max(p, 1e-12))
            bb = sb / nbytes
            print(f"{k:>5} {lam:>5.2f} {sb/len(Q):>16.3f} {bb:>10.3f}")
            oracle[k] = min(oracle.get(k, 1e9), bb)

    print("\n" + "=" * 52)
    b1 = min(res[ks[0]])
    for k in ks:
        b = min(res[k])
        print(f"  k={k:<4}: {b:.3f} bits/byte   (Δ vs k={ks[0]}: {b-b1:+.3f})")
    print("=" * 52)
    r1, rmax = recall[ks[0]], recall[ks[-1]]
    print(f"\n  recall@{ks[0]} = {100*r1:.1f}%  →  recall@{ks[-1]} = {100*rmax:.1f}%  "
          f"(×{rmax/max(r1,1e-9):.1f})")
    print("  ⇒ recall tăng MẠNH = trộn tìm được THẬT (retrieval).")
    print("    recall gần như đứng yên = cái lợi chỉ là BỚT TỰ TIN (calibration).")
    if args.show_cases:
        tok = state.vocab.token_of
        def txt(ids_):
            return " ".join(tok(int(t)) for t in ids_[::-1] if t) or "∅"
        K = ks[-1]
        hit = (Y[top[:, :K]] == G[:, None]).any(axis=1)
        # ca THÚ VỊ: gold CÓ trong hàng xóm, nhưng cosine xếp nó HẠNG THẤP
        cand = []
        for r in np.where(hit)[0]:
            pos = np.where(Y[top[r, :K]] == G[r])[0]
            cand.append((int(pos[0]), r))
        cand.sort(reverse=True)  # hạng của hàng xóm ĐÚNG càng thấp càng minh hoạ
        print(f"\n{'='*62}\n  ĐỌC TAY — cosine chấm SAI ở đâu ({args.show_cases} ca)\n{'='*62}")
        for rank_of_right, r in cand[: args.show_cases]:
            print(f"\n▶ ngữ cảnh : …{txt(Qc[r])}")
            print(f"  GOLD     : {tok(int(G[r]))!r}")
            print(f"  cosine dồn khối lượng vào top-5:")
            for j in range(5):
                nb = top[r, j]
                ok = "✓" if Y[nb] == G[r] else "✗"
                print(f"     #{j+1} sim={topsim[r,j]:.3f} {ok}  …{txt(Xc[nb])[-46:]:>46}  → {tok(int(Y[nb]))!r}")
            print(f"  ⇒ hàng xóm ĐÚNG nằm ở HẠNG {rank_of_right+1}/{K} (sim={topsim[r, rank_of_right]:.3f})")
            print(f"    → đáp án CÓ trong túi, cosine cân nó gần như BẰNG 0.")

    print(f"\n  ORACLE (chấm HOÀN HẢO) @k={ks[-1]}: {oracle[ks[-1]]:.3f} bits/byte")
    print("  ⇒ oracle ≪ ĐẾM : hàng xóm ĐỦ TỐT — hàm chấm là đòn bẩy THẬT (học 1 scorer nhỏ)")
    print("    oracle ≈ ĐẾM : chấm hoàn hảo cũng vô ích ⇒ gốc là BIỂU DIỄN ⇒ về G7")
    print("  ⚠ Oracle KHÔNG đạt được (nhìn trộm đáp án) — nó là DƯ ĐỊA TỐI ĐA, không phải đích.")
    print("\n⚠ Đối chiếu baseline ĐẾM trên CÙNG checkpoint: chạy scripts/eval_bits.py")
    print("  Trần đã đo (§7.5b): ~1 token ngữ cảnh ⇒ 6.42 bits/token. Túi-trọng-số-vị-trí")
    print("  KHÔNG phải attention (cần trọng số chiếu HỌC được) ⇒ kỳ vọng: k↑ có lợi, KHÔNG phá trần.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
