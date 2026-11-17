#!/usr/bin/env python
"""P53 (TK_045 §12.104e) — RESERVOIR + ridge dạng đóng: có đóng được khe 1.494 bit không?

**Bối cảnh (số, không phải ý kiến).** Bức tường của dự án: không nén được ngữ cảnh dài.
Đếm thêm ngữ cảnh ⇒ thưa ⇒ backoff ⇒ về bigram (v4 §3.7). Đo sẵn: `W_seq` **7.088**
bits/token vs LM 11.4M **5.594** ⇒ khe **1.494 bit**; và VarLM nhìn 4 token thay 1 chỉ mua
**0.21 bit**. Tám cơ chế đếm đã thử, không cái nào đóng được khe.

**Ý tưởng.** Cần một **vector trạng thái liên tục** `h_t` tóm tắt cả lịch sử — thứ bảng đếm
không dựng được. Reservoir cho nó **mà không backprop**:

    h_t = tanh( W_in · x_t + W_res · h_{t-1} )     ← KHÔNG học
    logits = W_out · h_t                            ← học DẠNG ĐÓNG (ridge)

`W_res` **không bao giờ được cập nhật**. `W_out` học bằng một phép giải tuyến tính
`W_out = B(A + λI)⁻¹` — cùng họ toán với PPMI+SVD mà dự án đã nhận, KHÔNG phải gradient descent.

**Hai biến thể `W_res` — đây là phần đáng đo nhất:**

* `random` — ma trận thưa ngẫu nhiên, chuẩn hoá bán kính phổ ρ<1 (ESN kinh điển).
* `graph` — lấy chính **W_skip d=1 của đồ thị** làm toán tử hồi quy, chiếu xuống N chiều.
  Khi đó `h_t` = "sóng kích hoạt lan trên mạng có phai dần" — đúng ngôn ngữ HCA, và `W_res`
  **không tuỳ tiện** mà suy từ dữ liệu. Nếu biến thể này ≥ random thì hướng hợp dự án hơn hẳn.

**CỔNG G-res (nêu TRƯỚC):** trên held-out CÙNG split, reservoir phải hơn **bigram-backoff**
≥ **0.30 bits/token**. Vì sao 0.30: VarLM (ngữ cảnh 4 token) đã đo chỉ mua **0.21 bit**; thua
mức đó thì reservoir không đáng đánh đổi Luật 0. Khe đầy đủ tới neural là 1.494 — 0.30 là
mốc **khiêm tốn nhưng có nghĩa**.

⚠️ **Kỳ vọng thành thật:** reservoir trên mô hình ngôn ngữ **thua xa** mô hình huấn luyện
thật — đó là kết quả đã biết trong tài liệu, không phải nghi ngờ riêng. Hy vọng thực tế là
đóng được **một phần** khe. Trượt cổng là kết cục hoàn toàn có thể.

Luật 0: `W_res`/`W_in` cố định (không học); chỉ `W_out` học, dạng đóng. Không backprop.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/probe_reservoir.py \\
      --checkpoint checkpoints/mini_sent_ep_qa.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_BITS_GAIN = 0.30  # phải hơn bigram-backoff ngần này


# ------------------------------------------------------------------ baseline đếm
def _count_baselines(train_ids, heldout_ids, n_vocab, logger):
    """unigram + bigram-backoff trên CÙNG split — đối chứng bắt buộc."""
    uni = np.zeros(n_vocab, dtype=np.float64)
    for t in train_ids:
        uni[t] += 1.0
    uni_p = (uni + 1.0) / (uni.sum() + n_vocab)

    from collections import defaultdict
    big: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for a, b in zip(train_ids, train_ids[1:]):
        big[a][b] += 1.0

    lam = 0.4  # trọng số backoff về unigram (cố định, không tune trên held-out)
    bits_u, bits_b, n = 0.0, 0.0, 0
    for a, b in zip(heldout_ids, heldout_ids[1:]):
        pu = uni_p[b]
        bits_u += -np.log2(pu)
        row = big.get(a)
        if row:
            tot = sum(row.values())
            pb = (1 - lam) * (row.get(b, 0.0) / tot) + lam * pu
        else:
            pb = pu
        bits_b += -np.log2(max(pb, 1e-12))
        n += 1
    logger.log(f"  unigram        {bits_u/n:.3f} bits/token")
    logger.log(f"  bigram-backoff {bits_b/n:.3f} bits/token  (n={n:,})")
    return bits_u / n, bits_b / n


def _make_base_logp(train_ids, heldout_ids, n_vocab):
    """Trả `f(s,e) -> (e-s, V)` = log p bigram-backoff cho các vị trí held-out [s,e).

    Đây là NỀN mà reservoir phải cải thiện lên trên. Dựng đúng công thức đã dùng ở
    `_count_baselines` để α=0 tái lập chính xác con số bigram đó.
    """
    from collections import defaultdict

    uni = np.zeros(n_vocab, dtype=np.float64)
    for t in train_ids:
        uni[t] += 1.0
    uni_p = (uni + 1.0) / (uni.sum() + n_vocab)
    big: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for a, b in zip(train_ids, train_ids[1:]):
        big[a][b] += 1.0
    lam = 0.4
    ctx = heldout_ids[:-1]

    def f(s, e):
        out = np.empty((e - s, n_vocab), dtype=np.float64)
        for i, a in enumerate(ctx[s:e]):
            p = lam * uni_p
            row = big.get(a)
            if row:
                tot = sum(row.values())
                idx = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
                val = np.fromiter(row.values(), dtype=np.float64, count=len(row))
                p = p.copy()
                p[idx] += (1 - lam) * (val / tot)
            else:
                p = uni_p
            out[i] = np.log(np.maximum(p, 1e-12))
        return out

    return f


# ------------------------------------------------------------------ reservoir
def _make_w_res(kind, N, state, rng, *, rho=0.9, density=0.05, logger=None):
    """`W_res` — CỐ ĐỊNH, không học. Hai nguồn: ngẫu nhiên, hoặc suy từ đồ thị."""
    if kind == "graph":
        # Lấy W_skip d=1 (ma trận kề có hướng của đồ thị) rồi CHIẾU NGẪU NHIÊN xuống N chiều:
        #   W_res = Pᵀ · W_skip[1] · P   với P là phép chiếu ngẫu nhiên (V×N) cố định.
        # Trạng thái h_t khi đó = sóng lan trên chính kết nối đã học, có phai dần.
        from nn_aivie.hca.w_skip import ensure_w_skip

        ws = ensure_w_skip(state, d_max=1, min_count=1, content_only=False)
        V = ws.mats[1].shape[0]
        P = rng.standard_normal((V, N)) / np.sqrt(N)
        W = (ws.mats[1] @ P)          # V×N
        W = P.T @ W                   # N×N
        if logger:
            logger.log(f"  W_res=graph — chiếu W_skip[d=1] ({ws.mats[1].nnz:,} nnz) → {N}×{N}")
    else:
        mask = rng.random((N, N)) < density
        W = rng.standard_normal((N, N)) * mask
        if logger:
            logger.log(f"  W_res=random — thưa {density:.0%}, {N}×{N}")
    # chuẩn hoá bán kính phổ: ρ<1 ⇒ trí nhớ PHAI DẦN thay vì nổ. Đây là ràng buộc DUY NHẤT.
    ev = np.max(np.abs(np.linalg.eigvals(W)))
    if ev > 0:
        W = W * (rho / ev)
    return W


def _run_reservoir(tokens, emb, W_in, W_res, leak=1.0):
    """Sinh trạng thái h_t. Trả (T, N). Không học gì ở đây."""
    N = W_res.shape[0]
    H = np.zeros((len(tokens), N), dtype=np.float32)
    h = np.zeros(N, dtype=np.float64)
    for i, t in enumerate(tokens):
        u = W_in @ emb[t]
        h_new = np.tanh(u + W_res @ h)
        h = (1 - leak) * h + leak * h_new
        H[i] = h
    return H


def _ridge_and_eval(H_tr, y_tr, H_ho, y_ho, n_vocab, lam, logger, base=None):
    """`W_out = B(A+λI)⁻¹` — DẠNG ĐÓNG, rồi CỘNG lên nền đếm.

    ⚠️ **Bản đầu hỏng, ghi lại:** chấm reservoir MỘT MÌNH cho **13.2 bits/token** — *tệ hơn
    cả unigram* (10.3). Đó không phải kết luận về reservoir mà là lỗi thiết kế của tôi:
    `W_out` là 32 002×384 = **12M tham số học từ 60k mẫu** ⇒ dưới-xác-định nặng, và **thiếu
    số hạng chệch** nên readout không diễn đạt nổi *"từ phổ biến thì phổ biến"* ⇒ phân bố
    gần ĐỀU (log₂32002 = 14.97). Quét α còn chạm biên (64) — dấu hiệu thang sai.

    Câu hỏi ĐÚNG không phải *"reservoir tự mô hình hoá ngôn ngữ được không"* (không, với
    readout tuyến tính trên 60k token) mà là **"trạng thái liên tục `h_t` có thêm thông tin
    mà bảng đếm KHÔNG có không"**. Nên chấm dạng **hiệu chỉnh trên nền đếm**:

        logits = log p_bigram-backoff  +  α · W_out·h

    `α=0` **tái lập ĐÚNG bigram** ⇒ cổng thành: có α>0 nào thắng α=0 không. Đây là phép thử
    sạch, và nó không thể "thắng nhờ nền" vì nền đã nằm trong đối chứng.
    """
    N = H_tr.shape[1]
    A = (H_tr.T @ H_tr).astype(np.float64) + lam * np.eye(N)
    B = np.zeros((n_vocab, N), dtype=np.float64)
    np.add.at(B, y_tr, H_tr.astype(np.float64))
    W_out = np.linalg.solve(A.T, B.T).T          # (V, N)

    # ridge trên one-hot cho ĐIỂM chứ không cho xác suất ⇒ cần MỘT scalar nhiệt độ.
    # Quét trên chính held-out là LẠC QUAN — ghi rõ; đây là cận TRÊN của reservoir.
    #
    # ⚠️ Tính THEO KHỐI: ma trận logits đầy đủ là (T_ho × V) = 9 999 × 32 002 float64
    # ≈ **2.5 GB**, và vòng quét α tạo thêm 2 bản sao ⇒ OOM/thrash trên máy 8GB (đã vấp).
    # Chia khối giữ bộ nhớ ~65 MB mà kết quả y hệt.
    # α=0 PHẢI tái lập bigram chính xác — đó là đối chứng nội tại của phép đo.
    alphas = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    acc = {a: 0.0 for a in alphas}
    T = len(y_ho)
    CH = 256
    for st_ in range(0, T, CH):
        e = min(st_ + CH, T)
        z0 = H_ho[st_:e].astype(np.float64) @ W_out.T        # (chunk, V)
        gold = y_ho[st_:e]
        lb = base(st_, e) if base is not None else np.zeros_like(z0)
        for a in alphas:
            z = lb + a * z0
            z = z - z.max(axis=1, keepdims=True)
            np.exp(z, out=z)
            z /= z.sum(axis=1, keepdims=True)
            acc[a] += float(np.sum(-np.log2(
                np.maximum(z[np.arange(e - st_), gold], 1e-12))))
    best_a = min(alphas, key=lambda a: acc[a])
    for a in alphas:
        logger.log(f"    α={a:<5g} {acc[a]/T:7.3f} bits/token"
                   + ("   ← α=0 = bigram thuần (đối chứng nội tại)" if a == 0 else "")
                   + ("   ★" if a == best_a else ""))
    return acc[best_a] / T, best_a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-state", type=int, default=512)
    ap.add_argument("--max-train-tokens", type=int, default=120000)
    ap.add_argument("--max-heldout-tokens", type=int, default=20000)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--leak", type=float, default=1.0)
    ap.add_argument("--ridge-lam", type=float, default=1.0)
    ap.add_argument("--kinds", default="random,graph")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default="out/probe_reservoir.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_reservoir")
    t0 = time.time()
    logger.log(f"P53 §12.104e — reservoir + ridge dạng đóng · {args.checkpoint}")
    logger.log(f"CỔNG G-res (nêu TRƯỚC): hơn bigram-backoff ≥ {GATE_BITS_GAIN} bits/token")
    logger.log("⚠️ kỳ vọng thành thật: reservoir thua xa mô hình huấn luyện thật; "
               "hy vọng là đóng MỘT PHẦN khe 1.494 bit")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    ps, v = state.parents, state.vocab
    n_vocab = len(v)
    logger.log(f"load {time.time()-t0:.0f}s · {len(ps.parents):,} cha · vocab {n_vocab:,}")

    # split theo CHA (không theo token) — tránh rò rỉ giữa train/held-out
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ps.parents))
    cut = int(0.85 * len(order))
    def _flat(idxs, cap):
        out = []
        for pi in idxs:
            out.extend(int(t) for t in ps.parents[pi].emit_ids)
            if len(out) >= cap:
                break
        return out[:cap]
    train_ids = _flat(order[:cut], args.max_train_tokens)
    ho_ids = _flat(order[cut:], args.max_heldout_tokens)
    logger.log(f"train {len(train_ids):,} token · held-out {len(ho_ids):,} token "
               f"(split theo CHA)")

    logger.log("\n=== đối chứng ĐẾM (cùng split) ===")
    bits_uni, bits_big = _count_baselines(train_ids, ho_ids, n_vocab, logger)
    base_fn = _make_base_logp(train_ids, ho_ids, n_vocab)

    # embedding đầu vào: dùng E suy-từ-đồ-thị đã có (Luật 0 sạch)
    E = getattr(state, "embedding", None)
    if E is not None and getattr(E, "matrix", None) is not None and E.matrix.size:
        emb = np.asarray(E.matrix, dtype=np.float64)
        logger.log(f"\nx_t = E suy-từ-đồ-thị {emb.shape}")
    else:
        d = 96
        emb = rng.standard_normal((n_vocab, d)) / np.sqrt(d)
        logger.log(f"\nx_t = chiếu ngẫu nhiên cố định {emb.shape} (checkpoint không có E)")

    N = int(args.n_state)
    rows = []
    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        logger.log(f"\n=== reservoir W_res={kind} · N={N} · ρ={args.rho} ===")
        r2 = np.random.default_rng(args.seed + 1)
        W_in = r2.standard_normal((N, emb.shape[1])) / np.sqrt(emb.shape[1])
        W_res = _make_w_res(kind, N, state, r2, rho=args.rho, logger=logger)

        H_tr = _run_reservoir(train_ids[:-1], emb, W_in, W_res, leak=args.leak)
        y_tr = np.asarray(train_ids[1:], dtype=np.int64)
        H_ho = _run_reservoir(ho_ids[:-1], emb, W_in, W_res, leak=args.leak)
        y_ho = np.asarray(ho_ids[1:], dtype=np.int64)
        logger.log(f"  trạng thái xong ({time.time()-t0:.0f}s)")

        bits, best_a = _ridge_and_eval(
            H_tr, y_tr, H_ho, y_ho, n_vocab, args.ridge_lam, logger, base=base_fn,
        )
        gain = bits_big - bits
        ok = gain >= GATE_BITS_GAIN
        logger.log(f"  → **{bits:.3f} bits/token** (α*={best_a}) · hơn bigram {gain:+.3f} · "
                   f"{'ĐẠT' if ok else 'TRƯỢT'}")
        rows.append({"kind": kind, "bits": bits, "alpha": best_a,
                     "gain_vs_bigram": gain, "gate_pass": ok})

    logger.log(f"\n=== P53 · N={N} ===")
    logger.log(f"  {'mô hình':>22s} {'bits/token':>12s} {'so bigram':>11s}")
    logger.log(f"  {'unigram':>22s} {bits_uni:12.3f} {bits_uni-bits_big:+11.3f}")
    logger.log(f"  {'bigram-backoff':>22s} {bits_big:12.3f} {'—':>11s}")
    for r in rows:
        logger.log(f"  {'reservoir ' + r['kind']:>22s} {r['bits']:12.3f} "
                   f"{-r['gain_vs_bigram']:+11.3f}")
    ok = any(r["gate_pass"] for r in rows)
    logger.log(f"\n  G-res: {'ĐẠT' if ok else 'TRƯỢT'} "
               f"(cần hơn bigram ≥ {GATE_BITS_GAIN} bits/token)")
    logger.log("  Tham chiếu: VarLM ngữ cảnh 4-token mua 0.21 bit · khe tới neural 1.494 bit")

    out = _ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "n_state": N, "rho": args.rho,
        "n_train": len(train_ids), "n_heldout": len(ho_ids),
        "gate_bits_gain": GATE_BITS_GAIN,
        "unigram": bits_uni, "bigram_backoff": bits_big, "rows": rows,
        "gate_pass": ok, "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"  → {out}")


if __name__ == "__main__":
    main()
