#!/usr/bin/env python
"""P86 (TK_045 §13.61) — **SUY LUẬN BẬC HAI**: `con khỉ` có thừa hưởng `là động vật` không?

Ý tưởng của tác giả (2026-08-04). Hai ví dụ, **một** cơ chế:

    con mèo / con chó → là động vật, nghịch ngợm ; con khỉ → nghịch ngợm
    ⇒ suy ra: con khỉ → là động vật                       (chung ngữ cảnh PHẢI)

    anh ấy đang → làm / triển khai / thực hiện ; ok … đi → làm / triển khai
    ⇒ suy ra: ok thực hiện đi                             (chung ngữ cảnh TRÁI)

Cùng phép toán: **đồng hiện bậc hai**. Hai thực thể dùng chung nhiều ngữ cảnh ⇒ suy ra chúng
dùng chung cả ngữ cảnh **chưa từng thấy**.

## Đo TRƯỚC, xây SAU

Probe này **không viết cơ chế mới nào**. Nó chỉ hỏi: **tín hiệu có ở đó không**, và **có hơn
thứ hệ ĐÃ CÓ không**. Sau mười ba vòng mà mọi cơ chế "hợp lý" đều chết, đây là thứ tự bắt buộc.

## Giao thức: BỎ NGUYÊN MỘT CHA, không phải bỏ một ô

Với mỗi cặp `(a, c)` **chỉ đồng hiện trong ĐÚNG MỘT cha** `P`:
1. **bỏ hẳn `P`** khỏi ma trận — đúng nghĩa *"chưa hề train con khỉ là động vật"*;
2. hỏi ba bộ chấm: `c` được xếp thứ mấy trong toàn bộ ứng viên cho `a`?

Bỏ một **ô** thì các ô khác của cùng cha vẫn còn ⇒ rò rỉ. Bỏ nguyên **cha** mới đúng câu hỏi.

## Ba bộ chấm — sàn, hệ hiện có, và ứng viên

| | |
|---|---|
| `ngau_nhien` | sàn — xếp ngẫu nhiên |
| `embedding` | **hệ ĐÃ CÓ**: cosine trên `E` (32002×96). Nếu nó đã làm được thì cơ chế mới không mua thêm gì |
| `bac_hai` | ứng viên: `s = PPMI(M)[a,:] @ PPMI(M)`, tức đi qua các thực thể tương đồng |

Sàn `embedding` là mục quan trọng nhất. Không có nó thì `bac_hai` chỉ cần hơn ngẫu nhiên là
trông "thành công", trong khi hệ có thể đã làm được từ lâu.

## CỔNG G-2nd (nêu TRƯỚC khi chạy)

1. **≥200 cặp** giữ lại — dưới mức đó mọi tỷ lệ đều là nhiễu.
2. `bac_hai` hit@10 **≥ 5×** `ngau_nhien` — tín hiệu phải tồn tại rõ rệt.
3. `bac_hai` hit@10 **≥ 1,3×** `embedding` — phải **mua thêm** so với thứ đã có.

**ĐẠT = cả ba.** Trượt (3) mà qua (2) ⇒ ý tưởng đúng nhưng **hệ đã có rồi**, và việc cần làm
là *dùng `E` ở khâu sinh*, không phải xây cơ chế mới.

## Dự đoán ghi TRƯỚC (Claude)

- (2) **đạt** — bậc hai là phép toán chuẩn mực, tín hiệu gần như chắc chắn có.
- (3) **50/50, tôi thật sự không biết.** `E` là SVD của chính đồng hiện, tức **cùng họ** với
  bậc hai; rất có thể chúng đo trùng nhau. Nếu trùng thì kết luận hữu ích: đừng xây thêm.
- Bệnh sẽ thấy: **hub tần suất** — token phổ biến chiếm top của mọi truy vấn (§13.20). Nếu
  PPMI + bỏ thành phần kỳ dị đầu không đủ, `bac_hai` sẽ thua cả `embedding`.

Chạy: `PYTHONPATH=src python scripts/probe_second_order.py --n-pairs 300`
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_MIN_PAIRS = 200
GATE_VS_RANDOM = 5.0
GATE_VS_EMBED = 1.3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--gidf-min", type=float, default=4.0,
                    help="lọc rác §13.20 — hư từ đồng hiện với mọi thứ")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_second_order.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_second_order")
    t0 = time.time()
    logger.log(f"P86 bậc hai · {args.checkpoint} · cổng: ≥{GATE_MIN_PAIRS} cặp · "
               f"≥{GATE_VS_RANDOM}× ngẫu nhiên · ≥{GATE_VS_EMBED}× embedding")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)

    # --- token NỘI DUNG: bỏ cấu trúc + lọc gidf (rác §13.20 đồng hiện với mọi thứ) -------
    keep = np.zeros(n, dtype=bool)
    for t in range(n):
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            if float(ps._gidf(t)) >= args.gidf_min:  # noqa: SLF001
                keep[t] = True
        except Exception:
            continue
    logger.log(f"token nội dung giữ lại: {int(keep.sum()):,}/{n:,} ({time.time()-t0:.0f}s)")

    # --- M[a,c] = số CHA chứa cả a và c ---------------------------------------------------
    par_toks: list[list[int]] = []
    for rec in ps.parents:
        toks = sorted({int(t) for t in rec.emit_ids if 0 <= int(t) < n and keep[int(t)]})
        par_toks.append(toks)
    pairs_of: dict[tuple[int, int], list[int]] = defaultdict(list)
    rows, cols = [], []
    for pi, toks in enumerate(par_toks):
        if len(toks) > 60:            # cha quá dài ⇒ O(k²) nổ; bỏ, ghi rõ
            continue
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                a, c = toks[i], toks[j]
                pairs_of[(a, c)].append(pi)
                rows += [a, c]
                cols += [c, a]
    M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=np.float64)
    M.sum_duplicates()
    logger.log(f"M: {M.nnz:,} ô khác 0 · {len(pairs_of):,} cặp ({time.time()-t0:.0f}s)")

    # --- PPMI trên M ----------------------------------------------------------------------
    tot = float(M.sum())
    rs = np.asarray(M.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        Mp = M.tocoo()
        pmi = np.log((Mp.data / tot) / ((rs[Mp.row] / tot) * (rs[Mp.col] / tot) + 1e-300))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi = np.maximum(pmi, 0.0)
    P = csr_matrix((pmi, (Mp.row, Mp.col)), shape=(n, n))
    P.eliminate_zeros()

    # --- embedding đã có sẵn --------------------------------------------------------------
    _emb = getattr(state, "embedding", None)
    E = np.asarray(_emb.matrix, dtype=np.float64)
    if E.shape[0] < n:                       # embedding có thể ngắn hơn vocab
        E = np.pad(E, ((0, n - E.shape[0]), (0, 0)))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    # --- cặp giữ lại: chỉ đồng hiện trong ĐÚNG MỘT cha ------------------------------------
    rng = random.Random(args.seed)
    cand = [(ac, pl[0]) for ac, pl in pairs_of.items() if len(pl) == 1]
    rng.shuffle(cand)
    cand = cand[: args.n_pairs]
    logger.log(f"cặp chỉ-một-cha: {len(cand):,} lấy mẫu ({time.time()-t0:.0f}s)")

    idx = np.flatnonzero(keep)
    hits = {"bac_hai": 0, "embedding": 0, "ngau_nhien": 0}
    ranks: dict[str, list[int]] = {k: [] for k in hits}
    nrand = float(len(idx))
    for (a, c), pi in cand:
        # BỎ NGUYÊN CHA: trừ đóng góp của cha `pi` khỏi hàng `a`
        toks = par_toks[pi]
        row = P[a].toarray().ravel().copy()
        for t in toks:
            if t != a:
                row[t] = 0.0            # xoá mọi liên kết a↔(token cùng cha đó)
        s2 = row @ P                    # bậc hai: đi qua thực thể tương đồng
        s2[a] = -np.inf
        se = En @ En[a]
        se[a] = -np.inf
        for name, sc in (("bac_hai", s2), ("embedding", se)):
            order = sc[idx]
            r = int((order > sc[c]).sum()) + 1
            ranks[name].append(r)
            if r <= args.topk:
                hits[name] += 1
        rr = rng.randint(1, int(nrand))
        ranks["ngau_nhien"].append(rr)
        if rr <= args.topk:
            hits["ngau_nhien"] += 1

    nn = len(cand)
    print(f"\n{'bộ chấm':<14}{'hit@%d' % args.topk:>9}{'MRR':>9}{'hạng trung vị':>16}")
    res = {}
    for k in ("ngau_nhien", "embedding", "bac_hai"):
        h = hits[k] / nn if nn else 0.0
        mrr = float(np.mean([1.0 / r for r in ranks[k]])) if nn else 0.0
        med = float(np.median(ranks[k])) if nn else 0.0
        res[k] = {"hit": h, "mrr": mrr, "median_rank": med}
        print(f"{k:<14}{h:>8.1%}{mrr:>9.3f}{med:>16,.0f}")

    g1 = nn >= GATE_MIN_PAIRS
    g2 = res["bac_hai"]["hit"] >= GATE_VS_RANDOM * max(res["ngau_nhien"]["hit"], 1e-9)
    g3 = res["bac_hai"]["hit"] >= GATE_VS_EMBED * max(res["embedding"]["hit"], 1e-9)
    print(f"\nCỔNG G-2nd: (1) {'✅' if g1 else '❌'} {nn} cặp · "
          f"(2) {'✅' if g2 else '❌'} vs ngẫu nhiên · (3) {'✅' if g3 else '❌'} vs embedding"
          f"  ⇒ {'ĐẠT' if (g1 and g2 and g3) else 'TRƯỢT'}")

    out = {"checkpoint": args.checkpoint, "n_pairs": nn, "topk": args.topk,
           "gate": {"min_pairs": GATE_MIN_PAIRS, "vs_random": GATE_VS_RANDOM,
                    "vs_embed": GATE_VS_EMBED},
           "ket_qua": res, "gate_pass": bool(g1 and g2 and g3),
           "seconds": round(time.time() - t0, 1)}
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
