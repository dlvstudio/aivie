#!/usr/bin/env python
"""P87 (TK_045 §13.62) — bậc hai **CÓ RÀNG BUỘC KHUNG**: đúng câu hỏi của tác giả.

§13.61 đo bậc hai trên **toàn bộ 31.649 ứng viên** và được hit@10 chỉ 2,3%. Nhưng tôi đã ghi
ngay lúc đó rằng **phép đo có thể sai, không phải ý tưởng**: ví dụ của tác giả hẹp hơn nhiều —

    `con khỉ là` → `động vật`

Ở đây **khung** (`là`) đã ràng buộc tập ứng viên xuống rất nhỏ. Probe này đo đúng tình huống
đó, và thêm một **đối chứng mà §13.61 KHÔNG có** — thứ quyết định:

## Đối chứng quyết định: KHUNG-MỘT-MÌNH

Với `con khỉ là → ?`, rất có thể `động vật` được đoán đúng **chỉ nhờ khung** — vì `là` thường
theo sau bởi `động vật` trong corpus, **bất kể chủ thể là gì**. Nếu vậy thì loại suy **không
mua thêm gì**, và ý tưởng chỉ đang ăn theo thống kê bigram sẵn có.

⇒ Câu hỏi thật của vòng này **không** phải *"bậc hai có xếp đúng không"* mà là:

> **Biết CHỦ THỂ có giúp gì thêm, so với chỉ biết KHUNG?**

## Giao thức

Với mỗi cha `P` giữ lại (bỏ **nguyên cha**, đúng nghĩa *"chưa hề train"*):
- `x` = một token nội dung trong `P`; `f` = token **ngay trước** `x` (khung);
- `a` = token nội dung **đứng trước `f`** (chủ thể);
- chỉ lấy ca mà bigram `(f, x)` xuất hiện ở **đúng một** cha — tức chứng cứ trực tiếp bị xoá
  sạch khi bỏ cha đó;
- tập ứng viên `C` = mọi token **từng theo sau `f`** ở phần còn lại, **cộng `x`** nếu thiếu
  (§13.31: không tiêm đích thì đo nhầm truy hồi thay vì xếp hạng — có ghi tỷ lệ `x` **tự tới
  được**).

Bốn bộ chấm trên **cùng** tập `C`:

| | |
|---|---|
| `ngau_nhien` | sàn |
| `khung` | **đối chứng quyết định** — chỉ `PPMI_seq(f → ·)`, không biết chủ thể |
| `khung_x_embedding` | khung × `cos(E[a], E[·])` — hệ **ĐÃ CÓ** |
| `khung_x_bac_hai` | khung × `PPMI(M)[a,:] @ PPMI(M)` — ý tưởng của tác giả |

## CỔNG G-frame (nêu TRƯỚC khi chạy)

1. **≥200 ca** giữ lại.
2. `khung_x_bac_hai` MRR **≥ 1,30×** `khung` — biết chủ thể phải **mua thêm**. Mục **chính**;
   trượt nó thì ý tưởng không có chỗ đứng riêng.
3. `khung_x_bac_hai` MRR **≥ 1,10×** `khung_x_embedding` — phải mua thêm so với `E` đã có.

**ĐẠT = cả ba.** Đạt (2) mà trượt (3) ⇒ ý tưởng đúng nhưng nên hiện thực bằng `E`, rẻ hơn.

## Dự đoán ghi TRƯỚC (Claude)

- (2) **đạt, nhưng sát** — khoảng 1,3–1,8×. Chủ thể phải có ích, nếu không thì mọi câu
  `X là …` sẽ cho cùng một đáp án, mà rõ ràng không phải vậy.
- (3) **trượt** — §13.61 cho thấy `E` và bậc hai cùng họ và ngang nhau.
- Nếu (2) **trượt** ⇒ kết luận mạnh và **đóng** ý tưởng: khung một mình đã giải thích hết, và
  "loại suy" chỉ là bigram đội lốt.

Chạy: `PYTHONPATH=src python scripts/probe_frame_analogy.py --n-cases 400`
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

from nn_aivie.core.adjacency import EdgeKind  # noqa: E402
from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_MIN_CASES = 200
GATE_VS_FRAME = 1.30
GATE_VS_EMBED = 1.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-cases", type=int, default=400)
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_frame_analogy.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_frame_analogy")
    t0 = time.time()
    logger.log(f"P87 khung+loại suy · {args.checkpoint} · cổng: ≥{GATE_MIN_CASES} ca · "
               f"≥{GATE_VS_FRAME}× khung · ≥{GATE_VS_EMBED}× embedding")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)

    keep = np.zeros(n, dtype=bool)
    for t in range(n):
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            if float(ps._gidf(t)) >= args.gidf_min:  # noqa: SLF001
                keep[t] = True
        except Exception:
            continue

    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    rowsum = np.zeros(n)
    colmass = np.zeros(n)
    tot_seq = 0.0
    for i, j, w in seq.items():
        if 0 <= i < n:
            rowsum[i] += w
        if 0 <= j < n:
            colmass[j] += w
        tot_seq += w

    def frame_scores(f: int) -> dict[int, float]:
        """PPMI_seq(f → ·) — khung một mình biết gì."""
        row = seq.row(int(f)) or {}
        out: dict[int, float] = {}
        for j, w in row.items():
            j = int(j)
            if w <= 0 or rowsum[f] <= 0 or colmass[j] <= 0:
                continue
            p = np.log((w / rowsum[f]) / (colmass[j] / tot_seq + 1e-300))
            if p > 0:
                out[j] = float(p)
        return out

    par_toks: list[list[int]] = []
    rows: list[int] = []
    cols: list[int] = []
    for rec in ps.parents:
        toks = sorted({int(t) for t in rec.emit_ids if 0 <= int(t) < n and keep[int(t)]})
        par_toks.append(toks)
        if len(toks) > 60:
            continue
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                rows += [toks[i], toks[j]]
                cols += [toks[j], toks[i]]
    M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    M.sum_duplicates()
    tot = float(M.sum())
    rs = np.asarray(M.sum(axis=1)).ravel()
    Mp = M.tocoo()
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = np.log((Mp.data / tot) / ((rs[Mp.row] / tot) * (rs[Mp.col] / tot) + 1e-300))
    pm[~np.isfinite(pm)] = 0.0
    P = csr_matrix((np.maximum(pm, 0.0), (Mp.row, Mp.col)), shape=(n, n))
    P.eliminate_zeros()
    logger.log(f"M {M.nnz:,} ô · P {P.nnz:,} ô ({time.time()-t0:.0f}s)")

    E = np.asarray(state.embedding.matrix, dtype=np.float64)
    if E.shape[0] < n:
        E = np.pad(E, ((0, n - E.shape[0]), (0, 0)))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    bg_par: dict[tuple[int, int], set[int]] = defaultdict(set)
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(len(e) - 1):
            bg_par[(e[i], e[i + 1])].add(pi)

    cases: list[tuple[int, int, int, int]] = []
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(2, len(e)):
            f, x = e[i - 1], e[i]
            if not (0 <= x < n and keep[x]):
                continue
            if len(bg_par[(f, x)]) != 1:
                continue
            a = next((e[k] for k in range(i - 2, -1, -1) if 0 <= e[k] < n and keep[e[k]]), None)
            if a is None or a == x:
                continue
            cases.append((int(a), int(f), int(x), pi))
            break
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    cases = cases[: args.n_cases]
    logger.log(f"ca giữ lại: {len(cases):,} ({time.time()-t0:.0f}s)")

    names = ("ngau_nhien", "khung", "khung_x_embedding", "khung_x_bac_hai")
    rr: dict[str, list[float]] = {k: [] for k in names}
    hit1 = dict.fromkeys(names, 0)
    n_reach = 0
    csz: list[int] = []
    for a, f, x, pi in cases:
        fs = frame_scores(f)
        fs.pop(x, None)
        if x in fs:
            n_reach += 1
        cand = sorted(set(fs) | {x})
        csz.append(len(cand))
        row = P[a].toarray().ravel().copy()
        for t in par_toks[pi]:
            if t != a:
                row[t] = 0.0
        s2 = row @ P
        se = En @ En[a]
        base = np.array([fs.get(c, 0.0) for c in cand])
        sc = {
            "ngau_nhien": np.array([rng.random() for _ in cand]),
            "khung": base,
            "khung_x_embedding": base * np.maximum(se[cand], 0.0),
            "khung_x_bac_hai": base * np.maximum(s2[cand], 0.0),
        }
        xi = cand.index(x)
        for k in names:
            r = int((sc[k] > sc[k][xi]).sum()) + 1
            rr[k].append(1.0 / r)
            if r == 1:
                hit1[k] += 1

    nc = len(cases)
    print(f"\nứng viên/ca: trung vị {int(np.median(csz)) if csz else 0} · "
          f"đích TỰ tới được: {(n_reach/nc if nc else 0):.1%}")
    print(f"\n{'bộ chấm':<20}{'MRR':>9}{'hit@1':>9}")
    res = {}
    for k in names:
        mrr = float(np.mean(rr[k])) if nc else 0.0
        h1 = hit1[k] / nc if nc else 0.0
        res[k] = {"mrr": mrr, "hit1": h1}
        print(f"{k:<20}{mrr:>9.3f}{h1:>8.1%}")

    r_fr = res["khung_x_bac_hai"]["mrr"] / max(res["khung"]["mrr"], 1e-9)
    r_em = res["khung_x_bac_hai"]["mrr"] / max(res["khung_x_embedding"]["mrr"], 1e-9)
    g1, g2, g3 = nc >= GATE_MIN_CASES, r_fr >= GATE_VS_FRAME, r_em >= GATE_VS_EMBED
    print(f"\nbac_hai/khung = {r_fr:.2f}× · bac_hai/embedding = {r_em:.2f}×")
    print(f"CỔNG G-frame: (1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'} · "
          f"(3) {'✅' if g3 else '❌'} ⇒ {'ĐẠT' if (g1 and g2 and g3) else 'TRƯỢT'}")

    out = {"checkpoint": args.checkpoint, "n_cases": nc,
           "ung_vien_trung_vi": int(np.median(csz)) if csz else 0,
           "dich_tu_toi_duoc": (n_reach / nc if nc else 0.0),
           "ket_qua": res, "ty_so_vs_khung": r_fr, "ty_so_vs_embedding": r_em,
           "gate_pass": bool(g1 and g2 and g3), "seconds": round(time.time() - t0, 1)}
    (_ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
