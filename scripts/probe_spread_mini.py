#!/usr/bin/env python
"""P90 (TK_045 §13.66) — lan truyền + **HỖ TRỢ ĐA THỰC THỂ**, đo trên mini.

Corpus đồ chơi (§13.65) đã cho ba ràng buộc, và cả ba **không phụ thuộc tham số**:

1. **chặn hub theo TẦN SUẤT tách NGƯỢC** — cầu tốt (`nghịch ngợm`) là thuộc tính chung của
   nhiều thành viên nên tần suất **cao theo định nghĩa**; cầu xấu (`nâu`) thì thấp;
2. điều kiện đúng là **số THỰC THỂ tiền bối trực tiếp** (`động vật` phải nhận từ **cả `chó`
   lẫn `mèo`**), không phải tổng kích thích, không phải lift so nền — đã thử cả ba, tách sai;
3. **loại từ / hư từ phải bị loại khỏi việc ĐẾM thực thể ủng hộ** (`con` làm hỏng ca cấm),
   nhưng **không** bị loại khỏi việc lan truyền.

Ràng buộc (3) là chỗ hệ thật có công cụ mà corpus đồ chơi không có: `gidf` + dải rác đã đo
(§13.20: 4,0–4,2). Nên đo được **mà không chọn ngưỡng bằng tay**.

## Nhiệm vụ: đúng nhiệm vụ B của §13.63 — không đổi, để so được

Cặp `(khung, đích)` chỉ có ở **đúng một cha**; bỏ **nguyên cha** đó ⇒ chứng cứ trực tiếp bị xoá
sạch, đúng nghĩa *"chưa hề train con khỉ là động vật"*. Mốc đã có:

| | MRR |
|---|---:|
| `ngau_nhien` | 0,1209 |
| `bac_hai` (một vòng) | 0,1506 |
| **`embedding`** | **0,2954** |

## Cơ chế đo: `lan_truyen`

    v1 = P[a]                      # chặng 1: thuộc tính/ngữ cảnh của chủ thể
    v2 = (v1 @ P) ⊙ mask_thuc_the  # chặng 2: THỰC THỂ tương đồng (lọc gidf + cấu trúc)
    điểm  = v2 @ P                 # chặng 3: tới đích
    hỗ trợ = (v2 > 0) @ (P > 0)    # ĐẾM thực thể phân biệt dẫn tới đích
    kết quả = điểm ⊙ (hỗ trợ ≥ 2)  # một thực thể = trùng hợp; hai trở lên = lớp

## CỔNG G-spread (nêu TRƯỚC)

1. **≥200 ca**.
2. `lan_truyen` MRR **≥ 1,10×** `embedding` — `E` có sẵn miễn phí, không hơn thì đừng xây.
3. `lan_truyen` MRR **≥ 1,30×** `bac_hai` — nếu không hơn bậc-hai-một-vòng thì hỗ-trợ-đa-thực-thể
   và chặng thứ ba **không mua gì**, và cơ chế rút về đúng thứ §13.61 đã bác.

## Dự đoán ghi TRƯỚC (Claude)

- (3) **đạt**: lọc hỗ trợ ≥2 là ràng buộc thật, một vòng thô không có.
- (2) **trượt**: §13.61 và §13.63-B đều cho `E` hơn hẳn tích thô; ba chặng có thưa hơn nhưng
  vẫn là đường đi, còn `E` dùng **hướng** — và đích ở nhiệm vụ B thường là token **hiếm**,
  nơi đường thì thưa mà hướng vẫn còn.
- Nếu (2) **đạt** ⇒ đây là cơ chế đầu tiên trong §13 vượt được `E` ở nhiệm vụ suy luận, và
  đáng đưa vào đường sinh.

Chạy: `PYTHONPATH=src python scripts/probe_spread_mini.py --n-cases 400`
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

GATE_MIN = 200
GATE_VS_EMBED = 1.10
GATE_VS_2ND = 1.30
MIN_SUPPORT = 2


def mid_rank(sc: np.ndarray, i: int) -> float:
    s = sc[i]
    return 1.0 + float((sc > s).sum()) + float((sc == s).sum() - 1) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-cases", type=int, default=400)
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--entity-gidf", type=float, default=6.0,
                    help="ngưỡng coi là THỰC THỂ khi ĐẾM hỗ trợ (§13.20: rác ở 4,0–4,2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_spread_mini.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_spread_mini")
    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)

    gidf = np.zeros(n)
    keep = np.zeros(n, dtype=bool)
    ent = np.zeros(n, dtype=bool)
    for t in range(n):
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            g = float(ps._gidf(t))  # noqa: SLF001
        except Exception:
            continue
        gidf[t] = g
        keep[t] = g >= args.gidf_min
        ent[t] = g >= args.entity_gidf
    logger.log(f"giữ {int(keep.sum()):,} · THỰC THỂ {int(ent.sum()):,} ({time.time()-t0:.0f}s)")

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
    Pb = csr_matrix((np.ones(P.nnz), P.indices, P.indptr), shape=P.shape)
    logger.log(f"P {P.nnz:,} ô ({time.time()-t0:.0f}s)")

    E = np.asarray(state.embedding.matrix, dtype=np.float64)
    if E.shape[0] < n:
        E = np.pad(E, ((0, n - E.shape[0]), (0, 0)))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    bg: dict[tuple[int, int], set[int]] = defaultdict(set)
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(len(e) - 1):
            bg[(e[i], e[i + 1])].add(pi)
    cases: list[tuple[int, int, int, int]] = []
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(2, len(e)):
            f, x = e[i - 1], e[i]
            if not (0 <= x < n and keep[x]) or len(bg[(f, x)]) != 1:
                continue
            a = next((e[k] for k in range(i - 2, -1, -1)
                      if 0 <= e[k] < n and keep[e[k]]), None)
            if a is None or a == x:
                continue
            cases.append((int(a), int(f), int(x), pi))
            break
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    cases = cases[: args.n_cases]
    logger.log(f"ca: {len(cases):,} ({time.time()-t0:.0f}s)")

    names = ("ngau_nhien", "embedding", "bac_hai", "lan_truyen")
    rr: dict[str, list[float]] = {k: [] for k in names}
    n_supp_ok = 0
    for a, f, x, pi in cases:
        fs = frame_scores(f)
        cand = sorted(set(fs) | {x})
        # bỏ NGUYÊN CHA khỏi hàng chủ thể
        v1 = P[a].toarray().ravel().copy()
        for t in par_toks[pi]:
            if t != a:
                v1[t] = 0.0
        s2 = v1 @ P                                   # bậc hai một vòng (mốc §13.61)
        v2 = s2 * ent                                 # chặng 2: chỉ giữ THỰC THỂ
        s3 = v2 @ P                                   # chặng 3: tới đích
        supp = (v2 > 0).astype(np.float64) @ Pb       # ĐẾM thực thể phân biệt
        s_lan = np.where(supp >= MIN_SUPPORT, s3, 0.0)
        if s_lan[x] > 0:
            n_supp_ok += 1
        se = En @ En[a]
        sc = {
            "ngau_nhien": np.array([rng.random() for _ in cand]),
            "embedding": np.maximum(se[cand], 0.0),
            "bac_hai": np.maximum(s2[cand], 0.0),
            "lan_truyen": np.maximum(s_lan[cand], 0.0),
        }
        xi = cand.index(x)
        for k in names:
            rr[k].append(1.0 / mid_rank(sc[k], xi))

    nc = len(cases)
    res = {k: float(np.mean(rr[k])) if nc else 0.0 for k in names}
    print(f"\nca = {nc} · đích có ≥{MIN_SUPPORT} thực thể ủng hộ: "
          f"{n_supp_ok}/{nc} = {(n_supp_ok/nc if nc else 0):.1%}")
    print(f"\n{'bộ chấm':<14}{'MRR':>9}")
    for k in names:
        print(f"{k:<14}{res[k]:>9.4f}")
    r_em = res["lan_truyen"] / max(res["embedding"], 1e-12)
    r_2 = res["lan_truyen"] / max(res["bac_hai"], 1e-12)
    g1, g2, g3 = nc >= GATE_MIN, r_em >= GATE_VS_EMBED, r_2 >= GATE_VS_2ND
    print(f"\nlan_truyen/embedding = {r_em:.2f}× · lan_truyen/bac_hai = {r_2:.2f}×")
    print(f"CỔNG G-spread: (1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'} · "
          f"(3) {'✅' if g3 else '❌'} ⇒ {'ĐẠT' if (g1 and g2 and g3) else 'TRƯỢT'}")

    (_ROOT / args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "n_cases": nc, "ket_qua": res,
         "ho_tro_du": (n_supp_ok / nc if nc else 0.0),
         "ty_so": {"vs_embedding": r_em, "vs_bac_hai": r_2},
         "gate_pass": bool(g1 and g2 and g3), "seconds": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
