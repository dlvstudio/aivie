#!/usr/bin/env python
"""P88 (TK_045 §13.63) — bậc hai + khung, **bản đã sửa ba lỗi của P87**.

§13.62: P87 báo ĐẠT nhưng **vô hiệu** — hạng tính bằng *"đếm cái lớn hơn hẳn"*, mà điểm của
đích luôn bằng 0 sau khi xoá chứng cứ, nên **hoà điểm thành phần thưởng** và bộ chấm **thưa
nhất** trông tốt nhất. Đo độ thưa, không đo tín hiệu.

Ba sửa, đúng như §13.62 đã kê:

1. **PHÁ HOÀ** — hạng giữa (`mid-rank`): `1 + #lớn_hơn + #hoà/2`. Bộ chấm thưa hết được thưởng.
2. **TÁCH LÀM HAI NHIỆM VỤ**, vì P87 trộn hai câu hỏi khác nhau vào một phép đo:
   - **A — BỔ TRỢ**: khung *có* chứng cứ về đích. Hỏi: **biết chủ thể có cải thiện không?**
     Đây là ví dụ thứ hai của tác giả (`ok thực → hiện đi`).
   - **B — SUY LUẬN THẬT**: cặp `(khung, đích)` chỉ có ở **đúng một cha**, bỏ cha đó ⇒ khung
     **không thể** biết đích. Đây là ví dụ thứ nhất (`con khỉ là → động vật`, chưa hề train).
     Ở B, **KHÔNG** nhân với khung — chính phép nhân đó đẻ ra hiện vật P87. Khung ở đây chỉ
     dùng để **giới hạn tập ứng viên**, và đối chứng là `ngẫu nhiên` + `embedding`.
3. **KIỂM SÀN TRƯỚC KHI ĐỌC**: ở nhiệm vụ A, `khung` **phải** hơn `ngẫu nhiên`. Không thoả ⇒
   **dừng, không in kết quả**. Sàn ngẫu nhiên là **máy dò lỗi thiết kế**, không chỉ là mốc.

## CỔNG (nêu TRƯỚC)

| | |
|---|---|
| A-1 | ≥200 ca · `khung` > `ngẫu nhiên` (kiểm sàn, bắt buộc) |
| A-2 | `khung_x_bac_hai` MRR **≥ 1,15×** `khung` — chủ thể có bổ trợ |
| A-3 | `khung_x_bac_hai` MRR **≥ 1,10×** `khung_x_embedding` — hơn `E` đã có |
| B-1 | ≥200 ca |
| B-2 | `bac_hai` MRR **≥ 2,0×** `ngẫu nhiên` — suy luận thật sự có |
| B-3 | `bac_hai` MRR **≥ 1,10×** `embedding` — hơn `E` đã có |

## Dự đoán ghi TRƯỚC (Claude)

- **A-2 đạt** (~1,2–1,6×): biết chủ thể phải giúp, nếu không thì mọi `X là …` cho cùng đáp án.
- **A-3 trượt**: §13.61 cho thấy `E` và bậc hai cùng họ.
- **B-2 đạt nhưng yếu** (~2–4×): suy luận thật khó hơn nhiều; corpus mini thưa.
- **B-3 50/50.**
- Nếu **B-2 trượt** ⇒ đóng ý tưởng: không có suy luận nào vượt sàn khi chứng cứ trực tiếp bị
  xoá, và mọi vẻ "loại suy" trước đó là bigram đội lốt.

Chạy: `PYTHONPATH=src python scripts/probe_frame_analogy2.py --n-cases 400`
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
GATE_A_VS_FRAME = 1.15
GATE_A_VS_EMBED = 1.10
GATE_B_VS_RANDOM = 2.0
GATE_B_VS_EMBED = 1.10


def mid_rank(scores: np.ndarray, i: int) -> float:
    """Hạng GIỮA — phá hoà. `1 + #lớn_hơn + #hoà/2`.

    P87 dùng `1 + #lớn_hơn`, tức mọi ứng viên hoà điểm với đích đều bị coi là **thua** nó ⇒
    bộ chấm nào đẩy nhiều ứng viên về cùng một giá trị (thường là 0) thì được thưởng máy móc.
    """
    s = scores[i]
    return 1.0 + float((scores > s).sum()) + float((scores == s).sum() - 1) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-cases", type=int, default=400)
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_frame_analogy2.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_frame_analogy2")
    t0 = time.time()
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

    E = np.asarray(state.embedding.matrix, dtype=np.float64)
    if E.shape[0] < n:
        E = np.pad(E, ((0, n - E.shape[0]), (0, 0)))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    logger.log(f"M {M.nnz:,} · P {P.nnz:,} ({time.time()-t0:.0f}s)")

    bg_par: dict[tuple[int, int], set[int]] = defaultdict(set)
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(len(e) - 1):
            bg_par[(e[i], e[i + 1])].add(pi)

    caseA: list[tuple[int, int, int, int]] = []
    caseB: list[tuple[int, int, int, int]] = []
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(2, len(e)):
            f, x = e[i - 1], e[i]
            if not (0 <= x < n and keep[x]):
                continue
            a = next((e[k] for k in range(i - 2, -1, -1)
                      if 0 <= e[k] < n and keep[e[k]]), None)
            if a is None or a == x:
                continue
            npar = len(bg_par[(f, x)])
            if npar >= 2 and len(caseA) < 10 ** 6:
                caseA.append((int(a), int(f), int(x), pi))
            elif npar == 1:
                caseB.append((int(a), int(f), int(x), pi))
            break
    rng = random.Random(args.seed)
    rng.shuffle(caseA)
    rng.shuffle(caseB)
    caseA, caseB = caseA[: args.n_cases], caseB[: args.n_cases]
    logger.log(f"ca A(bổ trợ) {len(caseA):,} · B(suy luận) {len(caseB):,}")

    def subj_vec(a: int, pi: int) -> np.ndarray:
        row = P[a].toarray().ravel().copy()
        for t in par_toks[pi]:
            if t != a:
                row[t] = 0.0
        return row @ P

    def run(cases, task: str) -> dict:
        names = (("ngau_nhien", "khung", "khung_x_embedding", "khung_x_bac_hai")
                 if task == "A" else ("ngau_nhien", "embedding", "bac_hai"))
        rr = {k: [] for k in names}
        for a, f, x, pi in cases:
            fs = frame_scores(f)
            cand = sorted(set(fs) | {x})
            base = np.array([fs.get(c, 0.0) for c in cand])
            s2 = subj_vec(a, pi)
            se = En @ En[a]
            sc = {"ngau_nhien": np.array([rng.random() for _ in cand])}
            if task == "A":
                sc["khung"] = base
                sc["khung_x_embedding"] = base * np.maximum(se[cand], 0.0)
                sc["khung_x_bac_hai"] = base * np.maximum(s2[cand], 0.0)
            else:
                sc["embedding"] = np.maximum(se[cand], 0.0)
                sc["bac_hai"] = np.maximum(s2[cand], 0.0)
            xi = cand.index(x)
            for k in names:
                rr[k].append(1.0 / mid_rank(sc[k], xi))
        return {k: float(np.mean(rr[k])) if rr[k] else 0.0 for k in names}

    A = run(caseA, "A")
    print(f"\n=== A — BỔ TRỢ (khung CÓ chứng cứ) · n={len(caseA)} ===")
    for k, val in A.items():
        print(f"  {k:<20} MRR {val:.4f}")

    if A["khung"] <= A["ngau_nhien"]:
        print("\n⛔ KIỂM SÀN TRƯỢT: `khung` không hơn `ngẫu nhiên` ⇒ thiết kế hỏng, "
              "KHÔNG đọc kết quả (đúng lỗi P87 §13.62).")
        return

    B = run(caseB, "B")
    print(f"\n=== B — SUY LUẬN THẬT (khung KHÔNG thể biết đích) · n={len(caseB)} ===")
    for k, val in B.items():
        print(f"  {k:<20} MRR {val:.4f}")

    a2 = A["khung_x_bac_hai"] / max(A["khung"], 1e-12)
    a3 = A["khung_x_bac_hai"] / max(A["khung_x_embedding"], 1e-12)
    b2 = B["bac_hai"] / max(B["ngau_nhien"], 1e-12)
    b3 = B["bac_hai"] / max(B["embedding"], 1e-12)
    ok = {
        "A-1": len(caseA) >= GATE_MIN and A["khung"] > A["ngau_nhien"],
        "A-2": a2 >= GATE_A_VS_FRAME, "A-3": a3 >= GATE_A_VS_EMBED,
        "B-1": len(caseB) >= GATE_MIN,
        "B-2": b2 >= GATE_B_VS_RANDOM, "B-3": b3 >= GATE_B_VS_EMBED,
    }
    print(f"\nA: bac_hai/khung {a2:.2f}× · bac_hai/embedding {a3:.2f}×")
    print(f"B: bac_hai/ngẫu nhiên {b2:.2f}× · bac_hai/embedding {b3:.2f}×")
    print("CỔNG: " + " · ".join(f"{k} {'✅' if val else '❌'}" for k, val in ok.items()))

    (_ROOT / args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "A": A, "B": B, "n_A": len(caseA), "n_B": len(caseB),
         "ty_so": {"A2": a2, "A3": a3, "B2": b2, "B3": b3}, "cong": ok,
         "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
