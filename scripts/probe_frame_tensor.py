#!/usr/bin/env python
"""P92 (TK_045 §13.67) — **TENXƠ TÁCH LÁT THEO KHUNG**: bản cuối của ý tưởng tác giả.

§13.66 cho tiêu chí "số khung chung" qua cổng chống-bão-hoà (7,5% ở ca vô can, so với 94,8%
của tiêu chí thực-thể đã chết). Vòng này dùng nó **làm bộ đo tương đồng**, và chuyển giao
**TRONG CÙNG MỘT KHUNG** — đúng chỗ §13.65 hỏng vì túi từ không phân biệt được vai.

## Cơ chế

    tương đồng(a, b) = |khung(a) ∩ khung(b)|          # đã qua cổng §13.66
    điểm(c) = Σ_b  tương đồng(a, b) · đếm(b · f · c)   # chuyển giao TRONG khung f

Đọc ra tiếng Việt: *`khỉ` được điểm cho `động vật` vì `chó` và `mèo` — hai thực thể có nhiều
khung chung với `khỉ` — từng xuất hiện trong đúng khung `… là động vật`.*

Khác §13.65 ở chỗ quyết định: `bàn` và `chó` chung khung `màu`, nhưng chuyển giao chỉ xảy ra
**trong khung `là`**, nơi `bàn` không có ai giống nó. Ca âm tính chết vì **cấu trúc**, không vì
ngưỡng.

## Nhiệm vụ: y hệt nhiệm vụ B của §13.63/§13.65 — để so được

Cặp `(khung, đích)` chỉ có ở **đúng một cha**; bỏ **nguyên cha** đó. Mốc đã đo trên cùng nhiệm
vụ (§13.65):

| | MRR |
|---|---:|
| `ngau_nhien` | 0,0803 |
| `bac_hai` một vòng | 0,1226 |
| `lan_truyen` 3 chặng | 0,1098 |
| **`embedding`** | **0,1780** |

## CỔNG G-tensor (nêu TRƯỚC)

1. **≥300 ca**.
2. `tach_lat` MRR **≥ 1,10×** `embedding` — `E` có sẵn miễn phí; không hơn thì ý tưởng không có
   chỗ đứng riêng và kết luận là "dùng `E`".
3. `tach_lat` MRR **≥ 1,30×** `bac_hai` — phải hơn hẳn bản túi-từ, nếu không thì việc tách lát
   không mua gì.

## Dự đoán ghi TRƯỚC (Claude)

- (3) **đạt**: tách lát là ràng buộc cấu trúc thật, túi từ không có.
- (2) **50/50 — tôi thật sự không biết**, và đây là lần đầu trong chuỗi này tôi không nghiêng
  hẳn về phía nào. Lý do nghi ngờ: `E` cũng gián tiếp mã hoá vị trí (nó là SVD của `W_seq` có
  hướng). Lý do hy vọng: `E` **trộn mọi khung vào một vector 96 chiều**, còn tách lát **giữ
  riêng** — và §13.66 vừa cho thấy vị trí là thứ tạo ra sức chọn lọc.
- Nếu (2) trượt ⇒ đóng chuỗi bốn phép đo: ý tưởng **đúng về hiện tượng**, nhưng **mọi** cách
  hiện thực bằng đếm đều thua SVD, và việc cần làm là dùng `E`.

Chạy: `PYTHONPATH=src python scripts/probe_frame_tensor.py --n-cases 400`
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

GATE_MIN = 300
GATE_VS_EMBED = 1.10
GATE_VS_2ND = 1.30


def mid_rank(sc: np.ndarray, i: int) -> float:
    s = sc[i]
    return 1.0 + float((sc > s).sum()) + float((sc == s).sum() - 1) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-cases", type=int, default=400)
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--max-b", type=int, default=200, help="số thực thể cùng khung xét mỗi ca")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_frame_tensor.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_frame_tensor")
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

    # --- KHUNG có xuất xứ CHA (để bỏ nguyên cha lúc đo) -----------------------------------
    # TK_046 #5 — KHUNG CÓ HƯỚNG. Bản trộn tính một token đứng TRƯỚC `a` và SAU `b` là khớp,
    # tức trùng giả. Ví dụ thứ hai của tác giả (`ok thực → hiện đi`) cần đúng ngữ cảnh TRÁI.
    framesL: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    framesR: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    frames: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    tri: dict[int, list[tuple[int, int, int]]] = defaultdict(list)   # f → [(b, c, cha)]
    par_toks: list[list[int]] = []
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        par_toks.append([t for t in e if 0 <= t < n and keep[t]])
        for i in range(len(e) - 1):
            a_, b_ = e[i], e[i + 1]
            if 0 <= a_ < n and 0 <= b_ < n and keep[a_] and keep[b_]:
                frames[a_][b_].add(pi)
                frames[b_][a_].add(pi)
                framesR[a_][b_].add(pi)     # b_ đứng SAU a_
                framesL[b_][a_].add(pi)     # a_ đứng TRƯỚC b_
        for i in range(1, len(e) - 1):
            b_, f_, c_ = e[i - 1], e[i], e[i + 1]
            if 0 <= c_ < n and keep[c_] and 0 <= b_ < n:
                tri[f_].append((b_, c_, pi))
    logger.log(f"khung {len(frames):,} · khung có tam-tố {len(tri):,} ({time.time()-t0:.0f}s)")

    # --- ma trận túi-từ để tính `bac_hai` (mốc) -------------------------------------------
    rows: list[int] = []
    cols: list[int] = []
    for toks in par_toks:
        u = sorted(set(toks))
        if len(u) > 60:
            continue
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                rows += [u[i], u[j]]
                cols += [u[j], u[i]]
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

    bg: dict[tuple[int, int], set[int]] = defaultdict(set)
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(len(e) - 1):
            bg[(e[i], e[i + 1])].add(pi)
    # §13.67 — ĐƠN VỊ GIỮ LẠI LÀ BỘ BA, không phải cặp.
    #
    # Nhiệm vụ B của §13.63 chọn ca sao cho bigram `(f, x)` chỉ có ở đúng một cha rồi bỏ cha
    # đó ⇒ `(f, x)` **không còn tồn tại ở đâu**. Nhưng cơ chế tách lát chuyển giao mẫu
    # `b · f · c` từ cha KHÁC, nên nó cần `(f, x)` có ở đâu đó — theo cấu tạo thì KHÔNG BAO
    # GIỜ có. Đo ra `0/400 đích được cho điểm`: nhiệm vụ và cơ chế **không tương thích**.
    #
    # Đối chiếu ví dụ gốc: `con chó là động vật` và `con mèo là động vật` ĐỀU có trong train,
    # nên `(là, động vật)` tồn tại ở 2 cha; thứ thiếu là **bộ ba** `(khỉ, là, động vật)`.
    # ⇒ giữ lại BỘ BA, và đòi `(f, x)` còn ở **≥1 cha khác** — đúng kịch bản tác giả mô tả.
    tri_par: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(2, len(e)):
            tri_par[(e[i - 2], e[i - 1], e[i])].add(pi)
    cases: list[tuple[int, int, int, int]] = []
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(2, len(e)):
            f, x = e[i - 1], e[i]
            if not (0 <= x < n and keep[x]):
                continue
            a = e[i - 2]
            if not (0 <= a < n and keep[a]) or a == x:
                continue
            if len(tri_par[(a, f, x)]) != 1:      # bộ ba phải DUY NHẤT ở cha này
                continue
            if len(bg[(f, x)] - {pi}) < 1:        # nhưng khung+đích PHẢI còn ở cha khác
                continue
            cases.append((int(a), int(f), int(x), pi))
            break
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    cases = cases[: args.n_cases]
    logger.log(f"ca: {len(cases):,} ({time.time()-t0:.0f}s)")

    def _excl(d: dict[int, set[int]] | None, pi: int) -> set[int]:
        return {k for k, pset in (d or {}).items() if pset - {pi}} if d else set()

    def frames_excl(t: int, pi: int) -> set[int]:
        """Khung TRỘN của `t`, BỎ khung chỉ do cha `pi` tạo ra."""
        return _excl(frames.get(t), pi)

    def frames_dir(t: int, pi: int) -> tuple[set[int], set[int]]:
        """Khung CÓ HƯỚNG: (trái, phải)."""
        return _excl(framesL.get(t), pi), _excl(framesR.get(t), pi)

    # §13.67 — ĐỐI CHỨNG QUYẾT ĐỊNH, suýt bỏ sót: nhiệm vụ mới đòi `(f, x)` CÒN ở cha khác,
    # nên `khung` một mình đã có thể cho `x` điểm dương. Không có nó thì không thể nói loại
    # suy làm được gì — đúng bài học §13.62.
    names = ("ngau_nhien", "khung", "embedding", "bac_hai", "tach_lat", "tach_lat_huong")
    rr: dict[str, list[float]] = {k: [] for k in names}
    n_hit_b = 0
    for a, f, x, pi in cases:
        fs = frame_scores(f)
        cand = sorted(set(fs) | {x})
        ci = {c: k for k, c in enumerate(cand)}

        fa = frames_excl(a, pi)
        faL, faR = frames_dir(a, pi)
        s_tl = np.zeros(len(cand))
        s_tld = np.zeros(len(cand))
        seen_b: dict[int, float] = {}
        seen_bd: dict[int, float] = {}
        rows_f = tri.get(f, ())
        for b_, c_, pj in rows_f[: args.max_b * 40]:
            if pj == pi or c_ not in ci or b_ == a:
                continue
            sb = seen_b.get(b_)
            if sb is None:
                sb = float(len(fa & frames_excl(b_, pi)))
                seen_b[b_] = sb
            if sb > 0:
                s_tl[ci[c_]] += sb
            sbd = seen_bd.get(b_)
            if sbd is None:
                bL, bR = frames_dir(b_, pi)
                sbd = float(len(faL & bL) + len(faR & bR))
                seen_bd[b_] = sbd
            if sbd > 0:
                s_tld[ci[c_]] += sbd
        if s_tl[ci[x]] > 0:
            n_hit_b += 1

        v1 = P[a].toarray().ravel().copy()
        for t in par_toks[pi]:
            if t != a:
                v1[t] = 0.0
        s2 = v1 @ P
        se = En @ En[a]
        base_f = np.array([fs.get(c, 0.0) for c in cand])
        sc = {
            "ngau_nhien": np.array([rng.random() for _ in cand]),
            "khung": base_f,
            "embedding": np.maximum(se[cand], 0.0),
            "bac_hai": np.maximum(s2[cand], 0.0),
            "tach_lat": s_tl,
            "tach_lat_huong": s_tld,
        }
        xi = ci[x]
        for k in names:
            rr[k].append(1.0 / mid_rank(sc[k], xi))

    nc = len(cases)
    res = {k: float(np.mean(rr[k])) if nc else 0.0 for k in names}
    print(f"\nca = {nc} · đích được tách-lát cho điểm > 0: "
          f"{n_hit_b}/{nc} = {(n_hit_b/nc if nc else 0):.1%}")
    print(f"\n{'bộ chấm':<14}{'MRR':>9}")
    for k in names:
        print(f"{k:<14}{res[k]:>9.4f}")
    r_em = res["tach_lat"] / max(res["embedding"], 1e-12)
    r_2 = res["tach_lat"] / max(res["bac_hai"], 1e-12)
    r_hu = res["tach_lat_huong"] / max(res["tach_lat"], 1e-12)
    print(f"tach_lat_huong/tach_lat = {r_hu:.2f}×   ← cải thiện #5 (tách hướng)")
    r_kh = res["tach_lat"] / max(res["khung"], 1e-12)
    print(f"tach_lat/khung = {r_kh:.2f}×   ← đối chứng quyết định")
    if res["khung"] <= res["ngau_nhien"]:
        print("⛔ KIỂM SÀN TRƯỢT: khung không hơn ngẫu nhiên ⇒ thiết kế hỏng")
    g1, g2, g3 = nc >= GATE_MIN, r_em >= GATE_VS_EMBED, r_2 >= GATE_VS_2ND
    print(f"\ntach_lat/embedding = {r_em:.2f}× · tach_lat/bac_hai = {r_2:.2f}×")
    print(f"CỔNG G-tensor: (1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'} · "
          f"(3) {'✅' if g3 else '❌'} ⇒ {'ĐẠT' if (g1 and g2 and g3) else 'TRƯỢT'}")

    (_ROOT / args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "n_cases": nc, "ket_qua": res,
         "dich_co_diem": (n_hit_b / nc if nc else 0.0),
         "ty_so": {"vs_embedding": r_em, "vs_bac_hai": r_2},
         "gate_pass": bool(g1 and g2 and g3), "seconds": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
