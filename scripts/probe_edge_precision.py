#!/usr/bin/env python
"""P93 (TK_046 §2) — cạnh suy ra có **ĐÚNG** không: độ chính xác trên nửa corpus GIỮ LẠI.

§13.67 đo *xếp đúng đích* và thắng (`tach_lat` 1,60× `E`). Nhưng nó **không** đo *có suy bậy
không* — P89 (`bàn → động vật`) cho thấy đó mới là chỗ khó. Vòng này đo đúng chỗ đó.

## Gộp hai việc làm một

Ticket TK_046 §8 kê hai việc: (1) đo ca âm tính, (2) cạnh phải **tự chứng minh**. Chúng là
**một** phép đo: chia corpus đôi, suy trên nửa **A**, rồi hỏi cạnh suy ra có **xuất hiện thật**
ở nửa **B** không.

- cạnh đúng ⇒ xuất hiện ở B ⇒ **độ chính xác cao**;
- cạnh suy bậy ⇒ không bao giờ thấy ở B ⇒ **độ chính xác thấp**.

Và nó cho luôn thứ §13.65 thiếu: **ngưỡng tự hiệu chỉnh** thay vì tôi chọn tay — chính cái bẫy
đã giết P89 (`hub_max = 0,85` vì tôi *biết trước* `con` ở 0,86).

## Ba / bốn bộ sinh cạnh, cùng ngân sách top-k

| | |
|---|---|
| `tach_lat` | cơ chế §13.67: `Σ_b |khung(a)∩khung(b)| · đếm(b·f·c)` |
| `embedding` | mốc: `Σ_b cos(E[a],E[b]) · đếm(b·f·c)` — cùng khung, chỉ khác tương đồng |
| `hebbian_E` | P98: giống `embedding` nhưng **chỉ viết** `(f,c)` khi ≥ `--min-support` bạn `b` ủng hộ |
| `ngau_nhien` | sàn khớp phân bố: bốc `c` ngẫu nhiên trong cùng khung `f` |

## CỔNG G-prec (nêu TRƯỚC) — bản P93

1. **≥200 chủ thể** có ≥1 cạnh suy ra.
2. `tach_lat` chính xác (lỏng) **≥ 2,0×** `ngau_nhien`
3. `tach_lat` **≥ 1,0×** `embedding`

## CỔNG G-prec-HE (P98B) — khi `--writer hebbian_E`

So `hebbian_E` với `embedding` / `ngau_nhien` (cùng chạy):

1. ≥200 chủ thể có ≥1 cạnh `hebbian_E`
2. `hebbian_E` lỏng **≥ 1,5×** `ngau_nhien`
3. `hebbian_E` lỏng **≥ 1,0×** `embedding`

Chạy: `PYTHONPATH=src python scripts/probe_edge_precision.py --n-subj 300`
P98B: thêm `--writer hebbian_E --min-support 2`
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

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_MIN_SUBJ = 200
GATE_VS_RANDOM = 2.0
GATE_VS_EMBED = 1.0
GATE_HE_VS_RANDOM = 1.5
GATE_HE_VS_EMBED = 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-subj", type=int, default=300)
    ap.add_argument("--topk", type=int, default=5, help="số cạnh suy ra mỗi chủ thể")
    ap.add_argument("--top-b", type=int, default=20, help="số thực thể tương đồng xét")
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-support", type=int, default=2,
                    help="P98 hebbian_E: số bạn b tối thiểu ủng hộ một (f,c)")
    ap.add_argument("--writer", default="all",
                    choices=("all", "hebbian_E"),
                    help="all=P93 ba bộ; hebbian_E=thêm bộ P98 và cổng G-prec-HE")
    ap.add_argument("--out", default="out/probe_edge_precision.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_edge_precision")
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

    rng = random.Random(args.seed)
    idx = list(range(len(ps.parents)))
    rng.shuffle(idx)
    half = len(idx) // 2
    A_par, B_par = set(idx[:half]), set(idx[half:])

    framesA: dict[int, set[int]] = defaultdict(set)
    triA: dict[int, list[tuple[int, int]]] = defaultdict(list)   # f → [(b, c)]
    haveA: set[tuple[int, int, int]] = set()
    for pi in A_par:
        e = [int(t) for t in ps.parents[pi].emit_ids]
        for i in range(len(e) - 1):
            a_, b_ = e[i], e[i + 1]
            if 0 <= a_ < n and 0 <= b_ < n and keep[a_] and keep[b_]:
                framesA[a_].add(b_)
                framesA[b_].add(a_)
        for i in range(1, len(e) - 1):
            b_, f_, c_ = e[i - 1], e[i], e[i + 1]
            if 0 <= b_ < n and 0 <= c_ < n and keep[b_] and keep[c_]:
                triA[f_].append((b_, c_))
                haveA.add((b_, f_, c_))

    triB: set[tuple[int, int, int]] = set()
    coB: dict[int, set[int]] = defaultdict(set)
    for pi in B_par:
        e = [int(t) for t in ps.parents[pi].emit_ids]
        for i in range(2, len(e)):
            triB.add((e[i - 2], e[i - 1], e[i]))
        u = {t for t in e if 0 <= t < n and keep[t]}
        for t in u:
            coB[t] |= u
    logger.log(f"A {len(A_par):,} cha · B {len(B_par):,} cha · khung A {len(framesA):,} "
               f"({time.time()-t0:.0f}s)")

    E = np.asarray(state.embedding.matrix, dtype=np.float64)
    if E.shape[0] < n:
        E = np.pad(E, ((0, n - E.shape[0]), (0, 0)))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    subj = [t for t in framesA if len(framesA[t]) >= 3 and t in coB]
    rng.shuffle(subj)
    subj = subj[: args.n_subj]
    logger.log(f"chủ thể: {len(subj):,} ({time.time()-t0:.0f}s)")

    bs = [b for b in framesA if len(framesA[b]) >= 3]
    writers = ["tach_lat", "embedding", "ngau_nhien"]
    if args.writer == "hebbian_E":
        writers.append("hebbian_E")
    stat = {k: {"n": 0, "chat": 0, "long": 0} for k in writers}
    n_subj_ok = 0
    n_subj_he = 0

    byb: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for f_, lst in triA.items():
        for b_, c_ in lst:
            byb[b_].append((f_, c_))

    def infer_edges(a: int, pairs: list[tuple[float, int]]) -> list[tuple[int, int]]:
        sc: dict[tuple[int, int], float] = defaultdict(float)
        for s_, b in pairs:
            if s_ <= 0:
                continue
            for f_, c_ in byb.get(b, ()):
                if c_ == a or (a, f_, c_) in haveA:
                    continue
                sc[(f_, c_)] += float(s_)
        return [fc for fc, _ in sorted(sc.items(), key=lambda kv: -kv[1])[: args.topk]]

    def infer_hebbian_E(a: int, pairs: list[tuple[float, int]]) -> list[tuple[int, int]]:
        """Giống embedding nhưng chỉ giữ (f,c) có ≥ min_support bạn b ủng hộ."""
        support: dict[tuple[int, int], set[int]] = defaultdict(set)
        sc: dict[tuple[int, int], float] = defaultdict(float)
        for s_, b in pairs:
            if s_ <= 0:
                continue
            for f_, c_ in byb.get(b, ()):
                if c_ == a or (a, f_, c_) in haveA:
                    continue
                support[(f_, c_)].add(int(b))
                sc[(f_, c_)] += float(s_)
        ok = [
            (fc, sc[fc]) for fc, bs_ in support.items()
            if len(bs_) >= int(args.min_support)
        ]
        ok.sort(key=lambda x: -x[1])
        return [fc for fc, _ in ok[: args.topk]]

    for a in subj:
        fa = framesA[a]
        base = sorted(((len(fa & framesA[b]), b) for b in bs if b != a),
                      reverse=True)[: args.top_b * 5]
        tl = infer_edges(a, [(float(s_), b) for s_, b in base[: args.top_b]])
        em_pairs = sorted(((float(En[a] @ En[b]), b) for _s, b in base),
                          reverse=True)[: args.top_b]
        em = infer_edges(a, em_pairs)
        pool = sorted({fc for _s, b in base[: args.top_b] for fc in byb.get(b, ())
                       if fc[1] != a and (a, fc[0], fc[1]) not in haveA})
        rd = rng.sample(pool, min(args.topk, len(pool))) if pool else []
        he = infer_hebbian_E(a, em_pairs) if "hebbian_E" in writers else []
        if tl:
            n_subj_ok += 1
        if he:
            n_subj_he += 1
        batch = [("tach_lat", tl), ("embedding", em), ("ngau_nhien", rd)]
        if "hebbian_E" in writers:
            batch.append(("hebbian_E", he))
        for name, edges in batch:
            for f_, c_ in edges:
                stat[name]["n"] += 1
                if (a, f_, c_) in triB:
                    stat[name]["chat"] += 1
                if c_ in coB.get(a, ()):
                    stat[name]["long"] += 1

    print(f"\nchủ thể có ≥1 cạnh suy ra: {n_subj_ok}/{len(subj)}"
          + (f" · hebbian_E: {n_subj_he}" if "hebbian_E" in writers else ""))
    print(f"\n{'bộ sinh cạnh':<14}{'cạnh':>7}{'chặt':>9}{'lỏng':>9}")
    res = {}
    for k in writers:
        s = stat[k]
        nn = max(s["n"], 1)
        res[k] = {"n": s["n"], "chat": s["chat"] / nn, "long": s["long"] / nn}
        print(f"{k:<14}{s['n']:>7}{s['chat']/nn:>8.1%}{s['long']/nn:>9.1%}")

    r_rand = res["tach_lat"]["long"] / max(res["ngau_nhien"]["long"], 1e-9)
    r_em = res["tach_lat"]["long"] / max(res["embedding"]["long"], 1e-9)
    g1 = n_subj_ok >= GATE_MIN_SUBJ
    g2 = r_rand >= GATE_VS_RANDOM
    g3 = r_em >= GATE_VS_EMBED
    print(f"\ntach_lat/ngau_nhien = {r_rand:.2f}× · tach_lat/embedding = {r_em:.2f}×")
    print(f"CỔNG G-prec: (1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'} · "
          f"(3) {'✅' if g3 else '❌'} ⇒ {'ĐẠT' if (g1 and g2 and g3) else 'TRƯỢT'}")

    he_gate = None
    if "hebbian_E" in writers:
        he_r = res["hebbian_E"]["long"] / max(res["ngau_nhien"]["long"], 1e-9)
        he_e = res["hebbian_E"]["long"] / max(res["embedding"]["long"], 1e-9)
        hg1 = n_subj_he >= GATE_MIN_SUBJ
        hg2 = he_r >= GATE_HE_VS_RANDOM
        hg3 = he_e >= GATE_HE_VS_EMBED
        he_gate = bool(hg1 and hg2 and hg3)
        print(f"hebbian_E/ngau_nhien = {he_r:.2f}× · hebbian_E/embedding = {he_e:.2f}×")
        print(f"CỔNG G-prec-HE: (1) {'✅' if hg1 else '❌'} · (2) {'✅' if hg2 else '❌'} · "
              f"(3) {'✅' if hg3 else '❌'} ⇒ {'ĐẠT' if he_gate else 'TRƯỢT'}")

    payload = {
        "checkpoint": args.checkpoint, "n_subj": len(subj), "n_subj_ok": n_subj_ok,
        "n_subj_he": n_subj_he if "hebbian_E" in writers else None,
        "min_support": args.min_support if "hebbian_E" in writers else None,
        "ket_qua": res,
        "ty_so": {"vs_random": r_rand, "vs_embedding": r_em},
        "gate_pass": bool(g1 and g2 and g3),
        "gate_he_pass": he_gate,
        "seconds": round(time.time() - t0, 1),
    }
    (_ROOT / args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
