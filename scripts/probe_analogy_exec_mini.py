#!/usr/bin/env python
"""P98C (TK_046) — mini: cạnh hebbian_E có giúp ĐIỀN KHUNG `a · f · ?` không?

Không dựng lại tầng K (chuỗi nhãn — đã chết TK_047 §7).
Cùng giao thức nửa A/B như P93: viết cạnh trên A, thi hành điền c trên prompt (a,f),
hỏi (a,f,c) có ở B không.

Cổng G-exec (nêu TRƯỚC):
1. ≥100 lần điền hebbian_E
2. precision (a,f,c)∈B của hebbian_E ≥ 1,0× embedding
3. ≥ 1,5× ngẫu nhiên (cùng f, bốc c từ pool khung)
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

GATE_N = 100
GATE_VS_EM = 1.0
GATE_VS_RD = 1.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-prompt", type=int, default=300)
    ap.add_argument("--top-b", type=int, default=20)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_analogy_exec_mini.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_analogy_exec_mini")
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
    byb: dict[int, list[tuple[int, int]]] = defaultdict(list)
    haveA: set[tuple[int, int, int]] = set()
    # prompts từ B: (a, f) xuất hiện ở B nhưng ta sẽ điền c bằng cạnh suy từ A
    prompts_B: list[tuple[int, int, int]] = []  # (a, f, c_gold)

    for pi in A_par:
        e = [int(t) for t in ps.parents[pi].emit_ids]
        for i in range(len(e) - 1):
            a_, b_ = e[i], e[i + 1]
            if 0 <= a_ < n and 0 <= b_ < n and keep[a_] and keep[b_]:
                framesA[a_].add(b_)
                framesA[b_].add(a_)
        for i in range(1, len(e) - 1):
            b_, f_, c_ = e[i - 1], e[i], e[i + 1]
            if keep[b_] and keep[c_] and 0 <= f_ < n:
                byb[b_].append((f_, c_))
                haveA.add((b_, f_, c_))

    triB: set[tuple[int, int, int]] = set()
    for pi in B_par:
        e = [int(t) for t in ps.parents[pi].emit_ids]
        for i in range(1, len(e) - 1):
            a_, f_, c_ = e[i - 1], e[i], e[i + 1]
            if keep[a_] and keep[c_]:
                triB.add((a_, f_, c_))
                prompts_B.append((a_, f_, c_))

    E = np.asarray(state.embedding.matrix, dtype=np.float64)
    if E.shape[0] < n:
        E = np.pad(E, ((0, n - E.shape[0]), (0, 0)))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    bs = [b for b in framesA if len(framesA[b]) >= 3]
    rng.shuffle(prompts_B)
    # chỉ prompt có a trong framesA
    prompts = [(a, f, g) for a, f, g in prompts_B if a in framesA and len(framesA[a]) >= 3]
    prompts = prompts[: args.n_prompt]
    logger.log(f"prompt điền: {len(prompts):,} · khung A {len(framesA):,} ({time.time()-t0:.0f}s)")

    def neighbors_E(a: int) -> list[tuple[float, int]]:
        fa = framesA[a]
        base = sorted(((len(fa & framesA[b]), b) for b in bs if b != a),
                      reverse=True)[: args.top_b * 5]
        return sorted(((float(En[a] @ En[b]), b) for _s, b in base),
                      reverse=True)[: args.top_b]

    def fill_embedding(a: int, f: int, pairs: list[tuple[float, int]]) -> int | None:
        sc: dict[int, float] = defaultdict(float)
        for s_, b in pairs:
            if s_ <= 0:
                continue
            for f_, c_ in byb.get(b, ()):
                if f_ != f or c_ == a or (a, f_, c_) in haveA:
                    continue
                sc[c_] += float(s_)
        if not sc:
            return None
        return max(sc.items(), key=lambda kv: kv[1])[0]

    def fill_hebbian(a: int, f: int, pairs: list[tuple[float, int]]) -> int | None:
        support: dict[int, set[int]] = defaultdict(set)
        sc: dict[int, float] = defaultdict(float)
        for s_, b in pairs:
            if s_ <= 0:
                continue
            for f_, c_ in byb.get(b, ()):
                if f_ != f or c_ == a or (a, f_, c_) in haveA:
                    continue
                support[c_].add(int(b))
                sc[c_] += float(s_)
        ok = {c: sc[c] for c, bs_ in support.items() if len(bs_) >= args.min_support}
        if not ok:
            return None
        return max(ok.items(), key=lambda kv: kv[1])[0]

    def fill_random(a: int, f: int, pairs: list[tuple[float, int]]) -> int | None:
        pool = sorted({
            c_ for _s, b in pairs for f_, c_ in byb.get(b, ())
            if f_ == f and c_ != a and (a, f_, c_) not in haveA
        })
        return rng.choice(pool) if pool else None

    hit = {"hebbian_E": 0, "embedding": 0, "ngau_nhien": 0}
    n_fill = {"hebbian_E": 0, "embedding": 0, "ngau_nhien": 0}

    for a, f, _gold in prompts:
        pairs = neighbors_E(a)
        for name, fn in (
            ("hebbian_E", fill_hebbian),
            ("embedding", fill_embedding),
            ("ngau_nhien", fill_random),
        ):
            c = fn(a, f, pairs)
            if c is None:
                continue
            n_fill[name] += 1
            if (a, f, c) in triB:
                hit[name] += 1

    print(f"\n{'bộ điền':<14}{'n_điền':>8}{'hit@B':>8}{'prec':>8}")
    prec = {}
    for k in ("hebbian_E", "embedding", "ngau_nhien"):
        p = hit[k] / max(n_fill[k], 1)
        prec[k] = p
        print(f"{k:<14}{n_fill[k]:>8}{hit[k]:>8}{p:>7.1%}")

    r_em = prec["hebbian_E"] / max(prec["embedding"], 1e-9)
    r_rd = prec["hebbian_E"] / max(prec["ngau_nhien"], 1e-9)
    g1 = n_fill["hebbian_E"] >= GATE_N
    g2 = r_em >= GATE_VS_EM
    g3 = r_rd >= GATE_VS_RD
    ok = g1 and g2 and g3
    print(f"\nhebbian_E/embedding = {r_em:.2f}× · hebbian_E/ngau_nhien = {r_rd:.2f}×")
    print(f"CỔNG G-exec: (1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'} · "
          f"(3) {'✅' if g3 else '❌'} ⇒ {'ĐẠT' if ok else 'TRƯỢT'}")

    (_ROOT / args.out).write_text(json.dumps({
        "n_prompt": len(prompts), "n_fill": n_fill, "hit": hit, "prec": prec,
        "ty_so": {"vs_embedding": r_em, "vs_random": r_rd},
        "gate_pass": ok, "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
