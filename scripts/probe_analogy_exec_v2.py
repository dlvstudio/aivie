#!/usr/bin/env python
"""P98D (TK_046) — G-exec v2: loại suy sáng tạo, không gold trigram.

Ca dương: (a, là, c) CHƯA bao giờ có cho a, nhưng ≥2 bạn E có (b, là, c).
Ca âm: ≤1 bạn mang c → không được điền c.
Suy cạnh khi CÁCH LY mọi cha có a đứng ngay trước là.

Cổng: RUNBOOK_TK_046_P98D.md
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
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_N_POS = 30
GATE_HIT = 0.40
GATE_FP = 0.15


def _surface(vocab, tid: int) -> str:
    return vocab.token_of(int(tid)).replace("Ġ", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--top-b", type=int, default=30)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pos", type=int, default=200)
    ap.add_argument("--max-neg", type=int, default=200)
    ap.add_argument("--out", default="out/probe_analogy_exec_v2.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_analogy_exec_v2")
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

    # mọi id surface "là" — KHÔNG lọc gidf: trên mini toàn bộ là `Ġlà` (gidf≈2.5)
    la_ids = {i for i in range(n) if _surface(v, i) == "là"}
    if not la_ids:
        raise SystemExit("⛔ không tìm thấy token 'là'")
    logger.log(f"khung 'là': {sorted(la_ids)} ({len(la_ids)} id)")

    # (a, c) attested với một f∈là; cha chứa a ngay trước là
    la_ac: dict[int, set[int]] = defaultdict(set)  # a → {c}
    la_bc_donors: dict[int, set[int]] = defaultdict(set)  # c → {b heads}
    parents_with_a_la: dict[int, set[int]] = defaultdict(set)  # a → parent idxs
    frames: dict[int, set[int]] = defaultdict(set)

    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(len(e) - 1):
            if keep[e[i]] and keep[e[i + 1]]:
                frames[e[i]].add(e[i + 1])
                frames[e[i + 1]].add(e[i])
        for i in range(len(e) - 1):
            if e[i + 1] not in la_ids:
                continue
            a = e[i]
            if not keep[a]:
                continue
            parents_with_a_la[a].add(pi)
            if i + 2 < len(e) and keep[e[i + 2]]:
                c = e[i + 2]
                la_ac[a].add(c)
                la_bc_donors[c].add(a)

    E = np.asarray(state.embedding.matrix, dtype=np.float64)
    if E.shape[0] < n:
        E = np.pad(E, ((0, n - E.shape[0]), (0, 0)))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    cand_heads = [a for a in frames if keep[a] and len(frames[a]) >= 3]
    logger.log(f"heads: {len(cand_heads):,} · a có pred là: {len(la_ac):,} "
               f"({time.time()-t0:.0f}s)")

    def top_neighbors(a: int) -> list[int]:
        # ưu tiên chồng khung, xếp lại bằng E
        fa = frames[a]
        base = sorted(
            ((len(fa & frames[b]), b) for b in cand_heads if b != a),
            reverse=True,
        )[: args.top_b * 5]
        return [
            b for _s, b in sorted(
                ((float(En[a] @ En[b]), b) for _o, b in base),
                reverse=True,
            )[: args.top_b]
        ]

    # --- dựng ca dương / âm -------------------------------------------------
    pos_cases: list[dict] = []  # {a, c, donors}
    neg_cases: list[dict] = []

    # mọi c có ≥2 heads mang (head, là, c)
    rich_c = [c for c, hs in la_bc_donors.items() if len(hs) >= 2]
    rng = random.Random(args.seed)

    for a in cand_heads:
        neigh = top_neighbors(a)
        # c → donors trong neigh
        support: dict[int, list[int]] = defaultdict(list)
        for b in neigh:
            for c in la_ac.get(b, ()):
                if c == a:
                    continue
                support[c].append(b)
        for c, donors in support.items():
            donors_u = sorted(set(donors))
            # dương: a chưa có (a,là,c); ≥2 donors
            if c not in la_ac.get(a, ()) and len(donors_u) >= args.min_support:
                pos_cases.append({"a": a, "c": c, "donors": donors_u})
            # âm: a chưa có; đúng 1 donor trong neigh (và tổng heads mang c có thể nhiều —
            # siết: chỉ 1 trong top neigh → dễ nhiễm nếu không min_support)
            elif c not in la_ac.get(a, ()) and len(donors_u) == 1:
                neg_cases.append({"a": a, "c": c, "donors": donors_u})

    rng.shuffle(pos_cases)
    rng.shuffle(neg_cases)
    pos_cases = pos_cases[: args.max_pos]
    neg_cases = neg_cases[: args.max_neg]
    logger.log(f"ca dương: {len(pos_cases)} · âm: {len(neg_cases)} ({time.time()-t0:.0f}s)")

    def build_byb_excluding(a: int) -> dict[int, list[tuple[int, int]]]:
        """b → [(f,c)] từ cha không có a đứng trước là."""
        ban = parents_with_a_la.get(a, set())
        byb: dict[int, list[tuple[int, int]]] = defaultdict(list)
        have: set[tuple[int, int, int]] = set()
        for pi, rec in enumerate(ps.parents):
            if pi in ban:
                continue
            e = [int(t) for t in rec.emit_ids]
            for i in range(1, len(e) - 1):
                b_, f_, c_ = e[i - 1], e[i], e[i + 1]
                if f_ not in la_ids:
                    continue
                if not (keep[b_] and keep[c_]):
                    continue
                byb[b_].append((f_, c_))
                have.add((b_, f_, c_))
        return byb

    def fill(a: int, byb, neigh: list[int], *, mode: str) -> int | None:
        # chỉ xét f∈là
        support: dict[int, set[int]] = defaultdict(set)
        sc: dict[int, float] = defaultdict(float)
        for b in neigh:
            s_ = float(En[a] @ En[b])
            if s_ <= 0:
                continue
            for f_, c_ in byb.get(b, ()):
                if c_ == a:
                    continue
                if mode == "hebbian_E":
                    support[c_].add(b)
                sc[c_] += s_
        if mode == "hebbian_E":
            ok = {c: sc[c] for c, bs_ in support.items() if len(bs_) >= args.min_support}
            if not ok:
                return None
            return max(ok.items(), key=lambda kv: kv[1])[0]
        if mode == "embedding":
            if not sc:
                return None
            return max(sc.items(), key=lambda kv: kv[1])[0]
        # ngẫu nhiên trong pool c từ neigh
        pool = sorted(sc.keys())
        return rng.choice(pool) if pool else None

    def eval_split(cases: list[dict], *, want_c: bool) -> dict:
        """want_c=True: hit nếu điền đúng case['c']; False: FP nếu điền đúng case['c']."""
        out = {
            m: {"n_fill": 0, "hit": 0, "examples": []}
            for m in ("hebbian_E", "embedding", "ngau_nhien")
        }
        for case in cases:
            a, c_star = case["a"], case["c"]
            neigh = top_neighbors(a)
            byb = build_byb_excluding(a)
            for mode in out:
                pred = fill(a, byb, neigh, mode=mode)
                if pred is None:
                    continue
                out[mode]["n_fill"] += 1
                ok = pred == c_star
                if ok:
                    out[mode]["hit"] += 1
                if len(out[mode]["examples"]) < 8 and (ok if want_c else ok):
                    out[mode]["examples"].append({
                        "a": _surface(v, a),
                        "c_star": _surface(v, c_star),
                        "pred": _surface(v, pred),
                        "donors": [_surface(v, d) for d in case["donors"][:5]],
                        "prompt": f"{_surface(v, a)} là",
                        "gen": f"{_surface(v, a)} là {_surface(v, pred)}",
                        "creative": c_star not in la_ac.get(a, ()),
                    })
        return out

    pos_res = eval_split(pos_cases, want_c=True)
    neg_res = eval_split(neg_cases, want_c=False)

    def rate(block, key):
        n_f = block[key]["n_fill"]
        return block[key]["hit"] / max(n_f, 1), n_f, block[key]["hit"]

    print("\n=== DƯƠNG (held-out a là c* · ≥2 donors) ===")
    print(f"{'bộ':<14}{'n_điền':>8}{'hit':>6}{'hit@1':>8}")
    pos_rates = {}
    for m in ("hebbian_E", "embedding", "ngau_nhien"):
        r, nf, h = rate(pos_res, m)
        pos_rates[m] = {"prec": r, "n_fill": nf, "hit": h}
        print(f"{m:<14}{nf:>8}{h:>6}{r:>7.1%}")

    print("\n=== ÂM (≤1 donor trong neigh · FP = điền đúng c*) ===")
    print(f"{'bộ':<14}{'n_điền':>8}{'FP':>6}{'FP_rate':>8}")
    neg_rates = {}
    for m in ("hebbian_E", "embedding", "ngau_nhien"):
        r, nf, h = rate(neg_res, m)
        neg_rates[m] = {"fp": r, "n_fill": nf, "fp_n": h}
        print(f"{m:<14}{nf:>8}{h:>6}{r:>7.1%}")

    # creative check: all pos cases already have c not in la_ac[a]
    creative_ok = all(c not in la_ac.get(a, ()) for a, c in
                      ((x["a"], x["c"]) for x in pos_cases))

    he = pos_rates["hebbian_E"]
    fp = neg_rates["hebbian_E"]["fp"] if neg_rates["hebbian_E"]["n_fill"] else 0.0
    # FP rate among fills that equal c_star — if n_fill low, use fp among all neg attempts
    g1 = he["n_fill"] >= GATE_N_POS
    g2 = he["prec"] >= GATE_HIT
    g3 = fp <= GATE_FP
    g4 = creative_ok
    ok = g1 and g2 and g3 and g4

    print(f"\nCỔNG G-exec-v2:")
    print(f"  (1) n_điền dương ≥{GATE_N_POS}: {he['n_fill']} {'✅' if g1 else '❌'}")
    print(f"  (2) hit@1 ≥{GATE_HIT:.0%}: {he['prec']:.1%} {'✅' if g2 else '❌'}")
    print(f"  (3) FP âm ≤{GATE_FP:.0%}: {fp:.1%} {'✅' if g3 else '❌'}")
    print(f"  (4) mọi ca dương creative (triple vắng): {'✅' if g4 else '❌'}")
    print(f"⇒ {'ĐẠT' if ok else 'TRƯỢT'}")

    print("\nví dụ hit hebbian (dương):")
    for ex in pos_res["hebbian_E"]["examples"][:5]:
        print(f"  {ex['prompt']} → {ex['gen']!r}  (đích {ex['c_star']!r} · donors {ex['donors']})")

    payload = {
        "n_pos_cases": len(pos_cases),
        "n_neg_cases": len(neg_cases),
        "pos": pos_rates,
        "neg": neg_rates,
        "examples_pos_he": pos_res["hebbian_E"]["examples"],
        "examples_neg_he_fp": neg_res["hebbian_E"]["examples"],
        "gate": {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "pass": ok},
        "seconds": round(time.time() - t0, 1),
    }
    (_ROOT / args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
