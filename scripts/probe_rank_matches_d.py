#!/usr/bin/env python
"""P31 (§12.125e) — đo ③ THẬT qua ParentStore.rank_matches trên v12.

Khác probe_ae_distance (DIY scorer) và gen_dir_blind (ar_matrix_core=True → không gọi
rank_matches). Đây gọi API production: A = dmax=1/accum=max (TK_029) vs B = dmax=4/accum=sum
(③ opt-in). Metric: thu hồi cha nguồn + coverage-tie ở điểm top.

Cổng G-tie (§12.31): ở q-len=3, (rank-1 B−A) ≥ +5pp HOẶC tie-rate giảm ≥20% tương đối.
Luật 0: chỉ đo, không train, không đụng A.T@ctx.
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans  # noqa: E402

CONFIGS = (
    ("A_d1_max", {"directed_bigram_dmax": 1, "directed_bigram_accum": "max",
                  "directed_bigram_lam": 0.5}),
    ("B_d4_sum", {"directed_bigram_dmax": 4, "directed_bigram_accum": "sum",
                  "directed_bigram_lam": 0.5}),
)


def _apply_knobs(ps, knobs: dict) -> None:
    ps.directed_bigram_bonus = float(getattr(ps, "directed_bigram_bonus", 1.5) or 1.5)
    if ps.directed_bigram_bonus <= 0:
        ps.directed_bigram_bonus = 1.5  # đo cần kênh hướng BẬT
    ps.directed_bigram_min_idf = float(getattr(ps, "directed_bigram_min_idf", 2.0))
    for k, v in knobs.items():
        setattr(ps, k, v)


def _expected_rank(ranked: list[tuple[int, float]], src: int, n_par: int) -> float:
    """Hạng kỳ vọng có xét hoà: #(cao hơn) + (#hoà+1)/2. Ngoài list → n_par."""
    by_pi = {pi: sc for pi, sc in ranked}
    ssrc = by_pi.get(src)
    if ssrc is None or ssrc <= 0:
        return float(n_par)
    higher = sum(1 for sc in by_pi.values() if sc > ssrc)
    ties = sum(1 for sc in by_pi.values() if sc == ssrc)
    return higher + (ties + 1) / 2.0


def _tie_stats(ranked: list[tuple[int, float]]) -> tuple[bool, int]:
    """(có_hoà_top, |tie-set| tại điểm top). ranked đã sort giảm."""
    if not ranked:
        return False, 0
    top = ranked[0][1]
    n = sum(1 for _, sc in ranked if sc == top)
    return n >= 2, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-query", type=int, default=800)
    ap.add_argument("--q-len", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-n", type=int, default=0,
                    help="0 = mọi ứng viên scored (cần cho tie/rank đầy đủ)")
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    ps = state.parents
    n_par = len(ps.parents)
    struct = [is_structural_token(v, i) for i in range(len(v))]

    first_seq: list[list[int]] = []
    for rec in ps.parents:
        sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        toks: list[int] = []
        for sp in iter_sentence_spans(sq, v):
            toks += [int(t) for t in _content_positions(
                sp, v, content_only=True, struct_mask=struct)]
        first_seq.append(toks)

    idx = list(range(n_par))
    random.Random(args.seed).shuffle(idx)
    q_ids = [i for i in idx if len(first_seq[i]) >= args.q_len][: args.n_query]
    print(f"[{time.time()-t0:.0f}s] {n_par:,} cha · {len(q_ids)} query "
          f"(≥{args.q_len} token nội dung) · top_n="
          f"{'all' if args.top_n <= 0 else args.top_n}", flush=True)

    # cố định cửa sổ query (cùng seed+1 như probe_ae_distance)
    rng = random.Random(args.seed + 1)
    queries: list[tuple[int, list[int]]] = []
    for src in q_ids:
        seq = first_seq[src]
        st_ = rng.randint(0, max(0, len(seq) - args.q_len))
        queries.append((src, seq[st_: st_ + args.q_len]))

    top_n = n_par if args.top_n <= 0 else args.top_n
    results: dict[str, dict] = {}

    for name, knobs in CONFIGS:
        _apply_knobs(ps, knobs)
        r1 = mrr = 0.0
        n_tie = 0
        tie_sizes: list[int] = []
        miss = 0
        for qi, (src, query) in enumerate(queries, 1):
            if qi % 50 == 0 or qi == 1:
                print(f"  [{name}] {qi}/{len(queries)} ({time.time()-t0:.0f}s)", flush=True)
            ranked = ps.rank_matches(
                query, v, None, fuzzy_topk=0, top_n=top_n,
            )
            er = _expected_rank(ranked, src, n_par)
            if er >= n_par:
                miss += 1
            r1 += 1.0 if er <= 1.5 else 0.0
            mrr += 1.0 / er
            is_tie, tsz = _tie_stats(ranked)
            if is_tie:
                n_tie += 1
            tie_sizes.append(tsz)
        m = len(queries)
        results[name] = {
            "r1": r1 / m,
            "mrr": mrr / m,
            "tie_rate": n_tie / m,
            "mean_tie": sum(tie_sizes) / m if m else 0.0,
            "miss": miss,
            "m": m,
        }
        print(f"[{time.time()-t0:.0f}s] {name}: rank-1={100*results[name]['r1']:.2f}% "
              f"MRR={results[name]['mrr']:.3f} tie={100*results[name]['tie_rate']:.1f}% "
              f"mean|tie|={results[name]['mean_tie']:.2f} miss={miss}", flush=True)

    a, b = results["A_d1_max"], results["B_d4_sum"]
    d_r1 = 100 * (b["r1"] - a["r1"])
    if a["tie_rate"] > 0:
        d_tie_rel = 100 * (a["tie_rate"] - b["tie_rate"]) / a["tie_rate"]
    else:
        d_tie_rel = 0.0
    gate_r1 = d_r1 >= 5.0
    gate_tie = d_tie_rel >= 20.0
    ok = gate_r1 or gate_tie

    print(f"\n[{time.time()-t0:.0f}s] === P31 q-len={args.q_len} n={a['m']} ===")
    print(f"  {'config':12s} {'rank-1':>8s} {'MRR':>8s} {'tie%':>8s} {'mean|tie|':>10s}")
    for name in ("A_d1_max", "B_d4_sum"):
        r = results[name]
        print(f"  {name:12s} {100*r['r1']:7.2f}% {r['mrr']:8.3f} "
              f"{100*r['tie_rate']:7.1f}% {r['mean_tie']:10.2f}")
    print(f"\n  Δ rank-1 (B−A): {d_r1:+.2f}pp")
    print(f"  Δ tie-rate tương đối (A→B giảm): {d_tie_rel:+.1f}%")
    print(f"  G-tie: {'ĐẠT' if ok else 'TRƯỢT'} "
          f"(rank-1≥+5pp={gate_r1} · tie↓≥20%rel={gate_tie})")


if __name__ == "__main__":
    main()
