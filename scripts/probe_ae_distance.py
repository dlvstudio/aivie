#!/usr/bin/env python
"""PHA ③ (§12.114) — A/E theo khoảng cách: match skip-gram vs túi-từ, đo thu hồi cha NGUỒN.

A hiện là túi thuần (`match × √power × gidf`, bỏ thứ tự). Ý tác giả: cặp token khớp ở đúng khoảng
cách phải mạnh hơn. Đo: query = token-nội-dung (theo thứ tự) của một cha; xếp hạng MỌI cha; cha
NGUỒN là đáp đúng. So 3 scorer bằng rank-1 / MRR của cha nguồn:

  bag       Σ_{t∈query} gidf(t)·[t ∈ cha]                      ← túi hiện tại
  skip      Σ_{(a,b) kề trong query} gidf(a)gidf(b)·[(a,b) là bigram-nội-dung của cha]  ← thứ tự
  combined  bag + w·skip

Nếu skip/combined thu hồi cha nguồn TỐT HƠN bag ⇒ khoảng cách loại được cha dương-tính-giả (chung
token, sai cấu trúc). Trần đã đo họ này thấp (skipxcov +0.007, cov gánh 95%) — đây kiểm chính xác.
Luật 0: đếm + gidf, closed-form.
"""
from __future__ import annotations

import argparse
import gc
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
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-query", type=int, default=300, help="số cha lấy làm query")
    ap.add_argument("--q-len", type=int, default=3, help="độ dài query NGẮN (bag sẽ hoà hàng loạt)")
    ap.add_argument("--skip-w", type=float, default=2.0)
    ap.add_argument("--covdir-beta", type=float, default=4.0,
                    help="§12.124: order làm MULTIPLIER trên cov (bắt chước rank_matches TK_029)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    n = len(v)
    struct = [is_structural_token(v, i) for i in range(n)]
    _gidf_fn = getattr(state.parents, "_gidf", None)

    def gidf(t):
        if callable(_gidf_fn):
            try:
                return float(_gidf_fn(int(t)))
            except (TypeError, ValueError, IndexError):
                return 1.0
        return 1.0

    # nội dung + bigram-nội-dung của mỗi cha
    pars = list(state.parents.parents)
    P = len(pars)
    cont_set: list[set] = []
    bigr_set: list[set] = []
    first_seq: list[list] = []
    for rec in pars:
        sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        toks = []
        for sp in iter_sentence_spans(sq, v):
            toks += [int(t) for t in _content_positions(sp, v, content_only=True, struct_mask=struct)]
        cont_set.append(set(toks))
        bigr_set.append(set(zip(toks, toks[1:])))
        first_seq.append(toks)

    # đảo chỉ mục token → cha (để không quét toàn bộ P mỗi query)
    tok2par = defaultdict(list)
    for pi, s in enumerate(cont_set):
        for t in s:
            tok2par[t].append(pi)
    big2par = defaultdict(list)
    for pi, s in enumerate(bigr_set):
        for ab in s:
            big2par[ab].append(pi)

    idx = list(range(P))
    random.Random(args.seed).shuffle(idx)
    q_ids = [i for i in idx if len(first_seq[i]) >= 3][:args.n_query]
    print(f"[{time.time()-t0:.0f}s] {P:,} cha · {len(q_ids)} query (≥3 token nội dung)", flush=True)

    def ranks(src, query):
        qset = list(dict.fromkeys(query))            # unique giữ thứ tự
        qbig = list(zip(query, query[1:]))
        bag = defaultdict(float)
        for t in qset:
            g = gidf(t)
            for pi in tok2par.get(t, ()):
                bag[pi] += g
        skip = defaultdict(float)
        for a, b in qbig:
            g = gidf(a) * gidf(b)
            for pi in big2par.get((a, b), ()):
                skip[pi] += g
        comb = defaultdict(float)
        for pi, s in bag.items():
            comb[pi] += s
        for pi, s in skip.items():
            comb[pi] += args.skip_w * s
        # covdir: bắt chước rank_matches TK_029 — cov là CHÍNH, order là MULTIPLIER bị chặn.
        # score = bag · (1 + β · order_hit / max_order). Kiểm cov có NUỐT lãi order không.
        max_ord = max(skip.values()) if skip else 1.0
        covdir = defaultdict(float)
        for pi, s in bag.items():
            covdir[pi] = s * (1.0 + args.covdir_beta * (skip.get(pi, 0.0) / max_ord))
        out = {}
        for nm, sc in (("bag", bag), ("skip", skip), ("comb", comb), ("covdir", covdir)):
            # hạng KỲ VỌNG của src có xét HOÀ: #(điểm cao hơn) + (#hoà + 1)/2.
            # Hoà quan trọng: query ngắn ⇒ bag cho hàng loạt cha CÙNG điểm ⇒ src chìm trong đám hoà.
            ssrc = sc.get(src, 0.0)
            if ssrc <= 0:
                out[nm] = float(P)
                continue
            higher = sum(1 for x in sc.values() if x > ssrc)
            ties = sum(1 for x in sc.values() if x == ssrc)   # gồm cả src
            out[nm] = higher + (ties + 1) / 2.0
        return out

    rng = random.Random(args.seed + 1)
    agg = {nm: {"r1": 0.0, "mrr": 0.0} for nm in ("bag", "skip", "comb", "covdir")}
    for src in q_ids:
        seq = first_seq[src]
        # query = cửa sổ NGẮN q_len token nội dung liền, vị trí ngẫu nhiên trong cha
        L = len(seq)
        st_ = rng.randint(0, max(0, L - args.q_len))
        query = seq[st_:st_ + args.q_len]
        rk = ranks(src, query)
        for nm in agg:
            r = rk[nm]
            agg[nm]["r1"] += 1.0 if r <= 1.5 else 0.0     # kỳ vọng-rank ≤1.5 ≈ đứng đầu (ít hoà)
            agg[nm]["mrr"] += 1.0 / r
    m = len(q_ids)
    print(f"\n[{time.time()-t0:.0f}s] thu hồi cha NGUỒN ({m} query):\n")
    print(f"  {'scorer':10s} {'rank-1':>8s} {'MRR':>8s}")
    for nm in ("bag", "skip", "comb", "covdir"):
        print(f"  {nm:10s} {100*agg[nm]['r1']/m:7.2f}% {agg[nm]['mrr']/m:8.3f}")
    dr1 = 100 * (agg["comb"]["r1"] - agg["bag"]["r1"]) / m
    print(f"\n  ③ combined − bag: rank-1 {dr1:+.2f}pp  "
          f"({'CÓ giá trị' if dr1 >= 2 else 'nhỏ/không — khớp trần skipxcov +0.007'})")


if __name__ == "__main__":
    main()
