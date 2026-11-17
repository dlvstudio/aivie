#!/usr/bin/env python
"""PHA 2 (§12.114) — bake-off DỰ ĐOÁN next-token: W_seq(d=1) vs W_skip-backoff vs VarLM(n-gram liền).

De-risk "W_skip thay W_seq". Đo held-out top-1: dựng 3 bộ dự đoán TỪ TRAIN, chấm trên TEST — CÙNG
một tập test, CÙNG một tập vị trí đủ điều kiện. Ba bộ:

  W_seq(d=1)        argmax_b count(t_i → b)                          ← chính là W_skip[d=1]
  W_skip-backoff    argmax_b Σ_{d=1..D} λ^{d-1} P(b | t_{i-d+1}, d)  ← bỏ phiếu skip có gap
  VarLM (n-gram)    stupid-backoff: hậu tố LIỀN dài nhất khớp        ← "= VarLM" [[long-range...]]

Cổng G-pred (§12.114): W_skip-backoff so VarLM. Đo cũ 43k: skip có-gap TRUNG TÍNH, VarLM +7.3pp.
Mục tiêu lộ trình = W_skip KHÔNG thua (hoặc VarLM tái hiện được như view của W_skip).

Luật 0: chỉ đếm + backoff, không backprop.
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.sent_intra import iter_sentence_spans  # noqa: E402


def spans_of(state, parents):
    v = state.vocab
    out = []
    for rec in parents:
        sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        for sp in iter_sentence_spans(sq, v):
            toks = [int(t) for t in sp]
            if len(toks) >= 2:
                out.append(toks)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--held-out", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-max", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--ngram", type=int, default=4, help="bậc n-gram liền tối đa (VarLM)")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument(
        "--max-pos",
        type=int,
        default=0,
        help="giới hạn số vị trí test (0 = hết). v12 full quá chậm pure-Python; 80k đủ cho G-pred",
    )
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()

    pars = list(state.parents.parents)
    idx = list(range(len(pars)))
    random.Random(args.seed).shuffle(idx)
    n_test = int(len(idx) * args.held_out)
    test_par = [pars[i] for i in idx[:n_test]]
    train_par = [pars[i] for i in idx[n_test:]]
    train = spans_of(state, train_par)
    test = spans_of(state, test_par)
    print(f"[{time.time()-t0:.0f}s] {len(pars):,} cha → train {len(train_par):,} "
          f"({len(train):,} span) · test {len(test_par):,} ({len(test):,} span)", flush=True)

    D = int(args.d_max)
    K = int(args.ngram)
    # --- dựng từ TRAIN ---
    skip = [defaultdict(Counter) for _ in range(D + 1)]     # skip[d][a][b]
    ngram = [defaultdict(Counter) for _ in range(K + 1)]    # ngram[o][ctx_tuple][b]
    for toks in train:
        L = len(toks)
        for i in range(L):
            for d in range(1, min(D, L - 1 - i) + 1):
                skip[d][toks[i]][toks[i + d]] += 1
            for o in range(1, K + 1):
                if i - o >= -1 and i + 1 < L and i - o + 1 >= 0:
                    ctx = tuple(toks[i - o + 1:i + 1])
                    if len(ctx) == o:
                        ngram[o][ctx][toks[i + 1]] += 1
    print(f"[{time.time()-t0:.0f}s] dựng xong: skip d1..{D}, n-gram 1..{K}", flush=True)

    # Precompute để khỏi sum/most_common lại mỗi vị trí (v12 hub Counter rất lớn).
    wseq_best = {
        a: c.most_common(1)[0][0] for a, c in skip[1].items() if c
    }
    skip_p: list[dict[int, dict[int, float]]] = [{} for _ in range(D + 1)]
    for d in range(1, D + 1):
        for a, c in skip[d].items():
            tot_c = float(sum(c.values())) or 1.0
            skip_p[d][a] = {
                b: cnt / tot_c for b, cnt in c.items() if cnt >= args.min_count
            }

    def pred_wseq(hist):
        return wseq_best.get(hist[-1])

    def pred_skip(hist):
        score: dict[int, float] = {}
        best_b, best_s = None, -1.0
        for d in range(1, min(D, len(hist)) + 1):
            dist = skip_p[d].get(hist[-d])
            if not dist:
                continue
            w = args.lam ** (d - 1)
            for b, p in dist.items():
                s = score.get(b, 0.0) + w * p
                score[b] = s
                if s > best_s:
                    best_s, best_b = s, b
        return best_b

    def pred_ngram(hist):
        for o in range(min(K, len(hist)), 0, -1):     # backoff hậu tố dài nhất
            ctx = tuple(hist[-o:])
            c = ngram[o].get(ctx)
            if c:
                return c.most_common(1)[0][0]
        return None

    preds = {"W_seq(d=1)": pred_wseq, "W_skip-backoff": pred_skip, "VarLM(n-gram liền)": pred_ngram}
    hit = {k: 0 for k in preds}
    # gom vị trí rồi (tuỳ chọn) lấy mẫu — cùng seed với held-out
    positions: list[tuple[list[int], int]] = []
    for toks in test:
        for i in range(len(toks) - 1):
            positions.append((toks[: i + 1], toks[i + 1]))
    rng = random.Random(args.seed + 7)
    if args.max_pos and len(positions) > args.max_pos:
        positions = rng.sample(positions, args.max_pos)
        print(f"[{time.time()-t0:.0f}s] chấm {len(positions):,} vị trí "
              f"(mẫu từ {sum(len(t)-1 for t in test):,} — --max-pos={args.max_pos})",
              flush=True)
    else:
        print(f"[{time.time()-t0:.0f}s] chấm {len(positions):,} vị trí", flush=True)

    tot = 0
    for hist, gold in positions:
        tot += 1
        if tot % 20_000 == 0:
            print(f"  … {tot:,}/{len(positions):,} ({time.time()-t0:.0f}s)", flush=True)
        for name, fn in preds.items():
            if fn(hist) == gold:
                hit[name] += 1
    print(f"\n[{time.time()-t0:.0f}s] held-out next-token top-1  (n_test={tot:,} vị trí):\n")
    base = hit["VarLM(n-gram liền)"] / tot if tot else 0.0
    for name in ("W_seq(d=1)", "W_skip-backoff", "VarLM(n-gram liền)"):
        acc = hit[name] / tot if tot else 0.0
        delta = "" if name == "VarLM(n-gram liền)" else f"  (vs VarLM {100*(acc-base):+.2f}pp)"
        print(f"    {name:20s} {100*acc:6.2f}%{delta}")

    ws = hit["W_skip-backoff"] / tot if tot else 0.0
    d = 100 * (ws - base)
    print("\n  G-pred:",
          "W_skip ≥ VarLM ⇒ thay được cả VarLM" if d >= 0 else
          ("hoà thực dụng (giữ VarLM as view)" if d >= -2 else
           "skip có-gap THUA n-gram liền ⇒ W_skip gánh embedding+nghiệm thu, KHÔNG gánh next-token"))


if __name__ == "__main__":
    main()
