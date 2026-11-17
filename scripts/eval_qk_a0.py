#!/usr/bin/env python3
"""
TK_017 R2 / doc 03 A0 — thí nghiệm rẻ chặn công lớn cho q·k directed.

Ba phép đo (held-out essay, KHÔNG retrain):
  1. ĐỌC TAY: top-k "đứng sau theo khuôn" của vài opener/content token — có nghĩa không?
  2. RULER A/B: bits/top1 base-LM vs micro-blend(raw ⊕ q·k) — aggregate không được tụt.
  3. UNSEEN: tập vị trí mà bigram thô MÙ (W_seq[w0→gold]=0) — q·k có kéo rank gold lên không?
     (Đây là chỗ tổng-quát-hoá khuôn phải chứng minh; bigram thô cho rank ~∞.)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.vocab import is_surface_token
from nn_aivie.hca.attention_qk import ensure_directed_qk
from nn_aivie.hca.readout import mask_non_surface
from nn_aivie.teaching.align import tokenize
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]


def _scores(state, qk, w0, vocab_size, *, micro_floor=0.15, qk_weight=0.0):
    base = np.zeros(vocab_size, dtype=np.float64)
    for v, w in state.adjacency.row_readout(w0).items():
        if v < vocab_size and w > 0:
            base[v] = w
    micro = np.zeros(vocab_size, dtype=np.float64)
    for v, w in state.adjacency.layer(EdgeKind.SEQUENTIAL).row(w0).items():
        if v < vocab_size and w > 0:
            micro[v] = w
    mmax = micro.max() or 1.0
    micro = micro / mmax
    if qk_weight > 0:
        q = np.maximum(qk.scores(w0), 0.0)
        qmax = q.max() or 1.0
        micro = micro + qk_weight * (q / qmax)  # blend sau chuẩn hoá từng kênh (doc03 rủi ro 2)
    scores = base * (micro_floor + micro)
    return mask_non_surface(scores, state.vocab)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4.pkl")
    ap.add_argument("--qk-weight", type=float, default=0.5)
    args = ap.parse_args()
    log = get_run_logger("eval_qk_a0")
    texts = [json.loads(l)["text"] for l in (_ROOT / "data/eval_v4/heldout_lm.jsonl").read_text("utf-8").splitlines() if l.strip()]
    state, _ = load_checkpoint(args.checkpoint, logger=log)
    with log.phase("build q·k"):
        qk = ensure_directed_qk(state, logger=log)

    print("\n=== 1. ĐỌC TAY — top đứng-sau theo KHUÔN (q·k) ===")
    for t in ["nếu", "tuy", "vì", "không", "Đọc", "học", "trời", "màu"]:
        print(f"  '{t}' → {qk.top(state.vocab, t, 8)}")

    V = len(state.vocab)
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)

    def ruler(qk_weight):
        sb = t1 = n = 0
        su_bits = su_rr = su_n = 0
        for text in texts:
            ids = list(state.vocab.encode_tokens(tokenize(text)))
            for i in range(1, min(len(ids), 40)):
                gold = ids[i]
                if gold >= V or not is_surface_token(state.vocab.token_of(gold)):
                    continue
                sc = _scores(state, qk, ids[i - 1], V, qk_weight=qk_weight)
                if not np.isfinite(sc[gold]):
                    continue
                fin = np.isfinite(sc)
                r = 1 + int(np.sum(sc[fin] > sc[gold]))
                sb += math.log2(r)
                t1 += r <= 1
                n += 1
                if seq.row(ids[i - 1]).get(gold, 0.0) <= 0:  # UNSEEN transition
                    su_bits += math.log2(r)
                    su_rr += 1.0 / r
                    su_n += 1
        return sb / n, t1 / n, n, su_bits / max(su_n, 1), su_rr / max(su_n, 1), su_n

    with log.phase("ruler base"):
        b0, t0, n, ub0, ur0, un = ruler(0.0)
    with log.phase("ruler +q·k"):
        b1, t1_, _, ub1, ur1, _ = ruler(args.qk_weight)
    print(f"\n=== 2. RULER A/B (n={n}) ===")
    print(f"  base           : bits {b0:.3f}  top1 {t0:.1%}")
    print(f"  +q·k (w={args.qk_weight}) : bits {b1:.3f}  top1 {t1_:.1%}   (Δ {b1-b0:+.3f})")
    print(f"\n=== 3. UNSEEN transitions (W_seq=0; n={un}) ===")
    print(f"  base : bits {ub0:.3f}  MRR {ur0:.4f}")
    print(f"  +q·k : bits {ub1:.3f}  MRR {ur1:.4f}   (Δbits {ub1-ub0:+.3f})")


if __name__ == "__main__":
    main()
