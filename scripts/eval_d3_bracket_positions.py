#!/usr/bin/env python3
"""
TK_017 Mode B — metric TARGETED: rank closer TẠI vị trí đóng-khung thật.

Chỉ chấm vị trí i khi token gold ids[i] là closer hợp lệ (FRAME_BRACKET row) của
một opener trong prefix ids[:i] (gap ≥ 2). Aggregate essay có thể âm trong khi
targeted dương — tách "tín hiệu đúng chỗ" vs "cách cắm sai".
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
from nn_aivie.eval.metrics import _base_lm_scores
from nn_aivie.training.frame_bracket import mine_frame_brackets
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]


def _is_bracket_close(state, prefix: list[int], gold: int, *, min_gap: int = 2) -> tuple[bool, int | None]:
    """Gold có phải closer của opener nào trong prefix (khoảng cách ≥ min_gap)?"""
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    if len(prefix) < min_gap + 1:
        return False, None
    for j, opener in enumerate(prefix):
        if len(prefix) - j < min_gap + 1:
            continue
        row = fb.row(opener)
        if gold in row and row[gold] > 0:
            return True, opener
    return False, None


def _fb_boost_from_opener(state, opener: int, vocab_size: int):
    boost = np.zeros(vocab_size, dtype=np.float64)
    row = state.adjacency.layer(EdgeKind.FRAME_BRACKET).row(opener)
    for b, wv in row.items():
        if b < vocab_size:
            boost[b] = wv
    return boost


def _eval_targeted(
    state,
    texts,
    *,
    fb_weight: float = 0.5,
    fb_floor: float = 0.2,
    max_len: int = 40,
    micro_floor: float = 0.15,
):
    """Single-pass A/B: cùng tập vị trí (gold finite ở base-LM) cho base vs +fb."""
    vocab_size = len(state.vocab)
    sum_bits_all = sum_bits_all_fb = sum_bits_tgt = sum_bits_tgt_fb = 0.0
    t1_all = t1_all_fb = t1_tgt = t1_tgt_fb = 0
    n_all = n_tgt = 0
    for text in texts:
        from nn_aivie.teaching.align import tokenize

        ids = list(state.vocab.encode_tokens(tokenize(text)))
        for i in range(1, min(len(ids), max_len)):
            gold = ids[i]
            if gold >= vocab_size or not is_surface_token(state.vocab.token_of(gold)):
                continue
            is_close, opener = _is_bracket_close(state, ids[:i], gold)
            scores = _base_lm_scores(state, ids[i - 1], vocab_size, micro_floor)
            if not np.isfinite(scores[gold]):
                continue
            finite = np.isfinite(scores)

            def _rank(sc):
                gs = sc[gold]
                return 1 + int(np.sum(sc[finite] > gs))

            r0 = _rank(scores)
            sum_bits_all += math.log2(r0)
            t1_all += r0 <= 1
            n_all += 1

            scores_fb = scores
            if is_close and opener is not None:
                boost = _fb_boost_from_opener(state, opener, vocab_size)
                bmax = boost.max()
                if bmax > 0:
                    scores_fb = scores * (fb_floor + fb_weight * boost / bmax)
            r1 = _rank(scores_fb)
            sum_bits_all_fb += math.log2(r1)
            t1_all_fb += r1 <= 1

            if is_close:
                sum_bits_tgt += math.log2(r0)
                t1_tgt += r0 <= 1
                sum_bits_tgt_fb += math.log2(r1)
                t1_tgt_fb += r1 <= 1
                n_tgt += 1

    def _avg(s, c):
        return s / c if c else 0.0

    return {
        "bits_all": _avg(sum_bits_all, n_all),
        "top1_all": _avg(t1_all, n_all),
        "n_all": n_all,
        "bits_tgt": _avg(sum_bits_tgt, n_tgt),
        "top1_tgt": _avg(t1_tgt, n_tgt),
        "n_tgt": n_tgt,
        "bits_all_fb": _avg(sum_bits_all_fb, n_all),
        "top1_all_fb": _avg(t1_all_fb, n_all),
        "bits_tgt_fb": _avg(sum_bits_tgt_fb, n_tgt),
        "top1_tgt_fb": _avg(t1_tgt_fb, n_tgt),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4_convpool.pkl")
    ap.add_argument("--fb-weight", type=float, default=0.5)
    args = ap.parse_args()
    log = get_run_logger("eval_d3_tgt")
    heldout = _ROOT / "data/eval_v4/heldout_lm.jsonl"
    texts = [
        json.loads(line)["text"]
        for line in heldout.read_text("utf-8").splitlines()
        if line.strip()
    ]
    state, _ = load_checkpoint(args.checkpoint, logger=log)
    with log.phase("mine_frame_brackets"):
        log.log(str(mine_frame_brackets(state, logger=log)))
    r = _eval_targeted(state, texts, fb_weight=args.fb_weight)
    print("\n=== Mode B — bracket-close positions (held-out essay) ===")
    print(f"  ALL positions     n={r['n_all']:4d}  bits {r['bits_all']:.3f}  top1 {r['top1_all']:.1%}")
    print(
        f"  TARGETED (close)  n={r['n_tgt']:4d}  bits {r['bits_tgt']:.3f}  top1 {r['top1_tgt']:.1%}"
    )
    print(f"\n  + fb @ close positions only (opener-specific boost, w={args.fb_weight}):")
    print(
        f"  ALL               bits {r['bits_all_fb']:.3f}  top1 {r['top1_all_fb']:.1%}"
        f"   (Δ {r['bits_all_fb']-r['bits_all']:+.3f})"
    )
    print(
        f"  TARGETED          bits {r['bits_tgt_fb']:.3f}  top1 {r['top1_tgt_fb']:.1%}"
        f"   (Δ {r['bits_tgt_fb']-r['bits_tgt']:+.3f})"
    )


if __name__ == "__main__":
    main()
