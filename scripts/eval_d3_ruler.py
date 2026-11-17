#!/usr/bin/env python3
"""
TK_016 D3 + G-probe — đo FRAME_BRACKET trên thước không bão hoà.

A/B: (base-LM) vs các cách cắm frame-bracket:
  - global (đã âm tính)
  - boundary-gated (đã âm tính)
  - entropy-gated (G-probe): chỉ fire khi base-LM phân vân + có opener trong hist

THẤP bits / CAO top1 = tốt. Không save checkpoint (mine in-process).
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


_SENT_END = {'.', '!', '?'}  # CHI de chia moc do, KHONG vao duong sinh


def _distribution_entropy(scores: np.ndarray) -> float:
    """Entropy (bits) trên phân bố softmax của ứng viên finite dương."""
    mask = np.isfinite(scores) & (scores > 0)
    finite = scores[mask]
    if finite.size == 0:
        return 0.0
    total = finite.sum()
    if total <= 0:
        return 0.0
    p = finite / total
    return float(-np.sum(p * np.log2(p + 1e-30)))


def _has_opener(state, prefix: list[int], *, window: int = 25) -> bool:
    """Có token mở khung (outgoing FRAME_BRACKET) trong cửa sổ lịch sử?"""
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    for a in prefix[-window:]:
        if fb.row(a):
            return True
    return False


def _fb_boost(state, prefix, vocab_size, *, window=25, decay=0.9):
    """Vector nâng token-đóng: mọi opener trong `window` token cuối → cộng FRAME_BRACKET row."""
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    boost = np.zeros(vocab_size, dtype=np.float64)
    hist = prefix[-window:]
    for d, a in enumerate(reversed(hist)):
        row = fb.row(a)
        if not row:
            continue
        w = decay ** d
        for b, wv in row.items():
            if b < vocab_size:
                boost[b] += w * wv
    return boost


def _apply_fb(scores, boost, *, fb_weight: float, fb_floor: float) -> np.ndarray:
    bmax = boost.max()
    if bmax <= 0:
        return scores
    return scores * (fb_floor + fb_weight * boost / bmax)


def _ranks(
    state,
    texts,
    *,
    mode: str = "none",
    fb_weight: float = 1.0,
    fb_floor: float = 0.2,
    entropy_min: float = 3.0,
    window: int = 25,
    decay: float = 0.9,
    max_len: int = 40,
    micro_floor: float = 0.15,
):
    """Chấm mean log2(rank). `mode`: none | global | boundary | entropy."""
    vocab_size = len(state.vocab)
    sum_bits = 0.0
    t1 = 0
    n = 0
    n_fire = 0
    for text in texts:
        from nn_aivie.teaching.align import tokenize

        ids = list(state.vocab.encode_tokens(tokenize(text)))
        for i in range(1, min(len(ids), max_len)):
            gold = ids[i]
            if gold >= vocab_size or not is_surface_token(state.vocab.token_of(gold)):
                continue
            prefix = ids[:i]
            scores = _base_lm_scores(state, ids[i - 1], vocab_size, micro_floor)
            if not np.isfinite(scores[gold]):
                continue

            fire = False
            if mode == "global":
                fire = True
            elif mode == "boundary":
                fire = state.vocab.token_of(ids[i - 1]) in _SENT_END  # eval-only: chia moc DO
            elif mode == "entropy":
                ent = _distribution_entropy(scores)
                fire = ent >= entropy_min and _has_opener(state, prefix, window=window)

            if fire:
                boost = _fb_boost(state, prefix, vocab_size, window=window, decay=decay)
                scores = _apply_fb(scores, boost, fb_weight=fb_weight, fb_floor=fb_floor)
                n_fire += 1

            gs = scores[gold]
            finite = np.isfinite(scores)
            rank = 1 + int(np.sum(scores[finite] > gs))
            sum_bits += math.log2(rank)
            t1 += rank <= 1
            n += 1
    return sum_bits / n if n else 0.0, t1 / n if n else 0.0, n, n_fire


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4.pkl")
    ap.add_argument("--fb-weight", type=float, default=0.5)
    ap.add_argument("--fb-floor", type=float, default=0.2)
    ap.add_argument("--entropy-min", type=float, default=3.0,
                    help="Ngưỡng entropy (bits) tối thiểu để fire gate")
    ap.add_argument("--sweep-entropy", action="store_true",
                    help="Sweep entropy_min ∈ {2,3,4,5,6} × fb_weight ∈ {0.3,0.5,1.0}")
    args = ap.parse_args()

    log = get_run_logger("eval_d3")
    heldout = _ROOT / "data/eval_v4/heldout_lm.jsonl"
    texts = [
        json.loads(line)["text"]
        for line in heldout.read_text("utf-8").splitlines()
        if line.strip()
    ]
    state, _ = load_checkpoint(args.checkpoint, logger=log)
    with log.phase("mine_frame_brackets"):
        log.log(str(mine_frame_brackets(state, logger=log)))

    with log.phase("ruler base"):
        b0, t0, n, _ = _ranks(state, texts, mode="none")

    print(f"\n=== D3 / G-probe A/B (held-out essay, n={n} vị trí) ===")
    print(f"  base-LM              : bits {b0:.3f}  top1 {t0:.1%}")

    with log.phase("ruler +fb global"):
        b1, t1, _, _ = _ranks(state, texts, mode="global",
                                fb_weight=args.fb_weight, fb_floor=args.fb_floor)
    print(f"  + fb global          : bits {b1:.3f}  top1 {t1:.1%}   (Δ {b1 - b0:+.3f})")

    with log.phase("ruler +fb boundary-gated"):
        b2, t2, _, _ = _ranks(state, texts, mode="boundary",
                                fb_weight=args.fb_weight, fb_floor=args.fb_floor)
    print(f"  + fb boundary-gated  : bits {b2:.3f}  top1 {t2:.1%}   (Δ {b2 - b0:+.3f})")

    if args.sweep_entropy:
        print("\n--- G-probe sweep (entropy-gated + opener) ---")
        best = (999.0, None)
        for ent in (2.0, 3.0, 4.0, 5.0, 6.0):
            for fw in (0.3, 0.5, 1.0):
                b, t, n_run, nf = _ranks(
                    state, texts, mode="entropy",
                    fb_weight=fw, fb_floor=args.fb_floor,
                    entropy_min=ent,
                )
                pct = 100.0 * nf / n_run if n_run else 0.0
                delta = b - b0
                print(
                    f"  ent≥{ent:.0f} w={fw:.1f}  bits {b:.3f}  top1 {t:.1%}"
                    f"  Δ {delta:+.3f}  fire {nf}/{n_run} ({pct:.1f}%)"
                )
                if delta < best[0]:
                    best = (delta, (ent, fw, b, t, nf, n_run))
        ent, fw, b, t, nf, n_run = best[1]
        print(
            f"\n  BEST Δ: ent≥{ent:.0f} w={fw:.1f} → bits {b:.3f} (Δ {b - b0:+.3f})"
            f"  fire {nf}/{n_run}"
        )
    else:
        with log.phase("ruler +fb entropy-gated"):
            b3, t3, _, nf = _ranks(
                state, texts, mode="entropy",
                fb_weight=args.fb_weight, fb_floor=args.fb_floor,
                entropy_min=args.entropy_min,
            )
        pct = 100.0 * nf / n if n else 0.0
        print(
            f"  + fb entropy-gated   : bits {b3:.3f}  top1 {t3:.1%}"
            f"   (Δ {b3 - b0:+.3f})  fire {nf}/{n} ({pct:.1f}%)"
            f"  [ent≥{args.entropy_min}, w={args.fb_weight}]"
        )


if __name__ == "__main__":
    main()
