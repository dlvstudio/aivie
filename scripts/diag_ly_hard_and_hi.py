#!/usr/bin/env python3
"""In hard-AND hi tokens + act của pi dời đô Thăng Long vs Tây Đô."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.matrix_core import ensure_parent_bipartite, parent_coactivation_logits


def main() -> None:
    gc.disable()
    state, _ = load_checkpoint("checkpoints/network_v12_sent.pkl")
    gc.enable()
    bi = ensure_parent_bipartite(state)
    ps = state.parents
    vocab = state.vocab
    hist = list(vocab.encode_prompt("Nhà Lý dời đô về"))
    print("hist", [vocab.token_of(t) for t in hist])
    for t in hist:
        print(f"  {vocab.token_of(t)!r} gidf={ps._gidf(t):.3f}")  # noqa: SLF001

    n = len(vocab)
    _b, succ, conf, act = parent_coactivation_logits(
        state, hist, vocab_size=n, top_k_parents=48
    )
    print(f"conf={conf:.3f} succ.max={succ.max():.4g}")
    targets = [99, 1322, 2814, 4347, 4254, 761]
    for pi in targets:
        em = [vocab.token_of(int(x)) for x in bi.emits[pi][:16]]
        print(f"pi={pi} act={act[pi]:.4g} emit={em}")

    # manual hi reconstruction
    gidf_hi = 4.0
    first_pos = {}
    for i, tid in enumerate(hist):
        tid = int(tid)
        if tid not in first_pos:
            first_pos[tid] = i
    late_cut = max(0, len(hist) - 3)
    late_ids = {tid for tid, pos0 in first_pos.items() if pos0 >= late_cut}
    early_ids = {tid for tid, pos0 in first_pos.items() if pos0 < late_cut}
    print("late", [vocab.token_of(t) for t in late_ids])
    print("early", [vocab.token_of(t) for t in early_ids])
    scored = []
    for tid, pos0 in first_pos.items():
        row = bi.A.getrow(tid)
        if row.nnz == 0:
            continue
        g = float(ps._gidf(tid))  # noqa: SLF001
        if g < gidf_hi:
            print(f"  skip sticky {vocab.token_of(tid)!r} gidf={g:.3f}<{gidf_hi}")
            continue
        sticky = g / ((1.0 + float(pos0)) ** 1.5)
        scored.append((sticky, -pos0, tid, g))
    scored.sort(reverse=True)
    print("scored sticky:")
    for s, _, tid, g in scored:
        print(f"  {vocab.token_of(tid)!r} sticky={s:.3f} gidf={g:.3f}")
    early_by_gidf = []
    for tid in early_ids:
        row = bi.A.getrow(tid)
        if row.nnz == 0:
            continue
        g = float(ps._gidf(tid))  # noqa: SLF001
        if g < gidf_hi:
            print(f"  skip early {vocab.token_of(tid)!r} gidf={g:.3f}")
            continue
        early_by_gidf.append((g, tid))
    early_by_gidf.sort(reverse=True)
    print("early_by_gidf", [(vocab.token_of(t), g) for g, t in early_by_gidf])
    hi = []
    for s, _, tid, g in scored:
        if tid in late_ids:
            hi.append(tid)
            break
    for g, tid in early_by_gidf:
        if tid not in hi:
            hi.append(tid)
            break
    print("hi AND", [vocab.token_of(t) for t in hi])
    # check A hits
    for pi in targets:
        ok = all(bi.A[tid, pi] > 0 for tid in hi[:2]) if len(hi) >= 2 else False
        print(f"  pi={pi} passes_hard={ok}")


if __name__ == "__main__":
    main()
