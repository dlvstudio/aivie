#!/usr/bin/env python3
"""H1: tìm cha dời đô/Thăng Long và so act với top hiện tại."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.matrix_core import ensure_parent_bipartite, parent_coactivation_logits
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.sent_intra import _surface_content_folds


def main() -> None:
    gc.disable()
    state, _ = load_checkpoint("checkpoints/network_v12_sent.pkl")
    gc.enable()
    bi = ensure_parent_bipartite(state)
    ps = state.parents
    vocab = state.vocab
    prompt = "Nhà Lý dời đô về"
    hist = list(vocab.encode_prompt(prompt))
    print("hist tokens:", [vocab.token_of(t) for t in hist])
    n = len(vocab)
    _bag, succ, conf, act = parent_coactivation_logits(
        state, hist, vocab_size=n, top_k_parents=48
    )
    print(f"conf={conf:.3f} succ.max={succ.max():.4g} act.max={act.max():.4g}")

    hits: list[tuple[int, float, str]] = []
    for pi, em in enumerate(bi.emits):
        if not len(em):
            continue
        txt = decode_surface(state, list(em)[:48])
        if "Thăng Long" in txt and ("dời" in txt or "đô" in txt or "Lý" in txt):
            hits.append((pi, float(act[pi]) if pi < act.size else -1.0, txt))
        elif "dời đô" in txt and "Lý" in txt:
            hits.append((pi, float(act[pi]) if pi < act.size else -1.0, txt))
    hits.sort(key=lambda x: -x[1])
    print(f"hits dời/ThăngLong: {len(hits)}")
    for pi, a, txt in hits[:20]:
        mid = list(getattr(ps.parents[pi], "match_ids", ()) or [])[:14]
        print(f"  pi={pi} act={a:.4g} match={[vocab.token_of(t) for t in mid]}")
        print(f"    emit={txt!r}")

    hf = _surface_content_folds(vocab, hist)
    print("hist folds", sorted(hf))
    scored: list[tuple[int, float, int, str]] = []
    for pi, em in enumerate(bi.emits):
        if not len(em):
            continue
        ef = _surface_content_folds(vocab, list(em))
        ov = len(hf & ef)
        if ov >= 3:
            scored.append(
                (
                    ov,
                    float(act[pi]) if pi < act.size else 0.0,
                    pi,
                    decode_surface(state, list(em)[:32]),
                )
            )
    scored.sort(key=lambda x: (-x[0], -x[1]))
    print("top overlap>=3:")
    for ov, a, pi, txt in scored[:15]:
        print(f"  ov={ov} act={a:.4g} pi={pi} {txt!r}")

    # raw A.T@ctx trước hard-AND — tái lập nhẹ
    import numpy as np

    ctx = np.zeros(bi.A.shape[0], dtype=np.float64)
    for tid in hist:
        tid = int(tid)
        if 0 <= tid < ctx.shape[0]:
            ctx[tid] += max(1.0, float(ps._gidf(tid)))  # noqa: SLF001
    raw = np.asarray(bi.A.T @ ctx).ravel()
    # check target parents raw vs after
    targets = [pi for pi, _, _ in hits[:10]]
    print("raw act (pre-spec) for targets:")
    for pi in targets[:10]:
        print(f"  pi={pi} raw={raw[pi]:.4g} final_act={act[pi]:.4g}")


if __name__ == "__main__":
    main()
