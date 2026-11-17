#!/usr/bin/env python3
"""TK_036 S0 — probe skip spans trên checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.skip_span_constants import skip_right_tuple
from nn_aivie.training.skip_span import mine_skip_spans
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v10_histwiki_paragraph.pkl")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    log = get_run_logger("probe_skip_spans")
    with log.phase("load"):
        state, _ = load_checkpoint(_ROOT / args.checkpoint, logger=log)
    with log.phase("mine"):
        stats = mine_skip_spans(state, logger=log)
    ps = state.parents
    vocab = state.vocab
    spans = getattr(ps, "skip_spans", []) or []
    ranked = sorted(spans, key=lambda s: -s["power"] * s["pmi"])[: args.top]
    log.log(f"stats={stats}")
    log.log("top skip spans (left → right | power pmi promoted):")
    for s in ranked:
        lt = vocab.token_of(s["left"])
        rtoks = [vocab.token_of(t) for t in skip_right_tuple(s["right"])]
        rt = " ".join(rtoks)
        mode_gap = max(s["gaps"], key=s["gaps"].get) if s["gaps"] else 0
        log.log(
            f"  {lt!r} → {rt!r} | power={s['power']} pmi={s['pmi']:.2f} "
            f"mode_gap={mode_gap} promoted={s.get('promoted')}"
        )


if __name__ == "__main__":
    main()
