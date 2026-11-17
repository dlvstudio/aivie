#!/usr/bin/env python3
"""TK_016 D3 — validate mine FRAME_BRACKET: nếu→thì, Hỏi:→Đáp:, vì→nên có nổi lên không."""

from __future__ import annotations

import argparse

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.training.frame_bracket import mine_frame_brackets, top_brackets
from nn_aivie.util.progress import get_run_logger

OPENERS = ["nếu", "vì", "bởi", "tuy", "Hỏi:", "do", "càng", "không", "vừa"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4.pkl")
    args = ap.parse_args()
    log = get_run_logger("probe_d3")
    state, _ = load_checkpoint(args.checkpoint, logger=log)
    with log.phase("mine_frame_brackets"):
        stats = mine_frame_brackets(state, logger=log)
    log.log(f"stats: {stats}")
    print("\n=== TOP FRAME_BRACKET (opener → đóng) ===")
    for op in OPENERS:
        tb = top_brackets(state, op, k=6)
        print(f"  '{op}' →  {', '.join(f'{t}·{w}' for t, w in tb) if tb else '(none)'}")


if __name__ == "__main__":
    main()
