#!/usr/bin/env python3
"""
TK_027 — dựng checkpoint có nơ-ron THỰC-THỂ (title bài) từ network_v4_full.

network_v4_full.pkl → + Entity:title (co-act hồ-sơ bài) → network_v5_entity.pkl.
Mọi process dài ghi progress ra file (Luật observability).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.training.ingest_entities import ingest_entities
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4_full.pkl")
    ap.add_argument("--statements", default="data/real_vi/statements_doc.jsonl")
    ap.add_argument("--out", default="checkpoints/network_v5_entity.pkl")
    ap.add_argument("--top-tokens", type=int, default=40)
    ap.add_argument("--min-sentences", type=int, default=5)
    args = ap.parse_args()

    log = get_run_logger("build_entity")
    ckpt = _ROOT / args.checkpoint
    log.log(f"LOAD {ckpt.name}")
    state, cache = load_checkpoint(ckpt, logger=log)
    v0 = len(state.vocab)
    log.log(f"trước: vocab={v0}")

    with log.phase("ingest_entities"):
        stats = ingest_entities(
            state, str(_ROOT / args.statements),
            top_tokens=args.top_tokens, min_sentences=args.min_sentences, logger=log,
        )
    log.log(f"ingest: {stats} → vocab={len(state.vocab)} (+{len(state.vocab)-v0})")

    out = _ROOT / args.out
    with log.phase("save checkpoint"):
        save_checkpoint(state, out, cache)
    log.log(f"✓ SAVED {out.name}")


if __name__ == "__main__":
    main()
