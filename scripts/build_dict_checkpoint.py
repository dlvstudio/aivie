#!/usr/bin/env python3
"""
TK_015 D1 / TK_016 D1′ — dựng checkpoint substrate TỪ ĐIỂN (network_v4).

Load `_t3_isg` → nạp từ điển (sense/POS + W_seq, **không** flat-dump parent) →
bind_sense_resonance (hút occurrence corpus) → rebuild embedding → save.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.embedding import build_embedding
from nn_aivie.training.frame_bracket import mine_frame_brackets
from nn_aivie.training.ingest_dictionary import ingest_dictionary
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]
_RAW = _ROOT / "data/dict/vi_dict_raw.jsonl"


def _group_entries(force: set[str], max_words: int | None) -> list[dict]:
    """Group raw rows (per-sense, ordered) → [{word, senses:[{pos,meaning,example}]}]."""
    entries: list[dict] = []
    cur_word: str | None = None
    for line in _RAW.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        w = str(row.get("word", "")).strip()
        if not w:
            continue
        sense = {"pos": row.get("part_of_speech", ""),
                 "meaning": row.get("meaning", ""), "example": row.get("example", "")}
        if w != cur_word:
            entries.append({"word": w, "senses": [sense]})
            cur_word = w
        else:
            entries[-1]["senses"].append(sense)
    if max_words is not None and len(entries) > max_words:
        kept = entries[:max_words]
        have = {e["word"] for e in kept}
        kept += [e for e in entries if e["word"] in force and e["word"] not in have]
        entries = kept
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/grade_v3_big_a2_t3_isg.pkl")
    ap.add_argument("--out", default="checkpoints/network_v4.pkl")
    ap.add_argument("--max-words", type=int, default=None)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--skip-resonance", action="store_true",
                    help="chỉ ingest substrate (debug nhanh)")
    args = ap.parse_args()

    log = get_run_logger("build_dict")
    force = {"đá", "đông", "chân", "chạy", "bàn", "cân", "cổ", "là"}
    entries = _group_entries(force, args.max_words)
    log.log(f"từ điển: {len(entries)} từ (raw {sum(1 for _ in _RAW.open())} nghĩa)")

    state, cache = load_checkpoint(args.checkpoint, logger=log)
    v0, p0 = len(state.vocab), len(state.parents.parents)
    log.log(f"trước: vocab={v0} parents={p0}")

    with log.phase("ingest_dictionary"):
        stats = ingest_dictionary(state, entries, logger=log, log_every=2000)
    log.log(f"ingest: {stats} → vocab={len(state.vocab)} (+{len(state.vocab)-v0}) "
            f"parents={len(state.parents.parents)} (+{len(state.parents.parents)-p0})")

    if not args.skip_resonance and state.parents is not None:
        with log.phase("bind_sense_resonance"):
            rstats = state.parents.bind_sense_resonance(
                state.adjacency, state.vocab, state.embedding, logger=log,
            )
        log.log(f"resonance: {rstats}")

    with log.phase("rebuild embedding"):
        state.embedding = build_embedding(state.adjacency, state.vocab, dim=args.dim)
    log.log(f"E shape={state.embedding.matrix.shape}")

    with log.phase("mine FRAME_BRACKET (persist — gỡ lazy-mine F3)"):
        if len(state.adjacency.layer(EdgeKind.FRAME_BRACKET)) == 0:
            log.log(str(mine_frame_brackets(state, logger=log)))
        else:
            log.log(f"FRAME_BRACKET edges={len(state.adjacency.layer(EdgeKind.FRAME_BRACKET))}")

    with log.phase("save checkpoint"):
        save_checkpoint(state, args.out, cache, bake_derived=True, logger=log)
    log.log(f"XONG → {args.out}")


if __name__ == "__main__":
    main()
