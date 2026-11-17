#!/usr/bin/env python3
"""
TK_028 — train data SỬ curated (history_vi.jsonl) vào mạng: parent priority CAO + entity.

network_v6_knowledge + history_vi.jsonl → network_v7_history.pkl:
  1. ingest_statements(câu sử, priority=0.85 — curated, lấn wiki 0.6 cho chủ đề sử).
  2. ingest_entities(statements_doc + add + history) → hồ-sơ Entity mạnh hơn (Nhà Lý/Văn Miếu…).
Ghi progress ra file (Luật observability).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.training.ingest import ingest_statements
from nn_aivie.training.ingest_entities import ingest_entities
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]


def _sentences(path: Path) -> list[str]:
    return [json.loads(l).get("text", "") for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v6_knowledge.pkl")
    ap.add_argument("--history", default="data/real_vi/history_vi.jsonl")
    ap.add_argument("--out", default="checkpoints/network_v7_history.pkl")
    ap.add_argument("--priority", type=float, default=0.85)
    args = ap.parse_args()

    log = get_run_logger("build_history")
    state, cache = load_checkpoint(_ROOT / args.checkpoint, logger=log)
    p0 = len(state.parents.parents)
    log.log(f"trước: parents={p0}")

    sents = _sentences(_ROOT / args.history)
    log.log(f"câu sử curated: {len(sents)}")
    with log.phase("ingest_statements (parents, priority cao)"):
        r = ingest_statements(state, sents, priority=args.priority, split_punct=True, logger=log)
    log.log(f"parents +{len(state.parents.parents)-p0} ({r})")

    with log.phase("ingest_entities (gộp mọi nguồn title)"):
        docs = [
            str(_ROOT / "data/real_vi/statements_doc.jsonl"),
            str(_ROOT / "data/real_vi/statements_doc_add.jsonl"),
            str(_ROOT / args.history),
        ]
        docs = [d for d in docs if Path(d).exists()]
        es = ingest_entities(state, docs, top_tokens=60, min_sentences=4, logger=log)
    log.log(f"entities: {es}")

    with log.phase("save"):
        save_checkpoint(state, _ROOT / args.out, cache)
    log.log(f"✓ SAVED {args.out}")


if __name__ == "__main__":
    main()
