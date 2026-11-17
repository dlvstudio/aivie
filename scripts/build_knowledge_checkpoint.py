#!/usr/bin/env python3
"""
TK_027 — bổ sung KIẾN THỨC (sử VN + chung) vào mạng: fact wiki mới → PARENT (đáp được) + Entity.

network_v4_full + statements_doc_add.jsonl → network_v6_knowledge.pkl:
  1. ingest_statements(câu mới) → nơ-ron cha (priority 0.6, ngoài-miền — two-tier bảo vệ curated).
  2. ingest_entities(cả statements_doc + add) → hồ-sơ Entity:title (disambiguation TK_027).
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
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line).get("text", "")
            if t:
                out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4_full.pkl")
    ap.add_argument("--add", default="data/real_vi/statements_doc_add.jsonl")
    ap.add_argument("--base-docs", default="data/real_vi/statements_doc.jsonl")
    ap.add_argument("--out", default="checkpoints/network_v6_knowledge.pkl")
    args = ap.parse_args()

    log = get_run_logger("build_knowledge")
    state, cache = load_checkpoint(_ROOT / args.checkpoint, logger=log)
    p0 = len(state.parents.parents)
    log.log(f"trước: parents={p0} vocab={len(state.vocab)}")

    add_sents = _sentences(_ROOT / args.add)
    log.log(f"câu mới: {len(add_sents)}")
    with log.phase("ingest_statements (parents)"):
        r = ingest_statements(state, add_sents, priority=0.6, split_punct=True, logger=log)
    log.log(f"parents +{len(state.parents.parents)-p0} ({r})")

    with log.phase("ingest_entities"):
        es = ingest_entities(
            state, [str(_ROOT / args.base_docs), str(_ROOT / args.add)],
            top_tokens=60, logger=log,
        )
    log.log(f"entities: {es}")

    with log.phase("save"):
        save_checkpoint(state, _ROOT / args.out, cache)
    log.log(f"✓ SAVED {args.out}")


if __name__ == "__main__":
    main()
