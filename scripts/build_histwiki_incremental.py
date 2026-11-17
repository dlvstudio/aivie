#!/usr/bin/env python3
"""
TK_024 C scale — nạp incremental history_wiki_para (priority 0.8) + refresh entities.

Dùng khi mở rộng `data/real_vi/history_wiki_para.jsonl` (fetch thêm bài). Có thể nạp toàn bộ
hoặc chỉ các title mới (--titles-file) để tránh trùng parent không cần thiết.

Ví dụ:
  python scripts/fetch_history_wiki.py --titles-file data/history_titles_expand.txt --append
  python scripts/build_histwiki_incremental.py \\
    --checkpoint checkpoints/network_v9_histwiki.pkl \\
    --titles-file data/history_titles_expand.txt \\
    --out checkpoints/network_v10_histwiki.pkl
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


def _load_titles(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    titles: set[str] = set()
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        titles.add(line)
    return titles


def _slug(title: str) -> str:
    return title.replace(" ", "_").replace("/", "_")


def _sentences(history: Path, titles: set[str] | None) -> list[str]:
    slug_titles: set[str] | None = None
    if titles is not None:
        slug_titles = {_slug(t) for t in titles}
    out: list[str] = []
    with history.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if titles is not None:
                title = row.get("title", "")
                doc_id = str(row.get("doc_id", ""))
                slug = doc_id[5:] if doc_id.startswith("hist:") else ""
                if title not in titles and slug not in slug_titles:
                    continue
            text = (row.get("text") or "").strip()
            if text:
                out.append(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v9_histwiki.pkl")
    ap.add_argument("--history", default="data/real_vi/history_wiki_para.jsonl")
    ap.add_argument("--titles-file", default=None, help="chỉ nạp title trong file (delta)")
    ap.add_argument("--out", default="checkpoints/network_v10_histwiki.pkl")
    ap.add_argument("--priority", type=float, default=0.8)
    ap.add_argument("--max-sentences", type=int, default=0, help="giới hạn số câu (0=toàn bộ)")
    args = ap.parse_args()

    log = get_run_logger("build_histwiki_incremental")
    titles = _load_titles(_ROOT / args.titles_file) if args.titles_file else None
    history = _ROOT / args.history

    state, cache = load_checkpoint(_ROOT / args.checkpoint, logger=log)
    p0 = len(state.parents.parents)
    log.log(f"trước: parents={p0}  titles_filter={len(titles) if titles else 'ALL'}")

    sents = _sentences(history, titles)
    if args.max_sentences and args.max_sentences > 0:
        sents = sents[: args.max_sentences]
    log.log(f"câu wiki lịch sử: {len(sents)}")
    if not sents:
        log.log("⚠ không có câu — kiểm tra --titles-file / fetch")
        return

    with log.phase("ingest_statements (wiki para, priority cao)"):
        ps = state.parents
        if ps is not None:
            ps.skip_spans = []
            ps._skip_lookup = {}
            ps.skip_ledger = None
            ps.ensure_skip_ledger(min_count=4, min_pmi=1.0)
        r = ingest_statements(state, sents, priority=args.priority, split_punct=True, logger=log)
    log.log(f"parents +{len(state.parents.parents) - p0} ({r})")
    if ps is not None and ps.skip_ledger is not None:
        ledger = ps.skip_ledger
        log.log(
            f"skip_ledger sau ingest: pairs={len(ledger.pair)} spans={len(ps.skip_spans)} "
            f"synced={ledger._synced}"
        )

    with log.phase("ingest_entities (gộp doc + history_wiki)"):
        docs = [
            str(_ROOT / "data/real_vi/statements_doc.jsonl"),
            str(_ROOT / "data/real_vi/statements_doc_add.jsonl"),
            str(_ROOT / "data/real_vi/history_vi.jsonl"),
            str(history),
        ]
        docs = [d for d in docs if Path(d).exists()]
        es = ingest_entities(state, docs, top_tokens=60, min_sentences=4, logger=log)
    log.log(f"entities: {es}")

    with log.phase("save"):
        save_checkpoint(state, _ROOT / args.out, cache)
    log.log(f"✓ SAVED {args.out}")


if __name__ == "__main__":
    main()
