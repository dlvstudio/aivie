#!/usr/bin/env python3
"""
TK_035 — POOL ĐOẠN 2–4 câu (giữ , . ! ?) cho continuation / pattern-completion.

Nguyên tắc (giống R3b two-tier):
  - Đơn vị = 2–4 câu liên tiếp trong cùng đoạn (wiki/lịch sử) hoặc cửa sổ trên văn nói.
  - Parent priority=0.36 (band 0.355–0.395) — KHÔNG cạnh tranh QA; chỉ đọc ở discourse-route.
  - KHÔNG _g1_and_wseq tại build (giữ ruler/base-LM).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.data.paragraph_units import (
    PARAGRAPH_POOL_PRIORITY,
    iter_units_from_conv_jsonl,
    iter_units_from_doc_jsonl,
)
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DOC_SOURCES = [
    "data/real_vi/statements_doc.jsonl",
    "data/real_vi/statements_doc_add.jsonl",
    "data/real_vi/history_wiki_para.jsonl",
    "data/real_vi/history_vi.jsonl",
]

DEFAULT_CONV_SOURCES = [
    "data/conv/stories_raw.jsonl",
    "data/conv/chat_raw.jsonl",
]


def _collect_units(
    doc_paths: list[Path],
    conv_paths: list[Path],
    *,
    min_sent: int,
    max_sent: int,
    max_units: int,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in doc_paths:
        for unit in iter_units_from_doc_jsonl(p, min_sent=min_sent, max_sent=max_sent, dedup=False):
            key = unit.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(unit)
            if len(out) >= max_units:
                return out
    for p in conv_paths:
        for unit in iter_units_from_conv_jsonl(p, min_sent=min_sent, max_sent=max_sent, dedup=False):
            key = unit.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(unit)
            if len(out) >= max_units:
                return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v9_histwiki.pkl")
    ap.add_argument("--out", default="checkpoints/network_v9_histwiki_paragraph.pkl")
    ap.add_argument("--max-units", type=int, default=25000)
    ap.add_argument("--min-sent", type=int, default=2)
    ap.add_argument("--max-sent", type=int, default=4)
    ap.add_argument("--doc", action="append", default=None, help="jsonl doc (repeatable)")
    ap.add_argument("--conv", action="append", default=None, help="jsonl conv (repeatable)")
    args = ap.parse_args()
    log = get_run_logger("build_paragraph_pool")

    doc_paths = [(_ROOT / d) for d in (args.doc or DEFAULT_DOC_SOURCES)]
    conv_paths = [(_ROOT / c) for c in (args.conv or DEFAULT_CONV_SOURCES)]
    doc_paths = [p for p in doc_paths if p.exists()]
    conv_paths = [p for p in conv_paths if p.exists()]

    state, cache = load_checkpoint(args.checkpoint, logger=log)
    ps, vocab = state.parents, state.vocab
    v0, p0 = len(vocab), len(ps.parents)

    with log.phase("thu đơn vị đoạn 2–4 câu"):
        units = _collect_units(
            doc_paths, conv_paths,
            min_sent=args.min_sent, max_sent=args.max_sent,
            max_units=args.max_units,
        )
        log.log(f"đơn vị unique: {len(units)} (doc files={len(doc_paths)}, conv={len(conv_paths)})")

    known = set(vocab._token_to_id.keys())
    picked: list[list[int]] = []
    skipped_oov = 0
    with log.phase("encode + lọc OOV"):
        for unit in units:
            toks = tokenize_punct(unit)
            if not toks or not all(t in known for t in toks):
                skipped_oov += 1
                continue
            ids = vocab.encode_tokens(toks)
            if len(ids) < 8:
                continue
            picked.append(ids)
        log.log(f"nạp được: {len(picked)} (bỏ OOV/thưa: {skipped_oov})")

    with log.phase(f"nạp parents priority={PARAGRAPH_POOL_PRIORITY}"):
        ps.seg.begin_batch()
        for k, ids in enumerate(picked):
            ps.add_example(ids, vocab, dap_id=None, priority=PARAGRAPH_POOL_PRIORITY, defer_index=True)
            if (k + 1) % 2000 == 0:
                log.progress("paragraph_pool", k + 1, len(picked))
        ps.seg.end_batch(ps.parents)
        ps.rebuild_hierarchical_index()
        ps.flush_skip_ledger(vocab, logger=log)
    log.log(
        f"parents {p0} → {len(ps.parents)} (+{len(ps.parents) - p0}); "
        f"vocab {v0} → {len(vocab)}"
    )

    with log.phase("save"):
        save_checkpoint(state, args.out, cache, bake_derived=True, logger=log)
    log.log(f"XONG → {args.out}")


if __name__ == "__main__":
    main()
