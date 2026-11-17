#!/usr/bin/env python3
"""
TK_033 S1 — build dictionary-seed cluster (seed_pairs) into an existing checkpoint.

Luật 0: chỉ build-time. Runtime không gọi NLP; chỉ dùng slot_chain/links đã build.

Usage:
  python scripts/build_segment_seed.py \
    --checkpoint checkpoints/network_v7_history.pkl \
    --out checkpoints/network_v7_history_segseed.pkl \
    --dict-raw data/dict/vi_dict_raw.jsonl \
    --extra-titles data/real_vi/history_vi.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.util.progress import FileProgressLog

ROOT = Path(__file__).resolve().parents[1]


def _encode_existing(vocab, text: str) -> list[int]:
    """Encode text into existing vocab ids only; break on OOV to avoid chaining across unknown."""
    ids: list[int] = []
    tid = vocab._token_to_id
    for tok in tokenize_punct(text):
        i = tid.get(tok)
        if i is None:
            # `Vocabulary.fold()` nhận token_id, không nhận string. Với string, dùng lowercase
            # (case-fold thô) để bắt title-cased token ("Napoléon"→"napoléon") nếu có.
            i = tid.get(tok.lower())
        if i is None:
            # treat as boundary
            ids.append(-1)
        else:
            ids.append(int(i))
    return ids


def _pairs_from_ids(ids: list[int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    run: list[int] = []
    for i in ids + [-1]:
        if i is None or i < 0:
            if len(run) >= 2:
                for a, b in zip(run, run[1:]):
                    out.add((a, b))
            run = []
            continue
        run.append(i)
    return out


def _seed_from_dict_raw(path: Path, vocab, *, log_every: int = 5000) -> tuple[set[tuple[int, int]], dict[str, int]]:
    pairs: set[tuple[int, int]] = set()
    stats = {"rows": 0, "words_multi": 0, "oov_words": 0}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["rows"] += 1
            try:
                w = json.loads(line).get("word", "")
            except Exception:
                continue
            w = str(w).strip()
            if not w or " " not in w:
                continue
            stats["words_multi"] += 1
            ids = _encode_existing(vocab, w)
            if all(i < 0 for i in ids):
                stats["oov_words"] += 1
                continue
            pairs |= _pairs_from_ids(ids)
            if log_every and stats["rows"] % log_every == 0:
                pass
    return pairs, stats


def _seed_from_titles_jsonl(paths: list[Path], vocab) -> tuple[set[tuple[int, int]], dict[str, int]]:
    pairs: set[tuple[int, int]] = set()
    stats = {"rows": 0, "titles": 0, "titles_multi": 0}
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                stats["rows"] += 1
                try:
                    title = json.loads(line).get("title", "")
                except Exception:
                    continue
                title = str(title).strip()
                if not title:
                    continue
                stats["titles"] += 1
                if " " not in title:
                    continue
                stats["titles_multi"] += 1
                ids = _encode_existing(vocab, title)
                pairs |= _pairs_from_ids(ids)
    return pairs, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="TK_033 S1: build dictionary-seed segment clusters")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dict-raw", default="data/dict/vi_dict_raw.jsonl")
    ap.add_argument("--extra-titles", action="append", default=[], help="jsonl path(s) with title field")
    ap.add_argument("--max-promotions", type=int, default=0, help="0=no limit; cap promotions to avoid explosion")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    ts = None
    if args.log_file:
        log_path = ROOT / args.log_file if not str(args.log_file).startswith("/") else Path(args.log_file)
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = ROOT / "out" / f"tk033_build_seed_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flog = FileProgressLog(log_path)

    ckpt = ROOT / args.checkpoint
    out = ROOT / args.out
    flog.log(f"START ckpt={ckpt.name} out={out.name}")

    state, cache = load_checkpoint(str(ckpt), logger=flog)
    ps = state.parents
    if ps is None:
        flog.log("ERROR: checkpoint missing parents")
        return 2
    if ps.seg is None or not ps.seg.bigram_count:
        # safety: ensure seg exists; rebuild uses existing source_ids
        dap_id = state.vocab.id_of(DAP_LABEL)
        flog.log("seg empty → rebuild_seg_graph first")
        ps.rebuild_seg_graph(state.vocab, dap_id=dap_id, progress=flog.log, max_promotions=0)

    dict_path = ROOT / args.dict_raw
    if not dict_path.exists():
        flog.log(f"ERROR: dict-raw missing: {dict_path}")
        return 3

    p_dict, st_dict = _seed_from_dict_raw(dict_path, state.vocab)
    flog.log(f"dict_raw: {st_dict} seed_pairs={len(p_dict)}")

    p_titles: set[tuple[int, int]] = set()
    st_titles = {}
    if args.extra_titles:
        extra = [ROOT / p for p in args.extra_titles]
        p_titles, st_titles = _seed_from_titles_jsonl(extra, state.vocab)
        flog.log(f"titles: {st_titles} seed_pairs={len(p_titles)}")

    seed_pairs = p_dict | p_titles
    before = len(ps.seg.seed_pairs)
    ps.seg.seed_pairs |= seed_pairs
    flog.log(f"seed_pairs: before={before} add={len(seed_pairs)} after={len(ps.seg.seed_pairs)}")

    # Promote: re-use existing ISG heap; seed pairs without count will be pushed to end.
    n_promoted = ps.seg.finalize_promotions(ps.parents, progress=flog.log, max_promotions=args.max_promotions)
    flog.log(f"promoted={n_promoted} clusters={len(ps.seg.clusters)}")

    flog.log(f"saving → {out}")
    save_checkpoint(state, str(out), cache)
    flog.log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

