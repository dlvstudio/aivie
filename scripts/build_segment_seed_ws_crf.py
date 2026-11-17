#!/usr/bin/env python3
"""
TK_034 — seed_pairs from a WS-CRF model (nn-aivie self-trained) and build clusters into checkpoint.

This is build-time only. It runs a CRF word segmentation model (B-W/I-W) on raw corpora, extracts
multi-syllable words (space-joined tokens), converts adjacent syllable pairs to `seed_pairs`, then
uses TK_033 seed_pairs mechanism to promote clusters.

Requires: `python-crfsuite` available (`pip install -e .[ws]`).

Example:
  python scripts/build_segment_seed_ws_crf.py \
    --checkpoint checkpoints/network_v7_history.pkl \
    --out checkpoints/network_v7_history_wsseed.pkl \
    --model-dir /path/to/ws_crf_model_dir \
    --corpus data/real_vi/history_vi.jsonl \
    --corpus data/real_vi/statements.jsonl \
    --corpus data/real_vi/statements_doc.jsonl \
    --corpus data/real_vi/statements_doc_add.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.util.progress import FileProgressLog
from nn_aivie.ws_crf.ws_model import WSCRFModel

ROOT = Path(__file__).resolve().parents[1]


def _iter_texts(jsonl_path: Path) -> list[str]:
    out: list[str] = []
    if not jsonl_path.exists():
        return out
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = (obj.get("text") or obj.get("statement") or "").strip()
            if text:
                out.append(text)
    return out


def _pairs_from_ws_output(words: list[str]) -> Counter[tuple[str, str]]:
    """Count adjacent syllable pairs from WS output words (space-joined tokens)."""
    pairs: Counter[tuple[str, str]] = Counter()
    for w in words:
        if " " not in w:
            continue
        syls = [s for s in w.split(" ") if s]
        for a, b in zip(syls, syls[1:]):
            pairs[(a, b)] += 1
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="TK_034: seed clusters from WS-CRF predictions")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", required=True, help="Directory containing model.crfsuite/dictionary_lower.txt/meta.json")
    ap.add_argument("--corpus", action="append", default=[], help="JSONL with field `text` (repeatable)")
    ap.add_argument("--min-pair-count", type=int, default=2, help="Only keep predicted pairs seen >=N times")
    ap.add_argument("--max-promotions", type=int, default=0, help="0=no limit; cap promotions to avoid explosion")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    log_path = Path(args.log_file) if args.log_file else (ROOT / "out" / "tk034_wsseed.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flog = FileProgressLog(log_path)

    ckpt = ROOT / args.checkpoint
    out = ROOT / args.out
    model_dir = Path(args.model_dir)
    flog.log(f"START ckpt={ckpt.name} out={out.name} model_dir={model_dir}")

    state, cache = load_checkpoint(str(ckpt), logger=flog)
    ps = state.parents
    if ps is None:
        flog.log("ERROR: checkpoint missing parents")
        return 3
    if ps.seg is None:
        flog.log("ERROR: checkpoint missing seg graph (unexpected)")
        return 4

    # Load nn-aivie WS-CRF artifacts
    model = WSCRFModel.load(model_dir)
    flog.log("✓ loaded WS-CRF model")

    corpora = [ROOT / p for p in (args.corpus or [])]
    if not corpora:
        flog.log("ERROR: need at least one --corpus jsonl")
        return 5

    pair_counts: Counter[tuple[str, str]] = Counter()
    n_sent = 0
    for cp in corpora:
        texts = _iter_texts(cp)
        flog.log(f"corpus={cp} sentences={len(texts)}")
        for s in texts:
            toks = s.strip().split()
            if not toks:
                continue
            tags = model.tag(toks)
            out_words = model.merge_iw(toks, tags)
            pair_counts.update(_pairs_from_ws_output(out_words))
            n_sent += 1
            if n_sent and n_sent % 20000 == 0:
                flog.log(f"  processed {n_sent} sentences; pairs={len(pair_counts)}")

    min_pair = int(args.min_pair_count)
    kept = {p for p, c in pair_counts.items() if c >= min_pair}
    flog.log(f"predicted_pairs total={len(pair_counts)} kept(min={min_pair})={len(kept)}")

    # Map to vocab ids; only keep pairs fully in vocab.
    tid = state.vocab._token_to_id
    seed_pairs: set[tuple[int, int]] = set()
    oov = 0
    for a, b in kept:
        ai = tid.get(a)
        if ai is None:
            ai = tid.get(a.lower())
        bi = tid.get(b)
        if bi is None:
            bi = tid.get(b.lower())
        if ai is None or bi is None:
            oov += 1
            continue
        seed_pairs.add((int(ai), int(bi)))
    flog.log(f"seed_pairs(in-vocab)={len(seed_pairs)} oov_pairs={oov}")

    before = len(ps.seg.seed_pairs)
    ps.seg.seed_pairs |= seed_pairs
    flog.log(f"seg.seed_pairs before={before} add={len(seed_pairs)} after={len(ps.seg.seed_pairs)}")

    n_promoted = ps.seg.finalize_promotions(ps.parents, progress=flog.log, max_promotions=int(args.max_promotions))
    flog.log(f"promoted={n_promoted} clusters={len(ps.seg.clusters)}")

    flog.log(f"saving → {out}")
    save_checkpoint(state, str(out), cache)
    flog.log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

