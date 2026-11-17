#!/usr/bin/env python3
"""
TK_045 P0.0 — thống kê độ dài emit / #span câu trên checkpoint (dry-run, không retrain).

    PYTHONPATH=src python -u scripts/probe_sentence_parents.py \
        --checkpoint checkpoints/network_v11_styled_hist.pkl \
        --log-file out/probe_sentence_parents.log
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.sent_intra import iter_sentence_spans  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", default="out/probe_sentence_parents.log")
    ap.add_argument("--mega-n", type=int, default=50, help="ngưỡng token = mega-parent")
    ap.add_argument("--sample", type=int, default=8)
    args = ap.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        logf.write(str(msg) + "\n")

    state, _ = load_checkpoint(args.checkpoint)
    ps = state.parents
    assert ps is not None
    lens: list[int] = []
    n_span: list[int] = []
    mega = 0
    for rec in ps.parents:
        seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        lens.append(len(seq))
        spans = iter_sentence_spans(seq, state.vocab)
        n_span.append(len(spans))
        if len(seq) > args.mega_n:
            mega += 1

    arr = np.asarray(lens, dtype=np.float64)
    sp = np.asarray(n_span, dtype=np.float64)
    log(f"parents={len(ps.parents)} episodes={len(getattr(ps, 'episodes', []) or [])}")
    log(
        f"emit_len mean={arr.mean():.1f} p50={np.median(arr):.0f} "
        f"p90={np.percentile(arr, 90):.0f} max={arr.max():.0f}"
    )
    log(
        f"spans/parent mean={sp.mean():.2f} p50={np.median(sp):.0f} "
        f"max={sp.max():.0f}"
    )
    log(f"mega emit>{args.mega_n}: {mega} ({100.0 * mega / max(len(lens), 1):.1f}%)")
    log(f"span hist: {Counter(int(x) for x in sp.tolist()).most_common(12)}")

    # sample mega
    shown = 0
    for pi, rec in enumerate(ps.parents):
        if shown >= args.sample:
            break
        seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        if len(seq) <= args.mega_n:
            continue
        toks = [state.vocab.token_of(t) for t in seq[:24]]
        log(f"\nmega[{pi}] len={len(seq)} spans={len(iter_sentence_spans(seq, state.vocab))}")
        log("  " + " ".join(toks) + (" …" if len(seq) > 24 else ""))
        shown += 1

    log(f"\n✓ {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
