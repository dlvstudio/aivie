#!/usr/bin/env python3
"""
TK_045 P0.1 — migrate mega-parent → 1 câu/parent + episodes.

    PYTHONPATH=src python -u scripts/migrate_sentence_parents.py \
        --checkpoint checkpoints/network_v11_styled_hist.pkl \
        --out checkpoints/network_v11_sent.pkl

    # thử nhanh, không ghi:
    ... --dry-run --max-source-parents 2000
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402
from nn_aivie.core.parent_store import ParentStore  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.sentence_parents import migrate_parents_to_sentences  # noqa: E402
from nn_aivie.hca.sent_intra import clear_sent_intra_cache, ensure_sent_intra  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--out", default="checkpoints/network_v11_sent.pkl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-mega-gt", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-spans-per-source", type=int, default=24)
    ap.add_argument("--max-new-parents", type=int, default=200_000)
    ap.add_argument(
        "--max-source-parents",
        type=int,
        default=0,
        help="0=all; >0 chỉ N cha đầu (thử nhanh)",
    )
    args = ap.parse_args()

    log = get_run_logger("migrate_sentence_parents")
    state, cache = load_checkpoint(args.checkpoint, logger=log)
    if args.max_source_parents > 0 and state.parents is not None:
        n = min(args.max_source_parents, len(state.parents.parents))
        log.log(f"subset first {n} parents for migrate")
        trimmed = ParentStore()
        trimmed.parents = list(state.parents.parents[:n])
        for i, rec in enumerate(trimmed.parents):
            key = rec.source_ids or (
                rec.answer_ids if rec.is_qa and rec.answer_ids else rec.emit_ids
            )
            trimmed._key_to_idx[tuple(key)] = i  # noqa: SLF001
            for t in set(rec.match_ids):
                trimmed._gdf[t] = trimmed._gdf.get(t, 0) + 1
                trimmed._df[t] = trimmed._df.get(t, 0) + 1
        trimmed.rebuild_hierarchical_index()
        state.parents = trimmed

    stats = migrate_parents_to_sentences(
        state,
        only_mega_gt=args.only_mega_gt,
        max_tokens=args.max_tokens,
        max_spans_per_source=args.max_spans_per_source,
        max_new_parents=args.max_new_parents,
        logger=log,
    )
    clear_sent_intra_cache()
    views = ensure_sent_intra(state, config=HCAConfig(), force=True)
    log.log(
        f"after: parents={stats.new_parents} episodes={stats.episodes} "
        f"L2R_nnz={views.W_l2r.nnz} agenda_nnz={views.W_agenda.nnz} "
        f"agenda_edges={views.n_agenda_edges}"
    )
    if args.dry_run:
        log.log("dry-run — không save")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.log(f"save → {out}")
    save_checkpoint(state, out, cache=cache, logger=log)
    gc.collect()
    log.log("✓ done")


if __name__ == "__main__":
    main()
