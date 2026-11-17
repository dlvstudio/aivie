#!/usr/bin/env python3
"""
TK_045 — probe cạnh trong câu L2R/R2L + W_agenda.

    PYTHONPATH=src python -u scripts/probe_sent_lr_edges.py \
        --checkpoint checkpoints/network_v11_styled_hist.pkl \
        --log-file out/probe_sent_lr_edges.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.sent_intra import (  # noqa: E402
    clear_sent_intra_cache,
    ensure_sent_intra,
    r2l_anchor_scores,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", type=str, default="out/probe_sent_lr_edges.log")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    ckpt = Path(args.checkpoint)
    log(f"load {ckpt}")
    state, _ = load_checkpoint(ckpt)
    cfg = HCAConfig()
    clear_sent_intra_cache()
    views = ensure_sent_intra(state, config=cfg, force=True)
    log(
        f"spans={views.n_sent_spans} L2R_pairs={views.n_l2r_pairs} "
        f"R2L_pairs={views.n_r2l_pairs} agenda={views.n_agenda_edges} "
        f"nnz_l2r={views.W_l2r.nnz} nnz_r2l={views.W_r2l.nnz} nnz_agenda={views.W_agenda.nnz}"
    )

    def top_hubs(mat, name: str) -> None:
        row_sum = np.asarray(mat.sum(axis=1)).ravel()
        idx = np.argsort(-row_sum)[: args.top]
        log(f"\n=== top hub rows {name} ===")
        for i in idx:
            if row_sum[i] <= 0:
                break
            tok = state.vocab.token_of(int(i))
            log(f"  {row_sum[i]:10.2f}  [{int(i)}] {tok!r}")

    top_hubs(views.W_l2r, "L2R")
    top_hubs(views.W_r2l, "R2L")
    top_hubs(views.W_agenda, "AGENDA")

    # Ví dụ neo R2L nếu có token trong vocab
    for probe in ("chủ", "Lan", "Quyền"):
        tid = state.vocab._token_to_id.get(probe)  # noqa: SLF001
        if tid is None:
            continue
        anchors = r2l_anchor_scores(
            views, [int(tid)], vocab_size=len(state.vocab), M=1
        )
        top = np.argsort(-anchors)[:8]
        log(f"\nR2L anchors from {probe!r}:")
        for j in top:
            if anchors[j] <= 0:
                break
            log(f"  {anchors[j]:8.3f}  {state.vocab.token_of(int(j))!r}")

    log(f"\n✓ wrote {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
