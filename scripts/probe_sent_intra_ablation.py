#!/usr/bin/env python3
"""
TK_045 P7.4 — ablation R2L / L2R mask / agenda trên prefix matrix.

    PYTHONPATH=src python -u scripts/probe_sent_intra_ablation.py \
        --checkpoint checkpoints/network_v11_styled_hist.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.vocab import SPECIAL_EOS  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.matrix_core import matrix_core_logits  # noqa: E402
from nn_aivie.hca.sent_intra import clear_sent_intra_cache  # noqa: E402


def _top(logits, vocab, k=8):
    idx = np.argsort(-logits)[:k]
    return [(vocab.token_of(int(i)), float(logits[i])) for i in idx if np.isfinite(logits[i])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", default="out/probe_sent_intra_ablation.log")
    ap.add_argument(
        "--prefix",
        default="Ngô Quyền là vị vua sáng lập nhà Ngô , được xem là người mở đầu thời kỳ độc lập tự chủ",
    )
    args = ap.parse_args()
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")

    state, _ = load_checkpoint(args.checkpoint)
    # encode_prompt / encode_tokens — Vocabulary không có .encode()
    if hasattr(state.vocab, "encode_prompt"):
        ids = list(state.vocab.encode_prompt(args.prefix))
    else:
        ids = state.vocab.encode_tokens(
            args.prefix.replace(",", " , ").replace(".", " . ").split()
        )
    ids = [int(t) for t in ids if 0 <= int(t) < len(state.vocab)]

    variants = {
        "full": HCAConfig(ar_matrix_qk_weight=0.0, ar_matrix_stack_weight=0.0, ar_matrix_know_weight=0.5),
        "no_l2r": HCAConfig(
            ar_sent_l2r_weight=0.0,
            ar_matrix_qk_weight=0.0,
            ar_matrix_stack_weight=0.0,
            ar_matrix_know_weight=0.5,
        ),
        "no_mask": HCAConfig(
            ar_sent_l2r_parent_mask=False,
            ar_matrix_qk_weight=0.0,
            ar_matrix_stack_weight=0.0,
            ar_matrix_know_weight=0.5,
        ),
        "no_agenda": HCAConfig(
            ar_sent_agenda_weight=0.0,
            ar_matrix_qk_weight=0.0,
            ar_matrix_stack_weight=0.0,
            ar_matrix_know_weight=0.5,
        ),
        "no_intra": HCAConfig(
            ar_sent_intra=False,
            ar_matrix_qk_weight=0.0,
            ar_matrix_stack_weight=0.0,
            ar_matrix_know_weight=0.5,
        ),
    }
    eos = state.vocab.id_of(SPECIAL_EOS)
    log(f"prefix tokens={len(ids)} last={state.vocab.token_of(ids[-1])!r}")
    for name, cfg in variants.items():
        clear_sent_intra_cache()
        logits = matrix_core_logits(state, ids, config=cfg)
        top = _top(logits, state.vocab, 10)
        eos_s = float(logits[eos]) if eos is not None else float("nan")
        log(f"\n=== {name}  EOS={eos_s:.3f} ===")
        for tok, s in top:
            log(f"  {s:8.3f}  {tok!r}")
    log(f"\n✓ {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
