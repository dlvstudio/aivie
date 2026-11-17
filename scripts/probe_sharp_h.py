#!/usr/bin/env python3
"""P5 entropy + P6 h channel probe (no_agenda, matrix)."""

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
from nn_aivie.hca.generate_hebbian import logits_topk_entropy  # noqa: E402
from nn_aivie.hca.matrix_core import matrix_core_logits  # noqa: E402
from nn_aivie.hca.sent_intra import clear_sent_intra_cache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_sent.pkl")
    ap.add_argument("--log-file", default="out/probe_sharp_h.log")
    ap.add_argument(
        "--prefix",
        default="Ngô Quyền là vị vua sáng lập nhà Ngô , được xem là người mở đầu thời kỳ độc lập tự chủ",
    )
    args = ap.parse_args()
    logf = open(_ROOT / args.log_file, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")

    state, _ = load_checkpoint(args.checkpoint)
    ids = list(state.vocab.encode_prompt(args.prefix))
    ids = [int(t) for t in ids if 0 <= int(t) < len(state.vocab)]
    clear_sent_intra_cache()
    variants = {
        "full": HCAConfig(
            ar_matrix_core=True,
            ar_no_agenda=True,
            ar_matrix_stack_weight=0.0,
            ar_sent_agenda_weight=0.0,
        ),
        "no_h": HCAConfig(
            ar_matrix_core=True,
            ar_no_agenda=True,
            ar_matrix_stack_weight=0.0,
            ar_sent_agenda_weight=0.0,
            ar_matrix_h_weight=0.0,
        ),
        "no_qk_l2r": HCAConfig(
            ar_matrix_core=True,
            ar_no_agenda=True,
            ar_matrix_stack_weight=0.0,
            ar_sent_agenda_weight=0.0,
            ar_matrix_qk_from_l2r=False,
        ),
    }
    eos = state.vocab.id_of(SPECIAL_EOS)
    log(f"prefix last={state.vocab.token_of(ids[-1])!r} tokens={len(ids)}")
    sidecar = Path(args.checkpoint).with_suffix(".sent_qk.npz")
    log(f"sent_qk sidecar exists={sidecar.is_file()} path={sidecar}")
    for name, cfg in variants.items():
        clear_sent_intra_cache()
        logits = matrix_core_logits(state, ids, config=cfg)
        H = logits_topk_entropy(logits, 12)
        top = np.argsort(-logits)[:8]
        log(f"\n=== {name}  H@12={H:.3f} EOS={float(logits[eos]):.3f} ===")
        for i in top:
            log(f"  {float(logits[i]):8.3f}  {state.vocab.token_of(int(i))!r}")
    log(f"\n✓ {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
