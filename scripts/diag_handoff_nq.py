#!/usr/bin/env python3
"""Diagnose episode handoff span after 1-sentence matrix gen."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.sent_intra import agenda_handoff_span


def main() -> None:
    state, _ = load_checkpoint("checkpoints/network_v12_sent.pkl")
    cfg1 = HCAConfig(
        ar_matrix_core=True,
        ar_no_agenda=True,
        ar_sent_agenda_handoff=False,
        ar_style_payoff=False,
        hybrid_temperature=0.0,
        ar_matrix_stack_weight=0.0,
    )
    cfg_hand = HCAConfig(
        ar_matrix_core=True,
        ar_no_agenda=True,
        ar_sent_agenda_handoff=True,
        ar_style_payoff=False,
        hybrid_temperature=0.0,
        ar_matrix_stack_weight=0.0,
    )
    prompt = "Ngô Quyền là vị vua sáng lập nhà Ngô, được"
    out1 = generate(state, prompt, max_len=40, config=cfg1, seed=42)
    print("1sent:", decode_surface(state, out1))
    span = agenda_handoff_span(state, out1, config=cfg_hand, eos_logit=5.0)
    print("span:", None if not span else decode_surface(state, span))
    out2 = generate(state, prompt, max_len=48, config=cfg_hand, seed=42)
    print("multi:", decode_surface(state, out2))


if __name__ == "__main__":
    main()
