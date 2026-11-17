#!/usr/bin/env python3
"""One-step: matrix logits vs scores after gates in _wseq_fallback."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import _wseq_fallback
from nn_aivie.hca.matrix_core import matrix_core_logits
from nn_aivie.hca.readout import decode_surface

ckpt = _ROOT / "checkpoints/network_v11_styled_hist.pkl"
gc.disable(); state,_=load_checkpoint(str(ckpt)); gc.enable()
state._spectral_stack_sidecar = str(ckpt.with_suffix(".spectral_stack.npz"))
prompt = "Ngô Quyền là vị vua sáng lập nhà Ngô, được"
ids = state.vocab.encode_prompt(prompt)
eos = state.vocab.id_of("<EOS>") if hasattr(state.vocab, "id_of") else 0
from nn_aivie.core.vocab import SPECIAL_EOS
eos = state.vocab._token_to_id.get(SPECIAL_EOS, 0)
cfg = HCAConfig(ar_matrix_core=True, ar_no_agenda=True, hybrid_temperature=0.0)
logits = matrix_core_logits(state, ids, config=cfg)
print("argmax logits", state.vocab.token_of(int(np.argmax(logits))))
rng = np.random.default_rng(0)
out = _wseq_fallback(state, ids, eos, 1, cfg, rng)
print("after 1 step", decode_surface(state, out)[len(prompt):][:40])
print("new token", state.vocab.token_of(out[-1]))
