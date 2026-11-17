#!/usr/bin/env python3
"""Sweep ar_style_weight + boundary inject on humor prompt + multizone."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.hca.readout import decode_surface

MARKERS = ("khiếu", "biếm", "hóm", "cười", "phết", "hơi lố", "kiểu như", "châm", "mỉa", "hài", "hệt")


def main() -> int:
    ck = "checkpoints/network_v11_styled_1k.pkl"
    print(f"load {ck}", flush=True)
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / ck))
    gc.enable()
    prompt = "Viết một đoạn văn hài hước về Ngô Quyền"
    log_path = _ROOT / "out/probe_style_boundary.log"
    lines = []
    configs = [
        ("MZ only", dict(ar_style_weight=0.0, ar_style_boundary_inject=False)),
        ("Hadamard 1.5", dict(ar_style_weight=1.5, ar_style_boundary_inject=False)),
        ("Boundary inj", dict(ar_style_weight=0.5, ar_style_boundary_inject=True,
                              ar_style_boundary_boost=25.0, ar_style_boundary_max=3)),
        ("Both", dict(ar_style_weight=1.5, ar_style_boundary_inject=True,
                      ar_style_boundary_boost=25.0, ar_style_boundary_max=3)),
    ]
    for name, kw in configs:
        cfg = HCAConfig(
            ar_verbatim_fallback=False,
            essay_ar_min_cov=0.0,
            ar_multizone=True,
            **kw,
        )
        ids = generate(state, prompt, max_len=80, config=cfg)
        text = decode_surface(state, ids)
        if text.startswith(prompt):
            text = text[len(prompt) :].lstrip()
        hits = [m for m in MARKERS if m in text.lower()]
        block = f"=== {name} hits={hits} ===\n{text[:360]}\n\n"
        print(block, flush=True)
        lines.append(block)
    log_path.write_text("".join(lines))
    print(f"→ {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
