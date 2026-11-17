#!/usr/bin/env python3
"""A/B: MZ / graph-inventory / corpus-inventory inject (styled_1k)."""
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
from nn_aivie.hca.style_inventory import style_inject_plan

MARKERS = (
    "khiếu", "biếm", "hóm", "cười", "phết", "hơi lố", "kiểu như",
    "châm", "mỉa", "hài", "hệt", "cốt", "kịch", "phim", "TikTok",
    "Gen Z", "deadline",
)


def main() -> int:
    ck = "checkpoints/network_v11_styled_1k.pkl"
    print(f"load {ck}", flush=True)
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / ck))
    gc.enable()
    prompt = "Viết một đoạn văn hài hước về Ngô Quyền"
    q = list(state.vocab.encode_prompt(prompt))
    for label, use_corp in (("graph", False), ("corpus", True)):
        plan = style_inject_plan(state, q, max_items=3, use_corpus=use_corp)
        print(
            f"plan_{label}:",
            [" ".join(state.vocab.token_of(t) for t in u) for u in plan],
            flush=True,
        )

    configs = [
        ("MZ only", dict()),
        ("Graph inv+phrase", dict(
            ar_style_boundary_inject=True, ar_style_use_inventory=True,
            ar_style_corpus_inventory=False, ar_style_prefer_phrase=True,
            ar_style_weight=0.5, ar_style_boundary_max=3,
        )),
        ("Corpus inv+phrase", dict(
            ar_style_boundary_inject=True, ar_style_use_inventory=True,
            ar_style_corpus_inventory=True, ar_style_prefer_phrase=True,
            ar_style_weight=0.5, ar_style_boundary_max=3,
        )),
        ("Corpus singles", dict(
            ar_style_boundary_inject=True, ar_style_use_inventory=True,
            ar_style_corpus_inventory=True, ar_style_prefer_phrase=False,
            ar_style_weight=0.5, ar_style_boundary_max=3,
        )),
    ]
    lines = []
    for name, kw in configs:
        print(f"… generate {name}", flush=True)
        cfg = HCAConfig(
            ar_verbatim_fallback=False,
            essay_ar_min_cov=0.0,
            ar_multizone=True,
            **kw,
        )
        ids = generate(state, prompt, max_len=60, config=cfg)
        text = decode_surface(state, ids)
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        hits = [m for m in MARKERS if m.lower() in text.lower()]
        block = f"=== {name} hits={hits} ===\n{text[:450]}\n\n"
        print(block, flush=True)
        lines.append(block)
    out = _ROOT / "out/probe_style_corpus_gen.log"
    out.write_text("".join(lines))
    print(f"→ {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
