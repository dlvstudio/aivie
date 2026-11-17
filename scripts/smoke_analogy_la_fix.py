#!/usr/bin/env python
"""Smoke sau siết name-bridge / pred title / lop."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.analogy_bridge_gen import (  # noqa: E402
    AnalogyBridgeConfig,
    build_analogy_bridge_index,
    generate_analogy_bridge,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402

PROMPTS = [
    "Việt Nam là",
    "học sinh là",
    "anh là",
    "em là",
    "mẹ là",
    "ông là",
    "Hà Nội là",
    "Hồng là",
]


def run(ckpt: str) -> None:
    print(f"\n===== {ckpt} =====", flush=True)
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / ckpt))
    gc.enable()
    cfg = AnalogyBridgeConfig()
    idx = build_analogy_bridge_index(state, cfg)
    print(
        f"L1={idx.meta['n_l1']} L2={idx.meta['n_l2']} triples={idx.meta['n_triples']}",
        flush=True,
    )
    for p in PROMPTS:
        cont, meta = generate_analogy_bridge(state, p, index=idx, cfg=cfg)
        full = list(state.vocab.encode_prompt(p)) + list(cont)
        out = decode_surface(state, full).strip()
        if meta.get("stop") != "analogy_la":
            print(f"{p:<16} {out:<28} — {meta.get('stop')}")
        else:
            br = ",".join((meta.get("bridges") or [])[:4])
            print(
                f"{p:<16} {out:<28} head={meta.get('head_surf')!r} "
                f"| {br} || {','.join((meta.get('donors') or [])[:4])}"
            )


def main() -> None:
    ckpts = sys.argv[1:] or [
        "checkpoints/mini_sent_ep_qa.pkl",
        "checkpoints/network_v12_subst_isg_pw.pkl",
    ]
    for c in ckpts:
        run(c)


if __name__ == "__main__":
    main()
