#!/usr/bin/env python
"""Smoke analogy_la sau siết hub — mini + optional v12."""
from __future__ import annotations

import sys
import time
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
    "anh là", "mẹ là", "Thái là", "ông là", "bà là", "Trung là",
    "em là", "Hồng là", "Sông là", "con khỉ là", "chó là", "mèo là",
    "Hà Nội là", "Việt Nam là", "học sinh là",
]


def run(ckpt: str) -> None:
    t0 = time.time()
    print(f"\n===== {ckpt} =====", flush=True)
    import gc
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / ckpt))
    gc.enable()
    cfg = AnalogyBridgeConfig()
    idx = build_analogy_bridge_index(state, cfg)
    print(
        f"vocab={len(state.vocab):,} parents={len(state.parents):,} "
        f"L1={idx.meta['n_l1']:,} L2={idx.meta['n_l2']:,} "
        f"triples={idx.meta['n_triples']:,} "
        f"drop_prop_df={idx.meta.get('n_prop_keys_dropped_df')} "
        f"heads cl/ph/ent="
        f"{idx.meta.get('n_head_cluster')}/"
        f"{idx.meta.get('n_head_phrase')}/"
        f"{idx.meta.get('n_head_entity')} "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )
    print(f"{'prompt':<16} {'output':<28} cầu / donors")
    print("-" * 80)
    for p in PROMPTS:
        cont, meta = generate_analogy_bridge(state, p, index=idx, cfg=cfg)
        full = list(state.vocab.encode_prompt(p)) + list(cont)
        out = decode_surface(state, full).strip()
        if meta.get("stop") != "analogy_la":
            print(f"{p:<16} {out:<28} — {meta.get('stop')}")
        else:
            br = ",".join((meta.get("bridges") or [])[:4])
            dons = ",".join((meta.get("donors") or [])[:6])
            print(f"{p:<16} {out:<28} {br} | {dons}  n={meta['n_bridges']}")


def main() -> None:
    ckpts = sys.argv[1:] or ["checkpoints/mini_sent_ep_qa.pkl"]
    for c in ckpts:
        run(c)


if __name__ == "__main__":
    main()
