#!/usr/bin/env python
"""Demo analogy_la trên network_v12_subst_isg_pw — prompt có ý nghĩa."""
from __future__ import annotations

import gc
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

CKPT = "checkpoints/network_v12_subst_isg_pw.pkl"
PROMPTS = [
    "anh là",
    "em là",
    "mẹ là",
    "cha là",
    "ông là",
    "bà là",
    "học sinh là",
    "Hà Nội là",
    "Việt Nam là",
    "Hồng là",
    "Sông Hồng là",
    "núi là",
    "con chó là",
    "con mèo là",
    "chó là",
    "mèo là",
    "vua là",
    "thầy là",
    "bác sĩ là",
    "Thủ đô là",
    "người Việt là",
]


def main() -> None:
    t0 = time.time()
    print(f"load {CKPT} …", flush=True)
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / CKPT))
    gc.enable()
    cfg = AnalogyBridgeConfig()
    print("build index …", flush=True)
    idx = build_analogy_bridge_index(state, cfg)
    print(
        f"ready ({time.time() - t0:.0f}s) "
        f"L1={idx.meta['n_l1']} L2={idx.meta['n_l2']} "
        f"triples={idx.meta['n_triples']}",
        flush=True,
    )
    print()
    print(f"{'prompt':<18} {'output':<36} cầu / donors")
    print("-" * 100)
    for p in PROMPTS:
        cont, meta = generate_analogy_bridge(state, p, index=idx, cfg=cfg)
        full = list(state.vocab.encode_prompt(p)) + list(cont)
        out = decode_surface(state, full).strip()
        if meta.get("stop") != "analogy_la":
            print(f"{p:<18} {out:<36} — {meta.get('stop')}")
        else:
            br = ",".join((meta.get("bridges") or [])[:5])
            dons = ",".join((meta.get("donors") or [])[:5])
            hs = meta.get("head_surf", "")
            print(
                f"{p:<18} {out:<36} head={hs!r} | {br} || {dons} "
                f"n={meta['n_bridges']}"
            )


if __name__ == "__main__":
    main()
