#!/usr/bin/env python
"""TK_046 P98G ship — CLI điền ``X là ?`` (cầu cụm L=1..2).

Xem RUNBOOK_ANALOGY_P98G_SHIP.md.

  PYTHONPATH=src .venv/bin/python scripts/gen_analogy_la.py \\
    --checkpoint checkpoints/mini_sent_ep_qa.pkl \\
    --prompt "anh là"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.analogy_bridge_gen import (  # noqa: E402
    AnalogyBridgeConfig,
    generate_analogy_bridge,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Sinh analogy_la (P98G defaults)")
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--prop-gidf-min", type=float, default=4.5)
    ap.add_argument("--compound-bonus", type=float, default=2.5)
    ap.add_argument("--min-strength", type=float, default=1.0)
    args = ap.parse_args()

    logger = get_run_logger("gen_analogy_la")
    cfg = AnalogyBridgeConfig(
        min_support=args.min_support,
        prop_gidf_min=args.prop_gidf_min,
        compound_bonus=args.compound_bonus,
        min_strength=args.min_strength,
        creative_only=False,
    )

    t0 = time.time()
    import gc
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    logger.log(f"loaded ({time.time()-t0:.0f}s)")

    cont, meta = generate_analogy_bridge(state, args.prompt, cfg=cfg)
    prompt_ids = list(state.vocab.encode_prompt(args.prompt))
    full = prompt_ids + list(cont)
    print(decode_surface(state, full).strip())
    print("# meta:", json.dumps(meta, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
