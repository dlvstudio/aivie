#!/usr/bin/env python3
"""
PROBE style inventory — boundary fitness + phrases (cho M4).

    PYTHONPATH=src python -u scripts/probe_style_inventory.py \
        --checkpoint checkpoints/network_v11_styled_1k.pkl \
        --log-file out/probe_style_inventory.log
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.style_inventory import (  # noqa: E402
    boundary_fitness,
    ranked_style_inventory,
    ranked_style_phrases,
    style_inject_plan,
)
from nn_aivie.hca.style_readout import style_boost_from_coact  # noqa: E402

PROMPTS = [
    "Viết một đoạn văn hài hước về Ngô Quyền",
    "Kể buồn về mùa thu Hà Nội",
    "So sánh Nhà Lý và Nhà Trần",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_1k.pkl")
    ap.add_argument("--log-file", default="out/probe_style_inventory.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    vocab = state.vocab
    log(f"[{time.time()-t0:.0f}s] ✓ parents {len(state.parents):,}\n")

    for prompt in PROMPTS:
        q = list(vocab.encode_prompt(prompt))
        log(f"═══ {prompt!r} ═══")
        raw, seeds = style_boost_from_coact(state, q, len(vocab))
        log(f"  seeds: {[vocab.token_of(t) for t in seeds]}")
        raw_top = sorted(
            ((float(raw[j]), j) for j in range(len(raw)) if raw[j] > 0),
            reverse=True,
        )[:10]
        log(
            "  raw_top: "
            + "  ".join(
                f"{vocab.token_of(j)!r}({s:.1f}|bf={boundary_fitness(state, j):.1f})"
                for s, j in raw_top
            )
        )
        inv = ranked_style_inventory(state, q, top_k=12)
        log(
            "  filtered: "
            + ("  ".join(f"{vocab.token_of(j)!r}({sc:.2f})" for j, sc in inv) or "∅")
        )
        ph = ranked_style_phrases(state, q, top_k=8)
        log(
            "  phrases: "
            + (
                "  ".join(
                    f"{tuple(vocab.token_of(t) for t in p.tokens)!r}({p.score:.2f})"
                    for p in ph
                )
                or "∅"
            )
        )
        plan = style_inject_plan(state, q, max_items=3)
        log(
            "  inject_plan: "
            + (
                " | ".join(
                    " ".join(vocab.token_of(t) for t in unit) for unit in plan
                )
                or "∅"
            )
        )
        log()

    log(f"[{time.time()-t0:.0f}s] → đọc tay: filtered/phrases có tone? raw_top có domain (cốt/kịch)?")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
