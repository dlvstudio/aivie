#!/usr/bin/env python3
"""
PROBE corpus style inventory (label-conditioned body ngrams) — cho M4.

    PYTHONPATH=src python -u scripts/probe_style_corpus_inv.py \
        --checkpoint checkpoints/network_v11_styled_1k.pkl \
        --log-file out/probe_style_corpus_inv.log
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
from nn_aivie.hca.style_corpus import (  # noqa: E402
    get_style_body_index,
    ranked_corpus_inventory,
    ranked_corpus_phrases,
)
from nn_aivie.hca.style_inventory import style_inject_plan  # noqa: E402
from nn_aivie.hca.style_readout import style_seeds_from_prompt  # noqa: E402

PROMPTS = [
    "Viết một đoạn văn hài hước về Ngô Quyền",
    "Kể buồn về mùa thu Hà Nội",
    "So sánh Nhà Lý và Nhà Trần",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_1k.pkl")
    ap.add_argument("--log-file", default="out/probe_style_corpus_inv.log")
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
    log(f"[{time.time()-t0:.0f}s] ✓ parents {len(state.parents):,}")

    idx = get_style_body_index(state)
    log(f"[{time.time()-t0:.0f}s] corpus bodies={idx.n_bodies if idx else 0} "
        f"seed_keys={len(idx.seed_uni) if idx else 0}\n")

    for prompt in PROMPTS:
        q = list(vocab.encode_prompt(prompt))
        seeds = style_seeds_from_prompt(q, vocab, state.parents)
        log(f"═══ {prompt!r} ═══")
        log(f"  seeds: {[vocab.token_of(t) for t in seeds]}")
        uni = ranked_corpus_inventory(state, q, top_k=12, index=idx)
        log(
            "  corpus_uni: "
            + ("  ".join(f"{vocab.token_of(j)!r}({sc:.2f})" for j, sc in uni) or "∅")
        )
        bi = ranked_corpus_phrases(state, q, top_k=8, index=idx)
        log(
            "  corpus_bi: "
            + (
                "  ".join(
                    f"{(vocab.token_of(u), vocab.token_of(v))!r}({sc:.2f})"
                    for (u, v), sc in bi
                )
                or "∅"
            )
        )
        plan = style_inject_plan(state, q, max_items=3, use_corpus=True)
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

    log(f"[{time.time()-t0:.0f}s] → đọc tay: corpus có tone (phết/hơi lố/kiểu)?")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
