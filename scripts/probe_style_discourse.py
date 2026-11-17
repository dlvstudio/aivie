#!/usr/bin/env python3
"""
PROBE discourse n-gram inventory (thân styled, Luật 0).

    PYTHONPATH=src python -u scripts/probe_style_discourse.py \
        --checkpoint checkpoints/network_v11_styled_1k.pkl \
        --log-file out/probe_style_discourse.log
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
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.style_corpus import (  # noqa: E402
    get_style_body_index,
    ranked_corpus_inventory,
    ranked_discourse_ngrams,
)
from nn_aivie.hca.style_fluency import annotate_inject_fits, mean_bigram_cosine  # noqa: E402
from nn_aivie.hca.style_inventory import style_inject_plan  # noqa: E402

PROMPTS = [
    "Viết một đoạn văn hài hước về Ngô Quyền",
    "Viết hài hước về Trần Hưng Đạo",
    "Viết hài hước về gen Z",
]


def _strip(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_1k.pkl")
    ap.add_argument("--log-file", default="out/probe_style_discourse.log")
    ap.add_argument("--max-len", type=int, default=50)
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
    vocab, emb = state.vocab, state.embedding
    idx = get_style_body_index(state)
    log(f"[{time.time()-t0:.0f}s] bodies={idx.n_bodies if idx else 0}")

    for prompt in PROMPTS:
        q = list(vocab.encode_prompt(prompt))
        log(f"\n═══ {prompt!r} ═══")
        uni = ranked_corpus_inventory(state, q, top_k=6, index=idx)
        log(
            "  old_uni: "
            + "  ".join(f"{vocab.token_of(j)!r}" for j, _ in uni)
        )
        disc = ranked_discourse_ngrams(state, q, top_k=8, index=idx)
        log(
            "  discourse: "
            + "  ".join(
                f"{tuple(vocab.token_of(t) for t in u)!r}({sc:.1f})"
                for u, sc in disc
            )
        )
        plan = style_inject_plan(state, q, max_items=3, use_corpus=True, discourse_ngram=True)
        log(
            "  plan: "
            + " | ".join(" ".join(vocab.token_of(t) for t in u) for u in plan)
        )

        style_ids = {t for u in plan for t in u}
        for name, kw in (
            ("MZ", dict()),
            ("Mix+discourse", dict(
                ar_style_boundary_inject=True,
                ar_style_mixture=True,
                ar_style_cos_thresh=0.12,
                ar_style_mixture_cooldown=2,
                ar_style_cos_margin=0.05,
                ar_style_use_inventory=True,
                ar_style_corpus_inventory=True,
                ar_style_discourse_ngram=True,
                ar_style_prefer_phrase=True,
                ar_style_weight=0.5,
                ar_style_boundary_max=3,
            )),
        ):
            cfg = HCAConfig(
                ar_verbatim_fallback=False, essay_ar_min_cov=0.0, ar_multizone=True, **kw,
            )
            ids = generate(state, prompt, max_len=args.max_len, config=cfg)
            text = _strip(decode_surface(state, ids), prompt)
            gen_ids = list(vocab.encode_prompt(text)) if text else []
            bigram = mean_bigram_cosine(emb, gen_ids)
            fits = annotate_inject_fits(emb, gen_ids, style_ids)
            fit = sum(fits) / len(fits) if fits else float("nan")
            hits = [" ".join(vocab.token_of(t) for t in u)
                    for u in plan if all(t in gen_ids for t in u)]
            # also partial token hits
            tok_hits = [vocab.token_of(t) for t in gen_ids if t in style_ids]
            log(
                f"  {name:14} bigram={bigram:.4f} fit={fit:.4f} "
                f"n={len(fits)} phrase_hits={hits} toks={tok_hits}"
            )
            log(f"    {text[:240]}")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
