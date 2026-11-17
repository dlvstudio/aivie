#!/usr/bin/env python3
"""
PROBE fact-safe style mix — fact liền + discourse n-gram + cosine.

    PYTHONPATH=src python -u scripts/probe_style_fact_mix.py \
        --log-file out/probe_style_fact_mix.log
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
from nn_aivie.hca.style_corpus import get_style_body_index, ranked_discourse_ngrams  # noqa: E402
from nn_aivie.hca.style_fluency import mean_bigram_cosine  # noqa: E402
from nn_aivie.hca.style_inventory import style_inject_plan  # noqa: E402

# prompt → fact surfaces phải còn trong output (không bị style xé)
CASES = [
    (
        "Viết một đoạn văn hài hước về Ngô Quyền",
        ["Ngô Quyền", "Bạch Đằng", "938"],
    ),
    (
        "Viết hài hước về Trần Hưng Đạo",
        ["Trần", "Nguyên"],
    ),
]

# lexicon mục tiêu từ eval_humor_lexicon
STYLE_TARGET = (
    "hệt như", "thế nhưng", "ngơ ngác", "lộng lẫy", "tơi bời",
    "xôn xao", "nơm nớp", "vĩnh viễn", "phũ phàng", "rốt cuộc",
)


def _strip(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()
    return text


def _fact_cover(text: str, keys: list[str]) -> float:
    if not keys:
        return 1.0
    hit = sum(1 for k in keys if k.lower() in text.lower())
    return hit / len(keys)


def _style_hits(text: str) -> list[str]:
    low = text.lower()
    return [s for s in STYLE_TARGET if s.lower() in low]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_1k.pkl")
    ap.add_argument("--log-file", default="out/probe_style_fact_mix.log")
    ap.add_argument("--max-len", type=int, default=70)
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

    variants = [
        ("MZ only", dict()),
        ("Mix mid (unsafe)", dict(
            ar_style_boundary_inject=True,
            ar_style_mixture=True,
            ar_style_fact_safe=False,
            ar_style_cos_thresh=0.12,
            ar_style_cos_margin=0.05,
            ar_style_mixture_cooldown=2,
            ar_style_use_inventory=True,
            ar_style_corpus_inventory=True,
            ar_style_discourse_ngram=True,
            ar_style_prefer_phrase=True,
            ar_style_weight=0.5,
            ar_style_boundary_max=2,
        )),
        ("Mix fact-safe", dict(
            ar_style_boundary_inject=True,
            ar_style_mixture=True,
            ar_style_fact_safe=True,
            ar_style_cos_thresh=0.10,
            ar_style_cos_margin=0.03,
            ar_style_mixture_cooldown=3,
            ar_style_use_inventory=True,
            ar_style_corpus_inventory=True,
            ar_style_discourse_ngram=True,
            ar_style_prefer_phrase=True,
            ar_style_weight=0.5,
            ar_style_boundary_max=2,
        )),
    ]

    for prompt, facts in CASES:
        q = list(vocab.encode_prompt(prompt))
        disc = ranked_discourse_ngrams(state, q, top_k=6, index=idx)
        plan = style_inject_plan(state, q, max_items=3, discourse_ngram=True)
        log(f"\n═══ {prompt!r} ═══")
        log(
            "  discourse: "
            + " | ".join(
                " ".join(vocab.token_of(t) for t in u) for u, _ in disc
            )
        )
        log(
            "  plan: "
            + " | ".join(" ".join(vocab.token_of(t) for t in u) for u in plan)
        )

        for name, kw in variants:
            cfg = HCAConfig(
                ar_verbatim_fallback=False,
                essay_ar_min_cov=0.0,
                ar_multizone=True,
                **kw,
            )
            ids = generate(state, prompt, max_len=args.max_len, config=cfg)
            text = _strip(decode_surface(state, ids), prompt)
            gen_ids = list(vocab.encode_prompt(text)) if text else []
            bigram = mean_bigram_cosine(emb, gen_ids)
            cover = _fact_cover(text, facts)
            hits = _style_hits(text)
            log(
                f"  {name:18} fact={cover:.2f} bigram={bigram:.4f} "
                f"style_hits={hits}"
            )
            log(f"    {text[:300]}")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    log("Đọc: fact≈1.0 + bigram gần MZ + style_hits có frame/redup → đạt.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
