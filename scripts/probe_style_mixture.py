#!/usr/bin/env python3
"""
PROBE style mixture — đo độ mượt bằng cosine E (không đọc tay suông).

    PYTHONPATH=src python -u scripts/probe_style_mixture.py \
        --checkpoint checkpoints/network_v11_styled_1k.pkl \
        --log-file out/probe_style_mixture_humor.log

Metrics (Luật 0 — chỉ E đã học):
  mean_bigram_cos  — trung bình cos(t_i, t_{i+1}) trên chuỗi sinh
  inject_fit_cos   — trung bình slot_fit tại token thuộc style plan
  n_style_hits     — số token style plan xuất hiện trong output
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.style_fluency import (  # noqa: E402
    annotate_inject_fits,
    mean_bigram_cosine,
)
from nn_aivie.hca.style_inventory import style_inject_plan  # noqa: E402

# Nhiều prompt chứa «hài hước» — history + đời thường + topic trong styled_sft.
PROMPTS = [
    "Viết một đoạn văn hài hước về Ngô Quyền",
    "Viết hài hước về Trần Hưng Đạo",
    "Viết hài hước về trận Bạch Đằng",
    "Viết hài hước về gen Z",
    "Viết hài hước về việc học",
]


def _strip_prompt(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_1k.pkl")
    ap.add_argument("--log-file", default="out/probe_style_mixture_humor.log")
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--cos-thresh", type=float, default=0.12)
    ap.add_argument(
        "--variants", default="mz,mix",
        help="comma: mz,hard,mix",
    )
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
    log(f"[{time.time()-t0:.0f}s] ✓ parents {len(state.parents):,}")
    log(f"prompts ({len(PROMPTS)} × hài hước):")
    for p in PROMPTS:
        log(f"  • {p}")

    want = {x.strip() for x in args.variants.split(",") if x.strip()}
    variant_defs = []
    if "mz" in want:
        variant_defs.append(("MZ only", dict()))
    if "hard" in want:
        variant_defs.append(("Hard boundary", dict(
            ar_style_boundary_inject=True,
            ar_style_mixture=False,
            ar_style_use_inventory=True,
            ar_style_corpus_inventory=True,
            ar_style_prefer_phrase=False,
            ar_style_weight=0.5,
            ar_style_boundary_max=3,
        )))
    if "mix" in want:
        variant_defs.append(("Mix every-pos", dict(
            ar_style_boundary_inject=True,
            ar_style_mixture=True,
            ar_style_cos_thresh=args.cos_thresh,
            ar_style_mixture_cooldown=2,
            ar_style_cos_margin=0.05,
            ar_style_use_inventory=True,
            ar_style_corpus_inventory=True,
            ar_style_prefer_phrase=False,
            ar_style_weight=0.5,
            ar_style_boundary_max=3,
        )))

    # aggregate per variant
    agg: dict[str, list[tuple[float, float, int]]] = {n: [] for n, _ in variant_defs}

    for pi, prompt in enumerate(PROMPTS, 1):
        q = list(vocab.encode_prompt(prompt))
        plan = style_inject_plan(
            state, q, max_items=3, use_corpus=True, prefer_phrase=False,
        )
        style_ids = {t for u in plan for t in u}
        log(f"\n═══ [{pi}/{len(PROMPTS)}] {prompt!r} ═══")
        log(
            "  plan: "
            + (" | ".join(" ".join(vocab.token_of(t) for t in u) for u in plan) or "∅")
        )

        for name, kw in variant_defs:
            log(f"  … {name}")
            cfg = HCAConfig(
                ar_verbatim_fallback=False,
                essay_ar_min_cov=0.0,
                ar_multizone=True,
                **kw,
            )
            ids = generate(state, prompt, max_len=args.max_len, config=cfg)
            text = _strip_prompt(decode_surface(state, ids), prompt)
            gen_ids = list(vocab.encode_prompt(text)) if text else []
            bigram = mean_bigram_cosine(emb, gen_ids)
            fits = annotate_inject_fits(emb, gen_ids, style_ids)
            fit_mean = sum(fits) / len(fits) if fits else float("nan")
            hits = [vocab.token_of(t) for t in gen_ids if t in style_ids]
            log(
                f"  {name:16}  bigram={bigram:.4f}  "
                f"fit={fit_mean:.4f}  n={len(fits)}  hits={hits}"
            )
            log(f"    {text[:220]}")
            agg[name].append((bigram, fit_mean, len(fits)))

    log("\n═══ aggregate (mean over prompts) ═══")
    for name, rows in agg.items():
        if not rows:
            continue
        b = sum(r[0] for r in rows) / len(rows)
        fits = [r[1] for r in rows if math.isfinite(r[1])]
        f = sum(fits) / len(fits) if fits else float("nan")
        n = sum(r[2] for r in rows) / len(rows)
        log(f"  {name:16}  mean_bigram={b:.4f}  mean_fit={f:.4f}  mean_n_style={n:.1f}")

    if "MZ only" in agg and "Mix every-pos" in agg and agg["MZ only"] and agg["Mix every-pos"]:
        mz_b = sum(r[0] for r in agg["MZ only"]) / len(agg["MZ only"])
        mix_b = sum(r[0] for r in agg["Mix every-pos"]) / len(agg["Mix every-pos"])
        log(f"  Δ Mix−MZ bigram = {mix_b - mz_b:+.4f}")

    log(f"[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
