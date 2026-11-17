#!/usr/bin/env python3
"""Probe matrix core: argmax vs multinomial sampling."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.matrix_core import (  # noqa: E402
    matrix_core_logits,
    parent_coactivation_logits,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402


def top(vocab, sc, k=8):
    sc = np.asarray(sc, float).copy()
    sc[~np.isfinite(sc)] = -1e30
    idx = np.argpartition(-sc, k - 1)[:k]
    idx = idx[np.argsort(-sc[idx])]
    return [(vocab.token_of(int(i)), round(float(sc[i]), 3)) for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", default="out/probe_matrix_core.log")
    ap.add_argument("--max-len", type=int, default=56)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=42)
    args = ap.parse_args()
    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    ckpt = _ROOT / args.checkpoint
    gc.disable()
    state, _ = load_checkpoint(str(ckpt))
    gc.enable()
    state._checkpoint_path = str(ckpt)  # type: ignore[attr-defined]
    state._spectral_stack_sidecar = str(ckpt.with_suffix(".spectral_stack.npz"))  # type: ignore

    prompt = "Ngô Quyền là vị vua sáng lập nhà Ngô, được"
    ids = state.vocab.encode_prompt(prompt)
    cfg = HCAConfig(ar_matrix_core=True, ar_no_agenda=True, hybrid_temperature=0.0)
    bag, succ, conf, _act = parent_coactivation_logits(state, ids, vocab_size=len(state.vocab))
    logits = matrix_core_logits(state, ids, config=cfg)
    log(f"prompt={prompt!r}")
    log(f"parent conf={conf:.3f}")
    log(f"parent_bag  top={top(state.vocab, bag)}")
    log(f"parent_succ top={top(state.vocab, succ)}")
    log(f"matrix logits top={top(state.vocab, logits)}")
    for tok in ("Ġxem", "Ġđể", "Ġgọi", "<EOS>"):
        tid = state.vocab._token_to_id.get(tok)
        if tid is not None:
            log(
                f"  {tok} rank={int(np.sum(logits > logits[tid])) + 1} "
                f"val={logits[tid]:.3f}"
            )

    # --- argmax baseline ---
    for name, kw in (
        (
            "argmax no_agenda+matrix",
            dict(
                ar_no_agenda=True,
                ar_multizone=False,
                ar_pattern_completion=False,
                ar_conv_pool=False,
                hybrid_temperature=0.0,
            ),
        ),
        (
            "argmax MZ+matrix",
            dict(
                ar_no_agenda=False,
                ar_multizone=True,
                ar_verbatim_fallback=False,
                essay_ar_min_cov=0.0,
                hybrid_temperature=0.0,
            ),
        ),
    ):
        c = HCAConfig(ar_style_payoff=False, ar_matrix_core=True, **kw)
        out = generate(state, prompt, max_len=args.max_len, config=c, seed=args.seed0)
        log(f"\n--- {name} ---")
        log(decode_surface(state, out)[:500])

    # --- multinomial samples (matrix path) ---
    log(
        f"\n=== sample no_agenda+matrix "
        f"T={args.temperature} top_k={args.top_k} n={args.n_samples} ==="
    )
    for i in range(args.n_samples):
        seed = args.seed0 + i
        c = HCAConfig(
            ar_style_payoff=False,
            ar_matrix_core=True,
            ar_no_agenda=True,
            ar_multizone=False,
            ar_pattern_completion=False,
            ar_conv_pool=False,
            hybrid_temperature=float(args.temperature),
            hybrid_top_k=int(args.top_k),
            ar_temperature=float(args.temperature),
        )
        out = generate(state, prompt, max_len=args.max_len, config=c, seed=seed)
        log(f"\n--- sample seed={seed} ---")
        log(decode_surface(state, out)[:500])

    # --- sample MZ (agenda vẫn neo; micro có thể sample) ---
    log(
        f"\n=== sample MZ+matrix T={args.temperature} top_k={args.top_k} n={args.n_samples} ==="
    )
    for i in range(args.n_samples):
        seed = args.seed0 + 100 + i
        c = HCAConfig(
            ar_style_payoff=False,
            ar_matrix_core=True,
            ar_no_agenda=False,
            ar_multizone=True,
            ar_verbatim_fallback=False,
            essay_ar_min_cov=0.0,
            hybrid_temperature=float(args.temperature),
            hybrid_top_k=int(args.top_k),
            ar_temperature=float(args.temperature),
        )
        out = generate(state, prompt, max_len=args.max_len, config=c, seed=seed)
        log(f"\n--- MZ sample seed={seed} ---")
        log(decode_surface(state, out)[:500])

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
