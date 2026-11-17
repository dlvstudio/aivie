#!/usr/bin/env python3
"""
Probe W_know + Oja: được→xem vs để; gen prefix agenda.

    PYTHONPATH=src python -u scripts/probe_w_know.py --log-file out/probe_w_know.log
"""

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
from nn_aivie.hca.context_positional import (  # noqa: E402
    ensure_varlm,
    ngram_varlm_micro,
    positional_wseq_micro,
)
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.knowledge_seq import (  # noqa: E402
    oja_compete,
    parent_span_next_micro,
    positional_know_micro,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402


def top(vocab, sc, k=8):
    idx = np.argpartition(-sc, min(k, len(sc) - 1) - 0)[:k]
    idx = idx[np.argsort(-sc[idx])]
    return [(vocab.token_of(int(i)), round(float(sc[i]), 3)) for i in idx if sc[i] > 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", default="out/probe_w_know.log")
    ap.add_argument("--max-len", type=int, default=64)
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
    ensure_varlm(state, fallbacks=[_ROOT / "checkpoints/network_v11.varlm.pkl"])
    vocab = state.vocab
    n = len(vocab)

    prompt = "Ngô Quyền là vị vua sáng lập nhà Ngô, được"
    ids = vocab.encode_prompt(prompt)
    log(f"prompt={prompt!r}")
    log(f"w0={vocab.token_of(ids[-1])!r}")

    pos = positional_wseq_micro(
        state, ids, vocab_size=n, use_ppmi=True, skip_structural=True
    )
    var = ngram_varlm_micro(state, ids, vocab_size=n)
    know_bi = positional_know_micro(state, ids, vocab_size=n)
    span = parent_span_next_micro(state, ids, vocab_size=n)
    flu = oja_compete(pos + 1.15 * var, beta=0.55)
    blend = flu + 2.0 * span + 0.9 * know_bi
    log(f"pos_ppmi  {top(vocab, pos)}")
    log(f"varlm     {top(vocab, var)}")
    log(f"know_bi   {top(vocab, know_bi)}")
    log(f"span_next {top(vocab, span)}")
    log(f"blend     {top(vocab, blend)}")

    for name, tok in (("xem", "Ġxem"), ("để", "Ġđể")):
        tid = vocab._token_to_id.get(tok)
        if tid is None:
            continue
        log(
            f"  rank {tok}: span={int(np.sum(span>span[tid]))+1 if span[tid]>0 else -1} "
            f"blend={int(np.sum(blend>blend[tid]))+1}"
        )

    modes = [
        ("no_agenda+know", dict(ar_no_agenda=True, ar_spectral_stack=False,
            ar_multizone=False, ar_pattern_completion=False, ar_conv_pool=False)),
        ("MZ", dict(ar_no_agenda=False, ar_spectral_stack=False, ar_multizone=True,
            ar_verbatim_fallback=False, essay_ar_min_cov=0.0)),
    ]
    for name, kw in modes:
        cfg = HCAConfig(ar_style_payoff=False, hybrid_temperature=0.0, **kw)
        out = generate(state, prompt, max_len=args.max_len, config=cfg)
        full = decode_surface(state, out)
        log(f"\n--- {name} ---")
        log(full[:500])

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
