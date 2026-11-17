#!/usr/bin/env python3
"""One-step debug: scores after each stage for được→?"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.context_positional import (  # noqa: E402
    content_token_ids,
    distant_content_micro,
    ensure_varlm,
    ngram_varlm_micro,
    order2_micro,
    positional_readout_base,
    positional_wseq_micro,
)
from nn_aivie.hca.knowledge_seq import (  # noqa: E402
    apply_gidf_target_gate,
    oja_compete,
    parent_span_next_micro,
    positional_know_micro,
)


def top(vocab, sc, k=6):
    sc = np.asarray(sc, dtype=np.float64).copy()
    sc[~np.isfinite(sc)] = -1e30
    idx = np.argpartition(-sc, k - 1)[:k]
    idx = idx[np.argsort(-sc[idx])]
    return [(vocab.token_of(int(i)), round(float(sc[i]), 4)) for i in idx]


def main() -> int:
    ckpt = _ROOT / "checkpoints/network_v11_styled_hist.pkl"
    gc.disable()
    state, _ = load_checkpoint(str(ckpt))
    gc.enable()
    ensure_varlm(state, fallbacks=[_ROOT / "checkpoints/network_v11.varlm.pkl"])
    vocab = state.vocab
    n = len(vocab)
    cfg = HCAConfig(ar_no_agenda=True, ar_spectral_stack=False)
    prompt = "Ngô Quyền là vị vua sáng lập nhà Ngô, được"
    out = vocab.encode_prompt(prompt)
    pos_lam, pos_len = cfg.ar_ctx_lambda, cfg.ar_ctx_len

    base = positional_readout_base(state, out, vocab_size=n, lam=pos_lam, ctx_len=pos_len)
    micro = positional_wseq_micro(
        state, out, vocab_size=n, lam=pos_lam, ctx_len=pos_len,
        use_ppmi=True, skip_structural=True,
    )
    micro = micro + cfg.ar_ngram_weight * ngram_varlm_micro(state, out, vocab_size=n)
    micro = micro + cfg.ar_order2_cont_weight * order2_micro(state, out, vocab_size=n)
    cid = content_token_ids(out, vocab)
    micro = micro + cfg.ar_distant_content_weight * distant_content_micro(
        state, cid, vocab_size=n, mode=cfg.ar_distant_mode
    )
    print("after flu", top(vocab, micro))
    micro = oja_compete(micro, beta=cfg.ar_oja_compete)
    span = parent_span_next_micro(state, out, vocab_size=n)
    know_bi = positional_know_micro(state, out, vocab_size=n)
    micro = micro + cfg.ar_know_weight * span + cfg.ar_know_bi_weight * know_bi
    print("span", top(vocab, span))
    print("after know-in-micro", top(vocab, micro))
    apply_gidf_target_gate(micro, state, weight=cfg.ar_gidf_target_weight)
    print("after gidf", top(vocab, micro))
    scores = base * (cfg.hybrid_micro_floor + cfg.hybrid_micro_weight * micro)
    print("after hadamard", top(vocab, scores))
    print("base top", top(vocab, base))
    xem = vocab._token_to_id.get("Ġxem")
    de = vocab._token_to_id.get("Ġđể")
    if xem is not None:
        print(f"base[xem]={base[xem]:.4g} micro[xem]={micro[xem]:.4g} sc[xem]={scores[xem]:.4g}")
    if de is not None:
        print(f"base[để]={base[de]:.4g} micro[để]={micro[de]:.4g} sc[để]={scores[de]:.4g}")
    kn = span
    peak = float(scores[np.isfinite(scores)].max())
    scores2 = scores + (peak * cfg.ar_know_weight) * kn
    print("after +know", top(vocab, scores2))
    print("argmax", vocab.token_of(int(np.argmax(scores2))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
