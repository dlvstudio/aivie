#!/usr/bin/env python3
"""Quick check: PPMI+skip+coact blend vs raw; short no-agenda gen."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.context_positional import (  # noqa: E402
    content_token_ids,
    distant_content_micro,
    ensure_varlm,
    ngram_varlm_micro,
    order2_micro,
    positional_wseq_micro,
)
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402


def top(vocab, sc, k=6):
    idx = np.argpartition(-sc, k - 1)[:k]
    idx = idx[np.argsort(-sc[idx])]
    return [(vocab.token_of(int(i)), round(float(sc[i]), 3)) for i in idx if sc[i] > 0]


def main() -> int:
    ckpt = _ROOT / "checkpoints/network_v11_styled_hist.pkl"
    gc.disable()
    state, _ = load_checkpoint(str(ckpt))
    gc.enable()
    state._checkpoint_path = str(ckpt)  # type: ignore[attr-defined]
    ensure_varlm(state, fallbacks=[_ROOT / "checkpoints/network_v11.varlm.pkl"])
    vocab = state.vocab
    n = len(vocab)
    prompt = "Viết một đoạn văn hài hước về Ngô Quyền"
    ids = vocab.encode_prompt(prompt)
    cid = content_token_ids(ids, vocab)

    pos_raw = positional_wseq_micro(
        state, ids, vocab_size=n, use_ppmi=False, skip_structural=False
    )
    pos_fix = positional_wseq_micro(
        state, ids, vocab_size=n, use_ppmi=True, skip_structural=True
    )
    var = ngram_varlm_micro(state, ids, vocab_size=n)
    o2 = order2_micro(state, ids, vocab_size=n)
    dist_w = distant_content_micro(state, cid, vocab_size=n, mode="wseq")
    dist_c = distant_content_micro(state, cid, vocab_size=n, mode="coact")
    dist_p = distant_content_micro(state, cid, vocab_size=n, mode="ppmi")
    print("pos_raw", top(vocab, pos_raw))
    print("pos_fix", top(vocab, pos_fix))
    print("varlm  ", top(vocab, var))
    print("dist_w ", top(vocab, dist_w))
    print("dist_c ", top(vocab, dist_c))
    print("dist_p ", top(vocab, dist_p))
    blend = pos_fix + 1.15 * var + 0.55 * o2 + 0.4 * dist_c
    print("blend  ", top(vocab, blend))

    ids2 = ids + vocab.encode_prompt(", một nhận")
    pos2 = positional_wseq_micro(
        state, ids2, vocab_size=n, use_ppmi=True, skip_structural=True
    )
    var2 = ngram_varlm_micro(state, ids2, vocab_size=n)
    blend2 = (
        pos2
        + 1.15 * var2
        + 0.4
        * distant_content_micro(
            state, content_token_ids(ids2, vocab), vocab_size=n, mode="coact"
        )
    )
    print("after nhận blend", top(vocab, blend2))

    cfg = HCAConfig(
        ar_no_agenda=True,
        ar_spectral_stack=False,
        ar_multizone=False,
        ar_pattern_completion=False,
        ar_conv_pool=False,
        hybrid_temperature=0.0,
    )
    out = generate(state, prompt, max_len=60, config=cfg)
    text = decode_surface(state, out)
    body = text[len(prompt) :].strip() if text.startswith(prompt) else text
    print("GEN:", body[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
