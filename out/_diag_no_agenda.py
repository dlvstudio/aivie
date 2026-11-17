"""Compare gens: agenda on/off + hist probe after Đáp:."""
from __future__ import annotations

from dataclasses import replace

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import _wseq_fallback, generate
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.w_skip import clear_w_skip_cache, hist_after_last_boundary
from nn_aivie.core.tokenizer import SPECIAL_EOS
import numpy as np

PROMPTS = [
    "Năm 1010, vua Lý Thái Tổ",
    "Nhà Lý dời đô về",
    "Hỏi: Nhà Lý dời đô về đâu? Đáp:",
    "Việt Nam là một quốc gia nằm ở",
    "Phở là món ăn",
]


def main() -> None:
    clear_w_skip_cache()
    st, _ = load_checkpoint("checkpoints/network_v12_sent.pkl")
    eos = st.vocab.id_of(SPECIAL_EOS)
    cfg = replace(HCAConfig(), ar_w_skip_use_mz_agenda=False, ar_multizone=False)

    p = "Hỏi: Nhà Lý dời đô về đâu? Đáp:"
    ids = list(st.vocab.encode_prompt(p))
    print("encode:", [st.vocab.token_of(t) for t in ids])
    print("hist_after:", [st.vocab.token_of(t) for t in hist_after_last_boundary(ids, st.vocab, pre_boundary_k=2)])

    for no_ag in (False, True):
        print("\n==== force_empty_agenda", no_ag, "====")
        for prompt in PROMPTS:
            generated = list(st.vocab.encode_prompt(prompt))
            if no_ag:
                out = _wseq_fallback(
                    st, generated, eos, 40, cfg, np.random.default_rng(0),
                    agenda_ids=[], entity_ids=(),
                )
            else:
                out = generate(st, prompt, max_len=40, config=cfg, seed=0)
            print("-", prompt)
            print(" ", decode_surface(st, out))


if __name__ == "__main__":
    main()
