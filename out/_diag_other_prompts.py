"""Diagnose top W_skip votes + short gens for non-smoke baseline prompts."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.hca.matrix_core import inner_digit_ids
from nn_aivie.hca.readout import decode_surface, mask_non_surface
from nn_aivie.hca.w_skip import (
    clear_w_skip_cache,
    hist_after_last_boundary,
    vote_logits,
)

PROMPTS = [
    "Nhà Lý dời đô về",
    "Ngô Quyền đánh bại quân Nam Hán năm",
    "Hỏi: Nhà Lý dời đô về đâu? Đáp:",
    "Việt Nam là một quốc gia nằm ở",
    "Phở là món ăn",
    "Trong thời kỳ Bắc thuộc, nhân dân ta",
    "Thủ đô Hà Nội có lịch sử hơn nghìn năm, từ khi",
]


def main() -> None:
    clear_w_skip_cache()
    st, _ = load_checkpoint("checkpoints/network_v12_sent.pkl")
    cfg = replace(HCAConfig(), ar_w_skip_use_mz_agenda=False, ar_multizone=False)
    n = len(st.vocab)
    d_max = int(getattr(cfg, "ar_w_skip_d_max", 4) or 4)
    lam = float(getattr(cfg, "ar_w_skip_lam", 0.85) or 0.85)
    topk = int(getattr(cfg, "ar_w_skip_vote_topk", 0) or 0)

    def key(tid: int) -> str:
        s = st.vocab.token_of(int(tid))
        return s.lstrip(MARK_SPACE).lstrip("Ċ")

    for p in PROMPTS:
        out = list(st.vocab.encode_prompt(p))
        vh = hist_after_last_boundary(out, st.vocab, pre_boundary_k=2) or out
        sc = vote_logits(
            st, list(vh), n_vocab=n, d_max=d_max, lam=lam, topk=topk, agg="sum"
        )
        sc[inner_digit_ids(st.vocab)] = 0.0
        sc = mask_non_surface(sc, st.vocab)
        pk = {key(t) for t in out}
        for tid in np.flatnonzero(np.isfinite(sc) & (sc > 0)):
            if key(int(tid)) in pk:
                sc[int(tid)] = -np.inf
        idx = np.argsort(-np.where(np.isfinite(sc), sc, -np.inf))[:12]
        print("=" * 60)
        print("PROMPT:", p)
        print(" hist:", [st.vocab.token_of(t) for t in vh[-10:]])
        for i in idx:
            print(f"  {float(sc[i]):8.3f}  {st.vocab.token_of(int(i))!r}")
        for seed in range(2):
            t = decode_surface(
                st, generate(st, p, max_len=40, config=cfg, seed=seed)
            )
            print(f" seed{seed}: {t}")
        print()


if __name__ == "__main__":
    main()
