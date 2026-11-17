"""Smoke other prompts after §12.149 fixes."""
from __future__ import annotations

from dataclasses import replace

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, _build_segment_plan, _split_frame
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.w_skip import clear_w_skip_cache, hist_after_last_boundary

PROMPTS = [
    "Năm 1010, vua Lý Thái Tổ",
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
    for p in PROMPTS:
        generated = list(st.vocab.encode_prompt(p))
        q_ids, _ = _split_frame(generated, st.vocab)
        query = list(q_ids) if q_ids else list(generated)
        plan = _build_segment_plan(st, query, cfg, max_facts=3, emitted_ids=generated)
        vh = hist_after_last_boundary(generated, st.vocab, pre_boundary_k=2)
        print("=" * 60)
        print(p)
        print(" hist:", [st.vocab.token_of(t) for t in vh[-12:]])
        print(" agenda:", [st.vocab.token_of(t) for t in plan.flat_agenda[:24]])
        for seed in range(2):
            t = decode_surface(st, generate(st, p, max_len=40, config=cfg, seed=seed))
            print(f" s{seed}: {t}")


if __name__ == "__main__":
    main()
