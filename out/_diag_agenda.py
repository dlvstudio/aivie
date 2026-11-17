"""Show agenda/entity for baseline prompts under mz on/off."""
from __future__ import annotations

from dataclasses import replace

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import _build_segment_plan, _split_frame
from nn_aivie.hca.w_skip import clear_w_skip_cache

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


def toks(st, ids, n=40):
    return [st.vocab.token_of(int(t)) for t in ids[:n]]


def main() -> None:
    clear_w_skip_cache()
    st, _ = load_checkpoint("checkpoints/network_v12_sent.pkl")
    for mz in (False, True):
        print("\n######## mz_agenda", mz, "########")
        cfg = replace(
            HCAConfig(),
            ar_w_skip_use_mz_agenda=mz,
            ar_multizone=mz,
        )
        for p in PROMPTS:
            generated = list(st.vocab.encode_prompt(p))
            q_ids, _ans = _split_frame(generated, st.vocab)
            query = list(q_ids) if q_ids else list(generated)
            cfg_plan = cfg
            if mz:
                cfg_plan = replace(cfg, ar_multizone=True)
            plan = _build_segment_plan(
                st, query, cfg_plan,
                max_facts=int(getattr(cfg, "ar_multizone_facts_top", 1) or 1) + 2,
                emitted_ids=generated,
            )
            ag = list(plan.flat_agenda)
            ents = tuple(plan.entity_ids)
            print("-" * 50)
            print(p)
            print(" q:", toks(st, query, 20))
            print(" agenda:", toks(st, ag, 30))
            print(" ents:", toks(st, ents, 20))


if __name__ == "__main__":
    main()
