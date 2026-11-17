#!/usr/bin/env python3
"""TK_024 nhánh C — probe generate trên entity-association query (A/B checkpoint).

So đáp sinh trước/sau khi nạp history_wiki_para (A1 theme_neuron dùng data dày hơn).
Chạy: python scripts/probe_entity_answers.py --checkpoint checkpoints/network_v8_histwiki.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.data.qa import DAP_LABEL, render_qa
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.hca.readout import decode_surface

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = [
    "Nhà Lý có gì",
    "Nhà Lý là gì",
    "Lý Thường Kiệt là ai",
    "Văn Miếu là gì",
    "Trần Hưng Đạo là ai",
    "Nhà Trần có gì",
    # control synonym (KHÔNG kỳ vọng ENTITY_LINK chữa — trần synonym)
    "Người chữa bệnh là ai",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    state, _ = load_checkpoint(ROOT / args.checkpoint)
    cfg = HCAConfig()
    ps = state.vocab
    from nn_aivie.hca.generate_hebbian import _content_ids, _knowledge_entity_span
    print(f"# {args.checkpoint}  vocab={len(state.vocab)} parents={len(state.parents)}\n")
    for q in PROMPTS:
        prompt, _ = render_qa(q, "")
        ids = generate(state, prompt, max_len=40, config=cfg)
        text = decode_surface(state, ids)
        ans = text.split(DAP_LABEL)[-1].strip() if DAP_LABEL in text else text
        # calibrate vá bồi-drift: entity/primary + gidf
        q_ids = list(state.vocab.encode_prompt(prompt))
        q_content = tuple(_content_ids(q_ids, state.vocab, skip=set()))
        span = _knowledge_entity_span(state.parents, q_content)
        if span is not None:
            entity, primary = span
            ent_s = " ".join(state.vocab.token_of(t) for t in entity)
            print(f"Q: {q}   [entity={ent_s!r} primary={state.vocab.token_of(primary)!r} "
                  f"gidf(primary)={state.parents._gidf(primary):.2f}]")
        else:
            print(f"Q: {q}   [no entity span]")
        print(f"   → {ans}\n")


if __name__ == "__main__":
    main()
