#!/usr/bin/env python3
"""
TK_017 R3-data — nạp corpus VĂN NÓI vào substrate CHUỖI (fix trần continuation §13).

ĐÚNG BÀI HỌC D1 (flat-dump âm tính): data truyện/hội thoại nạp vào ĐÚNG TẦNG —
CHỈ `_g1_and_wseq` (W_seq + co-occurrence, năng lực chuỗi) — KHÔNG add_example
(không thêm cha → retrieval/agenda nguyên vẹn; QA guard kỳ vọng giữ).

network_v4.pkl → + văn nói → network_v4_conv.pkl (A/B trước khi tác giả quyết thay).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.core.embedding import build_embedding
from nn_aivie.data.quality import is_quality_vi_sentence, split_sentences
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.training.ingest import _g1_and_wseq
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]


def _collect_sentences(max_sents: int, log) -> list[str]:
    out: list[str] = []
    stories = _ROOT / "data/conv/stories_raw.jsonl"
    chat = _ROOT / "data/conv/chat_raw.jsonl"
    n_story = n_chat = 0
    if stories.exists():
        for line in stories.read_text("utf-8").splitlines():
            if len(out) >= max_sents * 7 // 10:  # ~70% truyện
                break
            row = json.loads(line)
            for s in split_sentences(row.get("text") or ""):
                # nới max_words=40: văn nói câu dài hơn fact wiki
                if is_quality_vi_sentence(s, max_words=40):
                    out.append(s)
                    n_story += 1
    if chat.exists():
        for line in chat.read_text("utf-8").splitlines():
            if len(out) >= max_sents:
                break
            row = json.loads(line)
            for turn in row.get("turns") or []:
                for s in split_sentences(turn or ""):
                    if is_quality_vi_sentence(s, max_words=40):
                        out.append(s)
                        n_chat += 1
    log.log(f"câu văn nói: {len(out)} (story {n_story} + chat {n_chat})")
    return out[:max_sents]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4.pkl")
    ap.add_argument("--out", default="checkpoints/network_v4_conv.pkl")
    ap.add_argument("--max-sents", type=int, default=60000)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=96)
    args = ap.parse_args()
    log = get_run_logger("build_conv")

    sents = _collect_sentences(args.max_sents, log)
    state, cache = load_checkpoint(args.checkpoint, logger=log)
    v0, p0 = len(state.vocab), len(state.parents.parents)
    log.log(f"trước: vocab={v0} parents={p0}")

    with log.phase("ingest W_seq văn nói (KHÔNG parents)"):
        for k, s in enumerate(sents):
            ids = state.vocab.encode_tokens(tokenize_punct(s))
            if len(ids) >= 2:
                _g1_and_wseq(state, ids, args.eta)
            if (k + 1) % 5000 == 0:
                log.progress("ingest_conv", k + 1, len(sents))
    log.log(f"sau: vocab={len(state.vocab)} (+{len(state.vocab)-v0}) "
            f"parents={len(state.parents.parents)} (+{len(state.parents.parents)-p0} — PHẢI =0)")

    with log.phase("rebuild embedding"):
        state.embedding = build_embedding(state.adjacency, state.vocab, dim=args.dim)
    with log.phase("save"):
        save_checkpoint(state, args.out, cache)
    log.log(f"XONG → {args.out}")


if __name__ == "__main__":
    main()
