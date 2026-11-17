#!/usr/bin/env python3
"""
TK_017 R3b — POOL VĂN NÓI two-tier cho continuation (pattern-completion thay Markov greedy).

Nguyên tắc (bài học D1 + 0.5.10):
  - Câu văn nói → nơ-ron cha ở **priority=0.3** (pool riêng — two-tier: KHÔNG cạnh tranh
    QA/curated ở đường crisp; chỉ được truy hồi qua `restrict_parents` khi opener-route).
  - CHỌN câu bằng chính substrate D3: chỉ nạp câu CHỨA bracket-opener (row FRAME_BRACKET)
    → tập trung năng lực khung, không phình store (định luật 1: chọn, không cắm phẳng).
  - KHÔNG `_g1_and_wseq` → ruler/base-LM giữ NGUYÊN 4.028 (đã đo _conv: co-act văn nói
    pha loãng +0.205 — không lặp lại).
  - Mine FRAME_BRACKET tại build + SAVE (cạnh persist trong adjacency — gỡ nợ F3 lazy-mine
    15-40s mỗi process).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.data.quality import is_quality_vi_sentence, split_sentences
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.training.frame_bracket import mine_frame_brackets
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]
CONV_PRIORITY = 0.3


def _is_conv_sentence(text: str) -> bool:
    """Lọc kiếm hiệp/CJK/FX — giữ văn nói tiếng Việt sạch cho pool continuation."""
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return False
    if "FxPro" in text or "pips" in text.lower():
        return False
    words = text.split()
    caps_run = 0
    for w in words:
        if len(w) > 1 and w[0].isupper() and w[1:].islower():
            caps_run += 1
            if caps_run >= 4:
                return False
        else:
            caps_run = 0
    return True


def _sentences() -> list[str]:
    out: list[str] = []
    for name, key in [("stories_raw.jsonl", "text"), ("chat_raw.jsonl", None)]:
        p = _ROOT / "data/conv" / name
        if not p.exists():
            continue
        for line in p.read_text("utf-8").splitlines():
            row = json.loads(line)
            texts = [row.get(key) or ""] if key else (row.get("turns") or [])
            for t in texts:
                for s in split_sentences(t or ""):
                    if is_quality_vi_sentence(s, max_words=40):
                        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4.pkl")
    ap.add_argument("--out", default="checkpoints/network_v4_convpool.pkl")
    ap.add_argument("--max-parents", type=int, default=20000)
    args = ap.parse_args()
    log = get_run_logger("build_conv_pool")

    state, cache = load_checkpoint(args.checkpoint, logger=log)
    ps, vocab = state.parents, state.vocab
    v0, p0 = len(vocab), len(ps.parents)

    with log.phase("mine FRAME_BRACKET (persist vào checkpoint — gỡ nợ F3)"):
        if len(state.adjacency.layer(EdgeKind.FRAME_BRACKET)) == 0:
            log.log(str(mine_frame_brackets(state, logger=log)))
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)

    with log.phase("lọc câu văn nói CHỨA opener"):
        sents = _sentences()
        log.log(f"câu qua quality: {len(sents)}")
        known = set(vocab._token_to_id.keys())
        picked: list[list[int]] = []
        skipped_oov = 0
        for s in sents:
            if not _is_conv_sentence(s):
                continue
            toks = tokenize_punct(s)
            if not toks or not all(t in known for t in toks):
                skipped_oov += 1
                continue
            ids = vocab.encode_tokens(toks)
            if len(ids) < 4:
                continue
            if any(fb.row(t) for t in ids[:-2]):  # opener phải còn chỗ cho closer phía sau
                picked.append(ids)
            if len(picked) >= args.max_parents:
                break
        log.log(f"câu chứa opener: {len(picked)} (bỏ OOV: {skipped_oov})")

    with log.phase("nạp pool parents (priority=0.3, defer_index, KHÔNG W_seq)"):
        ps.seg.begin_batch()
        for k, ids in enumerate(picked):
            ps.add_example(ids, vocab, dap_id=None, priority=CONV_PRIORITY, defer_index=True)
            if (k + 1) % 2000 == 0:
                log.progress("pool", k + 1, len(picked))
        ps.seg.end_batch(ps.parents)
        ps.rebuild_hierarchical_index()
        ps.flush_skip_ledger(vocab, logger=log)
    log.log(f"parents {p0} → {len(ps.parents)} (+{len(ps.parents)-p0}); vocab {v0} → {len(vocab)}")

    with log.phase("save"):
        save_checkpoint(state, args.out, cache, bake_derived=True, logger=log)
    log.log(f"XONG → {args.out}")


if __name__ == "__main__":
    main()
