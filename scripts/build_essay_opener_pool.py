#!/usr/bin/env python3
"""
TK_017 R4b — sidecar pattern_templates cho prompt mở lửng (KHÔNG add ParentStore).

Tránh dilution phrase index / Para regression khi nạp thêm cha priority thấp.
Nguồn: data/conv + real_vi/essays + statements (câu chưa có trong store).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.data.quality import is_quality_vi_sentence, split_sentences
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]


def _is_clean_sentence(text: str) -> bool:
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return False
    if "FxPro" in text or "http://" in text or "https://" in text:
        return False
    words = text.split()
    if len(words) < 6 or len(words) > 40:
        return False
    caps_run = 0
    for w in words:
        if len(w) > 1 and w[0].isupper() and w[1:].islower():
            caps_run += 1
            if caps_run >= 4:
                return False
        else:
            caps_run = 0
    return True


def _conv_sentences() -> list[str]:
    out: list[str] = []
    for name, key in [("stories_raw.jsonl", "text"), ("chat_raw.jsonl", None)]:
        p = _ROOT / "data/conv" / name
        if not p.exists():
            continue
        for line in p.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            texts = [row.get(key) or ""] if key else (row.get("turns") or [])
            for t in texts:
                for s in split_sentences(t or ""):
                    if is_quality_vi_sentence(s, max_words=40) and _is_clean_sentence(s):
                        out.append(s)
    return out


def _essay_sentences() -> list[str]:
    out: list[str] = []
    p = _ROOT / "data/real_vi/essays.jsonl"
    if p.exists():
        for line in p.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for s in row.get("sentences") or []:
                s = (s or "").strip().lstrip("- ")
                if is_quality_vi_sentence(s, max_words=40) and _is_clean_sentence(s):
                    out.append(s)
    p2 = _ROOT / "data/real_vi/statements.jsonl"
    if p2.exists():
        for line in p2.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            s = (row.get("text") or "").strip()
            if is_quality_vi_sentence(s, max_words=40) and _is_clean_sentence(s):
                out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4_convpool.pkl")
    ap.add_argument("--out", default="checkpoints/network_v4_full.pkl")
    ap.add_argument("--max-templates", type=int, default=15000)
    args = ap.parse_args()
    log = get_run_logger("build_essay_opener_pool")

    state, cache = load_checkpoint(args.checkpoint, logger=log)
    vocab = state.vocab
    existing = set(getattr(state.parents, "_key_to_idx", {}).keys()) if state.parents else set()

    with log.phase("lọc câu sidecar pattern_templates"):
        raw = _conv_sentences() + _essay_sentences()
        log.log(f"câu thô: {len(raw)}")
        known = set(vocab._token_to_id.keys())
        seen: set[tuple[int, ...]] = set()
        picked: list[list[int]] = []
        skipped_oov = skipped_dup = 0
        for s in raw:
            toks = tokenize_punct(s)
            if not toks or not all(t in known for t in toks):
                skipped_oov += 1
                continue
            ids = tuple(vocab.encode_tokens(toks))
            if len(ids) < 6 or ids in seen or ids in existing:
                if ids in existing:
                    skipped_dup += 1
                continue
            seen.add(ids)
            picked.append(list(ids))
            if len(picked) >= args.max_templates:
                break
        log.log(f"chọn {len(picked)} mẫu sidecar (bỏ OOV: {skipped_oov}, trùng store: {skipped_dup})")

    state.pattern_templates = picked
    log.log(f"parents giữ nguyên {len(state.parents) if state.parents else 0}; templates={len(picked)}")

    with log.phase("save"):
        save_checkpoint(state, args.out, cache)
    log.log(f"XONG → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
