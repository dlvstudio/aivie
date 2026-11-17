#!/usr/bin/env python3
"""TK_032 — so sánh tokenize khoảng-trắng (nn-aivie) vs underthesea word_tokenize."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.teaching.align import tokenize  # noqa: E402

CORPUS = _ROOT / "data/real_vi/history_vi.jsonl"
OUT = _ROOT / "out/tk032_tokenize.log"
MAX_EXAMPLES = 25


def _load_sentences() -> list[str]:
    texts: list[str] = []
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("statement") or "").strip()
            if text:
                texts.append(text)
    return texts


def main() -> None:
    try:
        from underthesea import word_tokenize as uts_tokenize
    except ImportError as e:
        print(f"SKIP: underthesea not installed ({e})")
        sys.exit(1)

    sentences = _load_sentences()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_ws = n_uts = 0
    n_compounds = 0
    n_diff = 0
    examples: list[str] = []

    with OUT.open("w", encoding="utf-8") as log:
        def w(msg: str = "") -> None:
            print(msg)
            log.write(msg + "\n")

        w(f"corpus: {CORPUS.relative_to(_ROOT)}")
        w(f"sentences: {len(sentences)}")
        w("")

        for text in sentences:
            ws = tokenize(text)
            uts = uts_tokenize(text)
            n_ws += len(ws)
            n_uts += len(uts)
            compounds = [t for t in uts if " " in t]
            n_compounds += len(compounds)
            if ws != uts:
                n_diff += 1
                if len(examples) < MAX_EXAMPLES:
                    examples.append(
                        f"--- diff #{len(examples) + 1} ---\n"
                        f"text: {text}\n"
                        f"ws  ({len(ws)}): {ws}\n"
                        f"uts ({len(uts)}): {uts}\n"
                        f"compounds: {compounds}"
                    )

        avg_ws = n_ws / len(sentences) if sentences else 0.0
        avg_uts = n_uts / len(sentences) if sentences else 0.0

        w("=== STATS ===")
        w(f"sentence_count: {len(sentences)}")
        w(f"total_tokens_ws: {n_ws}")
        w(f"total_tokens_uts: {n_uts}")
        w(f"avg_tokens_ws: {avg_ws:.3f}")
        w(f"avg_tokens_uts: {avg_uts:.3f}")
        w(f"multi_word_compounds_uts: {n_compounds}")
        w(f"sentences_differ: {n_diff} ({100 * n_diff / len(sentences):.1f}%)" if sentences else "sentences_differ: 0")
        w(f"token_reduction: {n_ws - n_uts} ({100 * (n_ws - n_uts) / n_ws:.1f}%)" if n_ws else "token_reduction: 0")
        w("")
        w(f"=== EXAMPLES (max {MAX_EXAMPLES}) ===")
        for ex in examples:
            w(ex)
            w("")
        w("XONG")

    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
