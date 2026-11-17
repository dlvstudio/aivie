#!/usr/bin/env python3
"""TK_032 — so khớp NER underthesea (LOC/PER/ORG) với Entity:title trong graph."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.paths import viz_default_checkpoint  # noqa: E402

CORPUS = _ROOT / "data/real_vi/history_vi.jsonl"
OUT = _ROOT / "out/tk032_ner.log"
NER_LABELS = frozenset({"B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"})


def _load_sentences() -> list[tuple[str, str | None]]:
    rows: list[tuple[str, str | None]] = []
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("statement") or "").strip()
            title = obj.get("title")
            if text:
                rows.append((text, title))
    return rows


def _fold(s: str) -> str:
    return " ".join(s.lower().split())


def _ner_spans(ner_rows: list[tuple]) -> list[tuple[str, str]]:
    """Trả (surface, label) cho mỗi span PER/LOC/ORG."""
    spans: list[tuple[str, str]] = []
    cur_words: list[str] = []
    cur_label = ""
    for word, _pos, _chunk, ner in ner_rows:
        if ner in NER_LABELS:
            if ner.startswith("B-"):
                if cur_words:
                    spans.append((" ".join(cur_words), cur_label))
                cur_words = [word]
                cur_label = ner[2:]
            else:
                cur_words.append(word)
        else:
            if cur_words:
                spans.append((" ".join(cur_words), cur_label))
                cur_words = []
                cur_label = ""
    if cur_words:
        spans.append((" ".join(cur_words), cur_label))
    return spans


def _match_entity(span: str, entities: list[str]) -> str | None:
    sf = _fold(span)
    if len(sf) < 2:
        return None
    best: str | None = None
    best_len = 0
    for ent in entities:
        if len(ent.strip()) < 2:
            continue
        ef = _fold(ent)
        if sf == ef:
            return ent
        if len(sf) >= 3 and sf in ef and len(ef) > best_len:
            best, best_len = ent, len(ef)
        if len(ef) >= 3 and ef in sf and len(ef) > best_len:
            best, best_len = ent, len(ef)
    return best


def main() -> None:
    try:
        from underthesea import ner
    except ImportError as e:
        print(f"SKIP: underthesea not installed ({e})")
        sys.exit(1)

    ckpt = viz_default_checkpoint()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8") as log:
        def w(msg: str = "") -> None:
            print(msg)
            log.write(msg + "\n")

        w(f"checkpoint: {ckpt}")
        if not ckpt.exists():
            w("SKIP: checkpoint missing — no graph Entity: neurons to compare")
            w("XONG")
            print(f"✓ wrote {OUT} (skipped)")
            return

        import pickle

        w("loading vocab from checkpoint (lightweight) …")
        with ckpt.open("rb") as f:
            payload = pickle.load(f)
        vocab_tokens = payload.get("vocab_tokens") or []
        all_entities = [t[7:] for t in vocab_tokens if t.startswith("Entity:")]
        entities = sorted(e for e in all_entities if len(e.strip()) >= 2)
        w(f"Entity: count: {len(entities)} (raw {len(all_entities)}, bỏ {len(all_entities) - len(entities)} ký tự đơn)")
        w("")

        sentences = _load_sentences()
        n_ner = n_match = n_title_match = 0
        hits: list[str] = []
        misses: list[str] = []

        for text, json_title in sentences:
            ner_rows = ner(text)
            spans = _ner_spans(ner_rows)
            if not spans:
                continue
            for span, label in spans:
                n_ner += 1
                ent = _match_entity(span, entities)
                if ent:
                    n_match += 1
                    if len(hits) < 20:
                        hits.append(f"  [{label}] {span!r} → Entity:{ent}")
                elif len(misses) < 20:
                    misses.append(f"  [{label}] {span!r} (json title={json_title!r})")
            if json_title:
                if _match_entity(json_title, [s for s, _ in spans]) or any(
                    _fold(json_title) == _fold(s) for s, _ in spans
                ):
                    n_title_match += 1

        w(f"corpus sentences: {len(sentences)}")
        w(f"ner_spans (PER/LOC/ORG): {n_ner}")
        w(f"spans_matching_Entity: {n_match} ({100 * n_match / n_ner:.1f}%)" if n_ner else "spans_matching_Entity: 0")
        w(f"json_title_in_ner_spans: {n_title_match}/{len(sentences)}")
        w("")
        w("=== HITS (sample) ===")
        for h in hits:
            w(h)
        w("")
        w("=== MISSES (sample) ===")
        for m in misses:
            w(m)
        w("")
        w("XONG")

    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
