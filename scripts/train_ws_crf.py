#!/usr/bin/env python3
"""
TK_034 — Train WS-CRF (B-W/I-W) inside nn-aivie (self-contained).

Input format (simple, VLSP-style):
  Each line is a sentence; each word may contain syllables joined by '_' (e.g. 'Bác_sĩ').
  We expand to syllable tokens and label with B-W/I-W.

Outputs a model directory containing:
  - model.crfsuite
  - dictionary_lower.txt
  - meta.json

Install deps:
  pip install -e .[ws]

Example:
  python scripts/train_ws_crf.py --train data/ws/train.txt --test data/ws/test.txt --out models/ws_crf_nn_aivie_20260708
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pycrfsuite

from nn_aivie.ws_crf.ws_corpus import read_tagged_file
from nn_aivie.ws_crf.ws_features import WSFeatureConfig, build_ws_features

ROOT = Path(__file__).resolve().parents[1]


def _spans(tags: list[str]) -> set[tuple[int, int]]:
    """Chuyển B-W/I-W thành tập span (start, end-exclusive) của từng word."""
    spans: set[tuple[int, int]] = set()
    start = None
    for i, t in enumerate(tags):
        if t == "B-W":
            if start is not None:
                spans.add((start, i))
            start = i
        elif t != "I-W":
            if start is not None:
                spans.add((start, i))
            start = None
    if start is not None:
        spans.add((start, len(tags)))
    return spans


def _load_dictionary_lower(dict_raw_jsonl: Path) -> set[str]:
    """Build lowercase dictionary from HF raw jsonl (word field)."""
    out: set[str] = set()
    if not dict_raw_jsonl.exists():
        return out
    with dict_raw_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                w = json.loads(line).get("word", "")
            except Exception:
                continue
            w = str(w).strip().lower()
            if w:
                out.add(w)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train WS-CRF (underthesea-style templates) in nn-aivie")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--out", required=True, help="Output model directory")
    ap.add_argument("--dict-raw", default="data/dict/vi_dict_raw.jsonl")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--c1", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1e-3)
    ap.add_argument("--max-iterations", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-train", type=int, default=0, help="If >0, sample N sentences for quick dev")
    args = ap.parse_args()

    random.seed(int(args.seed))

    train_path = ROOT / args.train
    test_path = (ROOT / args.test) if args.test else None
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    dict_lower = _load_dictionary_lower(ROOT / args.dict_raw)
    (out_dir / "dictionary_lower.txt").write_text("\n".join(sorted(dict_lower)), "utf-8")

    cfg = WSFeatureConfig(window=int(args.window))
    train = read_tagged_file(train_path)
    if args.sample_train and int(args.sample_train) < len(train):
        random.shuffle(train)
        train = train[: int(args.sample_train)]

    test = read_tagged_file(test_path) if test_path else []

    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.set_params(
        {
            "c1": float(args.c1),
            "c2": float(args.c2),
            "max_iterations": int(args.max_iterations),
            "feature.possible_transitions": True,
            "feature.possible_states": True,
        }
    )

    for sent in train:
        xseq = [
            build_ws_features(sent.tokens, i, dictionary_lower=dict_lower, cfg=cfg)
            for i in range(len(sent.tokens))
        ]
        trainer.append(xseq, sent.tags)

    model_path = out_dir / "model.crfsuite"
    trainer.train(str(model_path))

    eval_stats: dict[str, float] = {}
    if test:
        tagger = pycrfsuite.Tagger()
        tagger.open(str(model_path))
        tok_ok = tok_all = 0
        n_gold = n_pred = n_correct = 0
        for sent in test:
            xseq = [
                build_ws_features(sent.tokens, i, dictionary_lower=dict_lower, cfg=cfg)
                for i in range(len(sent.tokens))
            ]
            pred = tagger.tag(xseq)
            tok_all += len(pred)
            tok_ok += sum(1 for a, b in zip(pred, sent.tags) if a == b)
            # word-level F1: so sánh span (start, end) của các word
            gold_spans = _spans(sent.tags)
            pred_spans = _spans(pred)
            n_gold += len(gold_spans)
            n_pred += len(pred_spans)
            n_correct += len(gold_spans & pred_spans)
        prec = n_correct / n_pred if n_pred else 0.0
        rec = n_correct / n_gold if n_gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        eval_stats = {
            "token_accuracy": round(tok_ok / tok_all, 4) if tok_all else 0.0,
            "word_precision": round(prec, 4),
            "word_recall": round(rec, 4),
            "word_f1": round(f1, 4),
        }
        print(f"eval: {eval_stats}")

    meta = {
        "model_file": "model.crfsuite",
        "dictionary_file": "dictionary_lower.txt",
        "window": cfg.window,
        "train_sentences": len(train),
        "test_sentences": len(test),
        "params": {
            "c1": float(args.c1),
            "c2": float(args.c2),
            "max_iterations": int(args.max_iterations),
        },
        "eval": eval_stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(f"✓ saved model → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

