#!/usr/bin/env python3
"""
Train duy nhất — nạp text_corpus (mỗi dòng jsonl = một chuỗi) → graph.

Chuẩn bị corpus (một lần):
  PYTHONPATH=src python scripts/normalize_corpus.py

Train from scratch:
  PYTHONPATH=src python scripts/train.py \\
    --manifest config/train_v10.yaml \\
    --out checkpoints/network_v10.pkl

Train tiếp thêm dataset:
  PYTHONPATH=src python scripts/train.py \\
    --checkpoint checkpoints/network_v10.pkl \\
    --out checkpoints/network_v10.pkl \\
    --dataset data/text_corpus/histwiki_para.jsonl \\
    --priority 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _fresh_vocab(bpe_path: str | None, log):
    """TK_041 — vocab BPE nếu có merge table, ngược lại vocab legacy (`.split()`).

    `--bpe ""` để train theo tokenizer cũ (so baseline). Mặc định = `checkpoints/bpe_v1.json`
    nếu tồn tại: đường mặc định phải là đường mới, nếu không nó sẽ không bao giờ được đo.
    """
    from nn_aivie.core.tokenizer import BPETokenizer
    from nn_aivie.core.vocab import Vocabulary

    if not bpe_path:
        log.log("▶ vocab LEGACY (.split() + FOLD_STRIP) — TK_041 tắt")
        return Vocabulary()
    p = Path(bpe_path)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        raise SystemExit(
            f"không có merge table {p} — chạy `scripts/train_bpe.py` trước, "
            f"hoặc `--bpe ''` để train theo tokenizer cũ"
        )
    tok = BPETokenizer.load(p)
    log.log(f"▶ vocab BPE {p.name}: {len(tok):,} token · fingerprint {tok.fingerprint()}")
    return Vocabulary(tok)


def main() -> int:
    ap = argparse.ArgumentParser(description="NN-AIVIE train — một script, chỉ text jsonl")
    ap.add_argument("--checkpoint", default=None, help="checkpoint nguồn (bỏ qua = graph trống)")
    ap.add_argument("--out", required=True, help="checkpoint đích")
    ap.add_argument("--manifest", default=None, help="YAML danh sách path + priority")
    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="thêm một file text_corpus (lặp được)",
    )
    ap.add_argument("--priority", type=float, default=None, help="cho --dataset (mặc định 0.6)")
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument(
        "--bpe",
        default="checkpoints/bpe_v1.json",
        help="TK_041 merge table BPE ('' = vocab legacy .split()+FOLD_STRIP)",
    )
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--max-texts", type=int, default=0, help="giới hạn text/file (0=không)")
    ap.add_argument("--bake", action="store_true", help="bake skip spans + seq index khi save")
    ap.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="log từng bài: dataset i/n, cha#, slots, cụm → out/train_progress.jsonl",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="verbose: log mỗi N bài (0=mặc định 1; file lớn nên 10–50)",
    )
    ap.add_argument(
        "--progress-jsonl",
        default="out/train_progress.jsonl",
        help="file JSONL tiến trình (tắt: --progress-jsonl '')",
    )
    ap.add_argument(
        "--sentence-parents",
        action="store_true",
        help="TK_045: tách câu lúc ingest — 1 câu=1 parent + episode theo bài",
    )
    ap.add_argument("--sentence-min-content", type=int, default=3)
    ap.add_argument("--sentence-max-tokens", type=int, default=64)
    ap.add_argument("--sentence-max-spans", type=int, default=32)
    args = ap.parse_args()

    from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
    from nn_aivie.core.state import GraphState
    from nn_aivie.training.dataset_spec import DEFAULT_MANIFEST, DEFAULT_PRIORITY, DatasetSpec
    from nn_aivie.training.pipeline import run_manifest
    from nn_aivie.util.progress import get_run_logger

    log = get_run_logger("train")
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    specs: list[DatasetSpec] = []
    if args.manifest:
        mpath = Path(args.manifest)
        if not mpath.is_absolute():
            mpath = _ROOT / mpath
        specs.extend(DatasetSpec.load_manifest(mpath))
    pri = args.priority if args.priority is not None else DEFAULT_PRIORITY
    for ds in args.dataset:
        p = Path(ds)
        if not p.is_absolute():
            p = _ROOT / p
        specs.append(
            DatasetSpec(
                path=str(p.relative_to(_ROOT)) if p.is_relative_to(_ROOT) else str(p),
                priority=pri,
                eta=args.eta,
                max_texts=args.max_texts,
            )
        )
    if not specs:
        log.log(f"không có dataset — dùng manifest mặc định {DEFAULT_MANIFEST.name}")
        specs = DatasetSpec.load_manifest(DEFAULT_MANIFEST)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_absolute():
            ckpt = _ROOT / ckpt
        with log.phase("load_checkpoint"):
            state, cache = load_checkpoint(ckpt, logger=log)
    else:
        log.log("▶ graph trống (fresh)")
        state = GraphState(vocab=_fresh_vocab(args.bpe, log))
        cache = None

    progress_path = args.progress_jsonl.strip() if args.verbose else None
    pp: Path | None = None
    if progress_path:
        pp = Path(progress_path)
        if not pp.is_absolute():
            pp = _ROOT / pp
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text("", encoding="utf-8")
        log.log(f"▶ progress → {pp}")

    with log.phase("train"):
        if args.sentence_parents:
            log.log(
                f"▶ sentence_parents ON "
                f"(min_content={args.sentence_min_content} "
                f"max_tokens={args.sentence_max_tokens} "
                f"max_spans={args.sentence_max_spans})"
            )
        totals = run_manifest(
            state, specs, repo_root=_ROOT, logger=log, dim=args.dim,
            verbose=args.verbose,
            log_every=args.log_every,
            progress_jsonl=pp if progress_path else None,
            sentence_parents=bool(args.sentence_parents),
            sentence_min_content=int(args.sentence_min_content),
            sentence_max_tokens=int(args.sentence_max_tokens),
            sentence_max_spans=int(args.sentence_max_spans),
        )
    log.log(f"ingested: {totals}")
    if state.parents is not None:
        log.log(
            f"parents={len(state.parents.parents)} "
            f"episodes={len(getattr(state.parents, 'episodes', []) or [])}"
        )

    with log.phase("save"):
        save_checkpoint(state, out, cache, bake_derived=args.bake, logger=log)
    log.log(f"✓ saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
