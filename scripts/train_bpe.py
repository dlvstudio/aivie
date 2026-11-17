#!/usr/bin/env python3
"""
TK_041 B1 — train merge table BPE trên corpus nhà → artifact GHIM.

    PYTHONPATH=src python -u scripts/train_bpe.py \
        --manifest config/train_v10.yaml --vocab-size 32000 \
        --out checkpoints/bpe_v1.json --log-file out/train_bpe.log

Artifact `checkpoints/bpe_v1.json` là BẤT BIẾN sau khi có checkpoint train theo nó: đổi
merge table = đổi mọi id = checkpoint thành rác LẶNG LẼ (không crash). Nên checkpoint ghim
`fingerprint()` và verify đối chiếu (TK_041 §7.2 mục 2).

Luật 0 sạch: merge table học từ chính data mình, không mượn tokenizer LLM (chốt tác giả).
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from collections import Counter
from pathlib import Path

import yaml

from nn_aivie.core.tokenizer import BPETokenizer, normalize, pre_tokenize
from nn_aivie.util.progress import FileProgressLog


def iter_texts(paths: list[Path], max_bytes: int | None, log) -> list[str]:
    texts: list[str] = []
    total = 0
    for p in paths:
        if not p.exists():
            log(f"⚠ bỏ qua (không có): {p}")
            continue
        n0 = len(texts)
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, str):
                    text = obj
                elif isinstance(obj, dict):
                    text = obj.get("text") or ""
                else:
                    continue
                if not text:
                    continue
                texts.append(text)
                total += len(text.encode())
                if max_bytes and total > max_bytes:
                    log(f"  {p.name}: {len(texts)-n0:,} đơn vị (chạm --max-bytes)")
                    return texts
        log(f"  {p.name}: {len(texts)-n0:,} đơn vị · tổng {total/1e6:.1f} MB")
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("config/train_v10.yaml"))
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/bpe_v1.json"))
    ap.add_argument("--max-bytes", type=int, default=None, help="cắt corpus (dev)")
    ap.add_argument("--min-char-count", type=int, default=2)
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    # also_stdout=False — hàm `log` dưới đây đã in stdout rồi (tránh in đúp).
    logger = FileProgressLog(args.log_file, also_stdout=False) if args.log_file else None
    t0 = time.time()

    def log(msg: str) -> None:
        line = f"[{time.time()-t0:7.1f}s] {msg}"
        print(line, flush=True)
        if logger:
            logger.log(msg)

    spec = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    paths = [Path(d["path"]) for d in spec["datasets"]]
    log(f"manifest {args.manifest} → {len(paths)} dataset")

    texts = iter_texts(paths, args.max_bytes, log)
    nbytes = sum(len(t.encode()) for t in texts)
    log(f"corpus: {len(texts):,} đơn vị · {nbytes/1e6:.1f} MB")

    tok = BPETokenizer.train(
        texts,
        vocab_size=args.vocab_size,
        min_char_count=args.min_char_count,
        progress=log,
    )
    log(f"merge table: {len(tok.merges):,} merge · bề mặt {len(tok):,} token")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    log(f"→ {args.out} · fingerprint {tok.fingerprint()}")

    # ---- acceptance B1: roundtrip + fertility, ĐO NGAY (không tin mà không đo) ----
    n_bpe = n_ws = 0
    bad = 0
    for t in texts:
        ids = tok.encode(t)
        n_bpe += len(ids)
        n_ws += len(t.split())
        if bad < 5 and tok.decode(ids) != normalize(t):
            bad += 1
            log(f"✗ ROUNDTRIP HỎNG: {t[:80]!r}")
    log(f"roundtrip: {'OK — byte-exact toàn corpus' if bad == 0 else f'{bad} HỎNG'}")
    log(
        f"fertility {n_bpe/max(1,n_ws):.3f} tok/từ-ws · "
        f"{nbytes/max(1,n_bpe):.2f} bytes/token · {n_bpe:,} token"
    )

    # R6: fertility TRUNG BÌNH che đuôi phân bố → dump từ hiếm để ĐỌC TAY.
    wf: Counter[str] = Counter()
    for t in texts:
        for w in normalize(t).split():
            wf[w] += 1
    rare = [w for w, c in wf.items() if c == 1][:200]
    worst = sorted(rare, key=lambda w: -len(tok.encode_pieces(w)))[:40]
    log("R6 — 40 từ hiếm tách tệ nhất (ĐỌC TAY, fertility trung bình KHÔNG đủ làm bằng chứng):")
    for w in worst:
        log(f"    {w!r:32} → {tok.encode_pieces(w)}")

    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
