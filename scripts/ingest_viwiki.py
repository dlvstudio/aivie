#!/usr/bin/env python3
"""
TK_042 §7.20b — GỠ NÚT DATA: thêm Wikipedia tiếng Việt vào corpus.

    # cần net + `pip install datasets` (chạy trên M4):
    PYTHONPATH=src python -u scripts/ingest_viwiki.py --max-tokens 160000000

## Vì sao (§7.20a)

42M tham số ở mức Chinchilla cần ~840M token; corpus hiện 19.7M — thiếu ~40×. G19 lần 1 thua
mốc 11.4M vì THIẾU DATA (+ bug Muon LR). viwiki cùng miền với `histwiki` đã dùng, sạch (HF
`wikimedia/wikipedia` đã strip markup) — một nguồn kéo corpus lên ~170M token (~9×).

## Nguồn & làm sạch

* `--source hf` (mặc định): `datasets.load_dataset("wikimedia/wikipedia", "20231101.vi")` —
  plain text, không markup. Nếu tên config đổi, dùng `--hf-config`.
* `--source file --path X.jsonl|.txt`: fallback nếu đã có dump (mỗi dòng = 1 bài text/json).

Lọc: bỏ bài < `--min-tokens` (stub), khử TRÙNG (hash 200 ký tự đầu), cắt tại `--max-tokens`
để kiểm soát ngân sách (thêm quá nhiều wiki làm loãng miền giáo-khoa/QA — giữ tỉ lệ hợp lý).
Ghi `data/text_corpus/viwiki.jsonl` (mỗi dòng = 1 bài, CÙNG format các file corpus khác).

⚠️ KHÔNG trộn vào held-out `data/real_vi/essays.jsonl` — giữ đo G7/G19 so được. viwiki chỉ
vào TRAIN. (essays là nguồn riêng, không phải wiki, nên không lo rò rỉ.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402


def iter_hf(cfg: str, log):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("cần `pip install datasets` cho --source hf (hoặc dùng --source file)")
    log(f"load_dataset wikimedia/wikipedia {cfg} (streaming) …")
    ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
    for row in ds:
        yield row.get("text") or ""


def iter_file(path: Path, log):
    log(f"đọc file {path} …")
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                yield o if isinstance(o, str) else (o.get("text") or "")
            except json.JSONDecodeError:
                yield line  # .txt thô: mỗi dòng một đơn vị


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "file"], default="hf")
    ap.add_argument("--path", default=None, help="cho --source file")
    ap.add_argument("--hf-config", default="20231101.vi")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--out", default="data/text_corpus/viwiki.jsonl")
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=160_000_000)
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    tok = BPETokenizer.load(_ROOT / args.bpe)
    log(f"tokenizer {args.bpe}: fingerprint {tok.fingerprint()}")

    src = (
        iter_hf(args.hf_config, log)
        if args.source == "hf"
        else iter_file(Path(args.path), log)
    )
    if args.source == "file" and not args.path:
        raise SystemExit("--source file cần --path")

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_tok = n_art = n_dup = n_stub = 0
    with out.open("w", encoding="utf-8") as fo:
        for text in src:
            text = (text or "").strip()
            if not text:
                continue
            h = hashlib.md5(text[:200].encode("utf-8")).hexdigest()
            if h in seen:
                n_dup += 1
                continue
            k = len(tok.encode(text))
            if k < args.min_tokens:
                n_stub += 1
                continue
            seen.add(h)
            fo.write(json.dumps(text, ensure_ascii=False) + "\n")
            n_tok += k
            n_art += 1
            if n_art % 5000 == 0:
                log(f"  {n_art:,} bài · {n_tok/1e6:.1f}M token (dup {n_dup:,} stub {n_stub:,})")
            if n_tok >= args.max_tokens:
                log(f"  đạt trần --max-tokens {args.max_tokens/1e6:.0f}M — dừng")
                break

    log(f"✓ {out}: {n_art:,} bài · {n_tok/1e6:.1f}M token (bỏ dup {n_dup:,} · stub {n_stub:,})")
    log("→ thêm vào config/train_v11.yaml (đã có) rồi train --manifest config/train_v11.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
