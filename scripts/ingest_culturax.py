#!/usr/bin/env python3
"""
CulturaX vi — nền web tiếng Việt lớn cho base fluent (TK_042, sau BƯỚC 10c nếu data thắng).

    # ⚠️ GATED: phải đăng nhập HF + chấp nhận điều khoản TRƯỚC (một lần):
    #   huggingface-cli login          # dán token từ huggingface.co/settings/tokens
    #   → mở huggingface.co/datasets/uonlp/CulturaX bấm "Agree and access"
    PYTHONPATH=src python -u scripts/ingest_culturax.py --max-tokens 500000000

## Vì sao (sau khi 10c chốt "data là đòn bẩy")

viwiki 160M kéo được 9× nhưng vẫn thiếu (42M Chinchilla cần ~840M token SEEN). CulturaX vi =
**55.4 tỷ token** (uonlp/CulturaX, mc4+OSCAR đã khử trùng/lọc ngôn ngữ) — nguồn thật để đưa
base lên fluent. Lấy một lát ~500M (cắt `--max-tokens`) là đủ cho 42M mà không nổ đĩa/thời gian.

## Khác viwiki: web text NHIỄU hơn wiki

* **Gated** → cần `token=True` (đã login). Nếu 401/403 = chưa login/chưa chấp nhận điều khoản.
* Lọc mạnh hơn: `vi_diacritic_ratio ≥ --min-diacritic` (bỏ trang Anh/code/spam), min token,
  khử TRÙNG (hash 200 ký tự đầu), bỏ doc quá dài (cắt — 1 doc mc4 có thể cực dài, không cần).

⚠️ File ~500M token ≈ **~2GB** — KHÔNG push git (như viwiki, ở lại M4). Chỉ push log.
⚠️ KHÔNG trộn held-out (`essays.jsonl`, `qa_heldout.jsonl`) — CulturaX chỉ vào TRAIN.
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
from nn_aivie.data.quality import vi_diacritic_ratio  # noqa: E402


def iter_culturax(log):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("cần `pip install datasets`")
    log("load_dataset uonlp/CulturaX vi (streaming, GATED — cần login + chấp nhận điều khoản) …")
    try:
        ds = load_dataset("uonlp/CulturaX", "vi", split="train", streaming=True, token=True)
    except Exception as e:  # 401/403 = chưa login/chưa accept; tên khác = network
        raise SystemExit(
            f"✗ load CulturaX lỗi: {e}\n"
            "  → `huggingface-cli login` + mở trang dataset bấm 'Agree and access' (đọc docstring)"
        )
    for row in ds:
        yield row.get("text") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--out", default="data/text_corpus/culturax_vi.jsonl")
    ap.add_argument("--min-tokens", type=int, default=64, help="bỏ doc ngắn (stub/spam)")
    ap.add_argument("--max-doc-tokens", type=int, default=4000, help="cắt doc quá dài")
    ap.add_argument("--min-diacritic", type=float, default=0.12, help="bỏ trang Anh/code")
    ap.add_argument("--max-tokens", type=int, default=500_000_000)
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    tok = BPETokenizer.load(_ROOT / args.bpe)
    log(f"tokenizer {args.bpe}: fingerprint {tok.fingerprint()}")

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_tok = n_doc = n_dup = n_stub = n_lang = 0
    with out.open("w", encoding="utf-8") as fo:
        for text in iter_culturax(log):
            text = (text or "").strip()
            if not text:
                continue
            if vi_diacritic_ratio(text[:400]) < args.min_diacritic:
                n_lang += 1
                continue
            h = hashlib.md5(text[:200].encode("utf-8")).hexdigest()
            if h in seen:
                n_dup += 1
                continue
            enc = tok.encode(text)
            if len(enc) < args.min_tokens:
                n_stub += 1
                continue
            if len(enc) > args.max_doc_tokens:  # cắt doc dài — giữ token-budget, không bỏ cả doc
                text = tok.decode(enc[: args.max_doc_tokens])
                enc = enc[: args.max_doc_tokens]
            seen.add(h)
            fo.write(json.dumps(text, ensure_ascii=False) + "\n")
            n_tok += len(enc)
            n_doc += 1
            if n_doc % 20000 == 0:
                log(f"  {n_doc:,} doc · {n_tok/1e6:.1f}M token "
                    f"(dup {n_dup:,} stub {n_stub:,} lang {n_lang:,})")
            if n_tok >= args.max_tokens:
                log(f"  đạt trần --max-tokens {args.max_tokens/1e6:.0f}M — dừng")
                break

    log(f"✓ {out}: {n_doc:,} doc · {n_tok/1e6:.1f}M token "
        f"(bỏ dup {n_dup:,} · stub {n_stub:,} · lang {n_lang:,})")
    log("→ thêm vào config/train_v12.yaml (đã có) rồi bash out/run_culturax.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
