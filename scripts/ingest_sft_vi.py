#!/usr/bin/env python3
"""
SFT DATA — kéo nguồn mở chất lượng về data/sft/ (TK_042 nhánh SFT, 2026-07-20).

    PYTHONPATH=src python -u scripts/ingest_sft_vi.py            # cả hai nguồn
    PYTHONPATH=src python -u scripts/ingest_sft_vi.py --source viquad --max 30000

## Nguồn (khảo sát + đọc mẫu 2026-07-20)

| repo | cỡ | vì sao chọn |
|---|---|---|
| bkai-foundation-models/vi-alpaca | 50,006 | sinh TRỰC TIẾP tiếng Việt (không phải dịch) — chất lượng nhất |
| taidng/UIT-ViQuAD2.0 | 28,454 | QA NGƯỜI VIẾT (UIT, wiki VI) — khớp miền history/wiki corpus nhà |

BỎ OpenOrca-Viet (121k) + Bactrian-X (67k): văn dịch (translationese) — với LM 42M,
chất lượng > số lượng; nhét văn dịch chỉ kéo văn phong xuống.

## Hai chốt thiết kế (ghi để khỏi vấp lại)

1. **LEAKAGE — `data/real_vi/essays.jsonl` rút TỪ vi-alpaca** và là HELD-OUT của
   `eval_bits.py` (mốc 5.594 + guard quên của SFT). Khoá dẫn xuất là `instruction`
   (xem `ingest_hf_vi.ingest_vi_alpaca`) → LỌC mọi row vi-alpaca có instruction trùng
   essays.jsonl. Sót = tự chấm điểm trên đề đã lộ.
2. **ViQuAD giữ CONTEXT trong prompt** (reading-comprehension, bỏ ~12k câu is_impossible).
   Closed-book trên fact mô hình chưa từng thấy = TRAIN NÓ BỊA — đúng bệnh "bịa mạch lạc"
   G15/§7.17d đã đo. RC-style dạy trích từ ngữ cảnh, không ép đoán.

Output: data/sft/sft_train.jsonl + sft_val.jsonl (~1%, stride cố định — tái lập được),
mỗi dòng {"prompt": "...Hỏi: <q>", "answer": "<a>", "source": ...}. Prompt KHÔNG chứa
"Đáp:" — script SFT tự nối " Đáp: " (special nguyên tử, seam-safe vì pre-tokenizer tách
tại special — TK_041).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_hf_vi import ALPACA_REPO, _hf_rows, _load_existing  # noqa: E402

from nn_aivie.data.quality import (  # noqa: E402
    normalize_ws,
    normalize_ws_keep_newlines,
    vi_diacritic_ratio,
)

VIQUAD_REPO = "taidng/UIT-ViQuAD2.0"
OUT_DIR = ROOT / "data" / "sft"
ESSAYS = ROOT / "data" / "real_vi" / "essays.jsonl"
VAL_STRIDE = 100  # item thứ 100k → val (~1%) — deterministic, không cần seed


def _viet_ok(text: str, min_ratio: float = 0.15) -> bool:
    """Gate nhẹ: đủ dấu tiếng Việt (chặn row tiếng Anh/code lọt). Ngưỡng thấp vì
    đáp hợp lệ có thể nhiều số/danh từ riêng."""
    t = text[:400]
    return bool(t) and vi_diacritic_ratio(t) >= min_ratio


def ingest_alpaca(max_n: int, writers, log) -> tuple[int, int]:
    leaked = _load_existing(ESSAYS, "instruction")
    if not leaked:
        # essays.jsonl là held-out của eval_bits — thiếu nó thì KHÔNG lọc được leakage.
        raise SystemExit(f"✗ {ESSAYS} không có/rỗng — cần nó để lọc leakage (đọc docstring)")
    log(f"  leakage filter: {len(leaked)} instruction từ {ESSAYS.name}")
    kept = skipped_leak = 0
    seen: set[str] = set()
    fetched = 0

    def _consume_row(row: dict) -> None:
        nonlocal kept, skipped_leak
        instr = normalize_ws(row.get("instruction", "") or "")
        extra = normalize_ws(row.get("input", "") or "")
        out = normalize_ws_keep_newlines(row.get("output", "") or "")
        if not instr or not out or instr in seen:
            return
        if instr in leaked:
            skipped_leak += 1
            return
        if not _viet_ok(instr) or not _viet_ok(out):
            return
        if len(out.split()) > 400:  # quá dài cho ctx 512 sau khi cộng prompt — bỏ, không cắt
            return
        seen.add(instr)
        q = f"{instr} {extra}".strip()
        writers(kept, {"prompt": f"Hỏi: {q}", "answer": out, "source": "vi_alpaca"})
        kept += 1

    for _idx, row in _hf_rows(ALPACA_REPO, max_rows=max_n):
        fetched += 1
        _consume_row(row)

    if fetched == 0:
        log("  ⚠ /rows trả rỗng cho vi-alpaca — fallback sang load_dataset(streaming)")
        from datasets import load_dataset

        for i, row in enumerate(load_dataset(ALPACA_REPO, split="train", streaming=True)):
            if i >= max_n:
                break
            _consume_row(row)
    return kept, skipped_leak


def ingest_viquad(max_n: int, writers, log, max_per_context: int = 2) -> tuple[int, int]:
    kept = skipped_imp = 0
    seen: set[str] = set()
    ctx_kept: dict[str, int] = {}
    fetched = 0

    def _consume_row(row: dict) -> None:
        nonlocal kept, skipped_imp
        if row.get("is_impossible"):
            skipped_imp += 1
            return
        q = normalize_ws(row.get("question", "") or "")
        ctx = normalize_ws_keep_newlines(row.get("context", "") or "")
        ans = row.get("answers") or {}
        a = normalize_ws((ans.get("text") or [""])[0] if isinstance(ans, dict) else "")
        if not q or not a or not ctx:
            return
        key = f"{q}|||{a}"
        if key in seen:
            return
        if not _viet_ok(ctx):
            return
        if len(ctx.split()) > 350:  # context + QA phải lọt ctx 512 token (fertility 1.195)
            return
        # ViQuAD có nhiều câu hỏi trên cùng một context dài; giữ quá nhiều sẽ làm SFT lặp nặng.
        # Giới hạn số QA/context để giảm overfit theo đoạn văn lặp nguyên khối.
        ckey = normalize_ws(ctx)
        if ctx_kept.get(ckey, 0) >= max_per_context:
            return
        seen.add(key)
        ctx_kept[ckey] = ctx_kept.get(ckey, 0) + 1
        writers(kept, {"prompt": f"{ctx}\nHỏi: {q}", "answer": a, "source": "viquad"})
        kept += 1

    for _idx, row in _hf_rows(VIQUAD_REPO, max_rows=max_n):
        fetched += 1
        _consume_row(row)

    # datasets-server /rows đôi lúc trả rỗng (rate-limit/network); fallback streaming để khỏi
    # ghi ra dataset trống và tưởng ViQuAD = 0 thật.
    if fetched == 0:
        log("  ⚠ /rows trả rỗng cho ViQuAD — fallback sang load_dataset(streaming)")
        from datasets import load_dataset

        for i, row in enumerate(load_dataset(VIQUAD_REPO, split="train", streaming=True)):
            if i >= max_n:
                break
            _consume_row(row)
    return kept, skipped_imp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source", choices=["all", "vi_alpaca", "viquad"], default="all")
    ap.add_argument("--max", type=int, default=60000, help="số row QUÉT tối đa mỗi nguồn")
    ap.add_argument(
        "--viquad-max-per-context",
        type=int,
        default=1,
        help="giới hạn số QA giữ trên mỗi context ViQuAD để giảm lặp prompt",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_p = OUT_DIR / "sft_train.jsonl"
    val_p = OUT_DIR / "sft_val.jsonl"

    def log(m: str) -> None:
        print(m, flush=True)

    with train_p.open("w", encoding="utf-8") as ftr, val_p.open("w", encoding="utf-8") as fva:

        def writers(i: int, obj: dict) -> None:
            f = fva if i % VAL_STRIDE == 0 else ftr
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if args.source in ("all", "vi_alpaca"):
            log(f"── vi-alpaca ({ALPACA_REPO}) ──")
            k, leak = ingest_alpaca(args.max, writers, log)
            log(f"  giữ {k:,} · LOẠI {leak} trùng held-out essays (leakage — phải >0 mới đúng)")
            if leak == 0:
                log("  ⚠ leak=0 ĐÁNG NGỜ: essays.jsonl rút từ chính repo này — kiểm tra lại filter")
        if args.source in ("all", "viquad"):
            log(f"── ViQuAD 2.0 ({VIQUAD_REPO}) ──")
            k, imp = ingest_viquad(
                args.max,
                writers,
                log,
                max_per_context=max(1, args.viquad_max_per_context),
            )
            log(f"  giữ {k:,} · bỏ {imp:,} is_impossible")

    n_tr = sum(1 for _ in train_p.open(encoding="utf-8"))
    n_va = sum(1 for _ in val_p.open(encoding="utf-8"))
    log(f"→ {train_p}: {n_tr:,} · {val_p}: {n_va:,} (stride {VAL_STRIDE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
