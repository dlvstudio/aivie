#!/usr/bin/env python3
"""
QA HELD-OUT — thước MIỀN ĐÍCH cho các thí nghiệm scale-data (TK_042, 2026-07-20).

    PYTHONPATH=src python -u scripts/build_qa_heldout.py

## Vì sao cần file này

Miền đích tác giả chốt = **QA (Hỏi/Đáp)**, KHÁC miền `essays.jsonl` (instruction vi-alpaca) đang
làm held-out của pretrain. Đo generation-quality-cho-QA trên essays = đo nhầm thước (v11 wiki
được 6.510 trên essays là LỆCH MIỀN, không phải data hại). File này là thước QA cố định, RESERVE
vĩnh viễn — KHÔNG bao giờ đưa vào train (pretrain hay SFT).

## Nguồn (disjoint với train + pretrain — kiểm được)

| nguồn | format | vì sao sạch |
|---|---|---|
| **ViQuAD validation** (3,814) | RC: `{context}\\nHỏi: {q} Đáp: {a}` | split VALIDATION — SFT chỉ dùng TRAIN ⇒ disjoint; khớp phân bố SFT (62% viquad-RC). (⚠️ split TEST GIẤU đáp án — null — nên KHÔNG dùng được) |
| **grade QA val** (31) | closed-book: `Hỏi: {q} Đáp: {a}` | val (ngoài grade train); DEDUP câu lọt `qa_supervised.jsonl` (pretrain) — "Lao động có ích gì" ĐÃ lọt |

⚠️ **CONFOUND context-leak (đọc bằng answer-only):** v11/v12 train trên viwiki (= Wikipedia),
mà ViQuAD context CŨNG là Wikipedia ⇒ context có thể đã thấy lúc train ⇒ bits toàn-đơn-vị của
v11/v12 tụt GIẢ vì thuộc lòng context, không phải QA giỏi hơn. → So cross-model bằng
`eval_lm_bits.py --answer-only` (chỉ chấm token SAU `Đáp:`, context không tính). Mid-training
eval toàn-đơn-vị vẫn OK để chọn best-checkpoint TRONG một run (confound hằng số nội bộ run).

⚠️ Grade val NHỎ (31) + có leakage → phần phụ closed-book; khối lượng đến từ ViQuAD validation.

Output: `data/sft/qa_heldout.jsonl` — mỗi dòng `{"text": ..., "source": ...}`.
Dùng: `train_neural_lm.py --holdout data/sft/qa_heldout.jsonl` (thay/bổ sung essays).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_hf_vi import VIQUAD_REPO, _hf_rows  # noqa: E402 — split="test" (param mới)

from nn_aivie.core.tokenizer import normalize  # noqa: E402
from nn_aivie.data.quality import normalize_ws, normalize_ws_keep_newlines, vi_diacritic_ratio  # noqa: E402

OUT = ROOT / "data" / "sft" / "qa_heldout.jsonl"
QA_SUPERVISED = ROOT / "data" / "text_corpus" / "qa_supervised.jsonl"  # QA trong pretrain
SFT_TRAIN = ROOT / "data" / "sft" / "sft_train.jsonl"


def _pretrain_qa_blob() -> str:
    """Toàn bộ QA pretrain (qa_supervised) chuẩn hoá thành 1 CHUỖI — check leakage bằng SUBSTRING.
    (Set exact-match KHÔNG bắt được: qa_supervised lưu 'Hỏi:…?Đáp:…' đầy đủ, khác mặt chữ held-out
    — đó là vì sao bản trước báo 'loại 0 leak' dù 'Lao động có ích gì' ĐÃ nằm trong pretrain.)"""
    return normalize(QA_SUPERVISED.read_text(encoding="utf-8")) if QA_SUPERVISED.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--viquad-max", type=int, default=250, help="số câu ViQuAD validation giữ")
    ap.add_argument("--scan", type=int, default=3000, help="số row validation QUÉT để lọc")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pre_blob = _pretrain_qa_blob()
    rows = []

    # --- ViQuAD VALIDATION (khối lượng chính, RC — test giấu đáp án nên không dùng được) ---
    seen_q: set[str] = set()
    kept_vq = fetched = 0
    for _idx, row in _hf_rows(VIQUAD_REPO, max_rows=args.scan, split="validation"):
        fetched += 1
        if row.get("is_impossible"):
            continue
        q = normalize_ws(row.get("question", "") or "")
        ctx = normalize_ws_keep_newlines(row.get("context", "") or "")
        ans = row.get("answers") or {}
        a = normalize_ws((ans.get("text") or [""])[0] if isinstance(ans, dict) else "")
        if not q or not a or not ctx or q in seen_q:
            continue
        if vi_diacritic_ratio(ctx[:400]) < 0.15 or len(ctx.split()) > 350:
            continue
        if pre_blob and normalize(q) in pre_blob:  # câu hỏi đã ở pretrain (hiếm với viquad)
            continue
        seen_q.add(q)
        rows.append({"text": f"{ctx}\nHỏi: {q} Đáp: {a}", "source": "viquad_val"})
        kept_vq += 1
        if kept_vq >= args.viquad_max:
            break
    if fetched == 0:
        print("⚠ ViQuAD /rows validation trả rỗng — kiểm mạng / thử lại (KHÔNG ghi thước rỗng)")

    # --- grade QA val (phụ, closed-book on-target, dedup leakage pretrain qua SUBSTRING) ---
    kept_g = leaked = 0
    for f in sorted(ROOT.glob("data/grade_0*/qa/val.jsonl")):
        for line in f.open(encoding="utf-8"):
            o = json.loads(line)
            q, a = o.get("question", ""), o.get("answer", "")
            if not q or not a:
                continue
            if pre_blob and (normalize(q) in pre_blob or normalize(a) in pre_blob):
                leaked += 1
                continue
            rows.append({"text": f"Hỏi: {q} Đáp: {a}", "source": "grade_val"})
            kept_g += 1

    if not rows:
        raise SystemExit("✗ 0 held-out — ViQuAD fetch rỗng và grade val leak hết; KHÔNG ghi")
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"→ {OUT}: {len(rows)} đơn vị (viquad_val {kept_vq} · grade_val {kept_g}, "
          f"loại {leaked} leak pretrain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
