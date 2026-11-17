#!/usr/bin/env python3
"""
TK_042 G0 — THƯỚC THẬT: cross-entropy (bits/token + bits/BYTE) trên held-out.

    PYTHONPATH=src python scripts/eval_bits.py --checkpoint checkpoints/network_mini_bpe.pkl

## Vì sao phải viết lại thay vì vá `eval_lm_ruler.py`

Thước cũ có BA lỗi độc lập, cái thứ ba là chí mạng với TK_042:

1. **`bits` = `log2(rank)`, KHÔNG phải cross-entropy** (`eval/metrics.py:116`). "4.155 bits"
   nghĩa là "token đúng ở hạng ~17.8" — KHÔNG liên quan lượng tin. Nhãn "bits" mời gọi lỗi
   phạm trù: chia cho bytes/token ra "0.83 bits/byte" = giỏi hơn GPT-2 trên tiếng Anh = bất
   khả. (Chính tác giả ticket này đã mắc lỗi đó 2026-07-16.)

2. **Bỏ qua vị trí mô hình THẤT BẠI**: `if not isfinite(scores[gold]): n_oov += 1; continue`.
   Tức chỉ chấm trên tập con mà mô hình đã có câu trả lời → **thiên lệch chọn mẫu**, số đẹp
   một cách giả tạo. Thước ở đây **chấm MỌI vị trí**; mô hình không có xác suất cho gold thì
   phải trả giá bằng bits — đó là SỰ THẬT.

3. **Hạng ⊥ Calibration.** Log TK_031: KN smoothing `bits 1.43→2.69, top1 46%→28%` → kết luận
   "KN âm". Nhưng KN được thiết kế để sửa CALIBRATION, đánh đổi top-1 lấy đuôi phân bố. Thước
   hạng **mù với thứ KN cải thiện**. Mà calibration CHÍNH LÀ thứ việc sinh cần — lấy mẫu là
   lấy từ *phân bố*, không phải từ *thứ hạng*. → v4 ghi "ruler leaky" là nghi đúng, sai lý do.

## Thước này đo gì

`bits/token = -mean log2 P(gold | ngữ cảnh)` — cross-entropy THẬT, mọi vị trí.
`bits/byte  = tổng bits / tổng byte UTF-8 của held-out`  ← **đại lượng DUY NHẤT so được
xuyên TOKENIZER** (TK_041 §6): cùng một held-out, đổi tokenizer, số vẫn so được.

⚠️ **KHÔNG so được xuyên NGÔN NGỮ / xuyên CORPUS.** Tiếng Việt ~5.5 byte/token (dấu UTF-8 tốn
2–3 byte/chữ) → byte tiếng Việt "rẻ" hơn byte tiếng Anh → bits/byte tiếng Việt thấp hơn một
cách hệ thống. So với "GPT-2 tiếng Anh ~1.0 bits/byte" là **LỖI PHẠM TRÙ** (tác giả ticket đã
mắc 2026-07-16, lần thứ ba trong một phiên). Muốn có mốc tuyệt đối: chạy một LM thật trên
**CHÍNH held-out này**. Chưa làm ⇒ **chưa có mốc ngoài** ⇒ dùng thước này để so
BIGRAM vs HẠNG-THẤP vs PREDICTIVE-CODING **trên cùng held-out**, không để tự khen.

Mô hình phải trả về **phân bố hợp lệ, không có số 0** (nội suy về unigram) — nếu không CE = ∞.
Đó không phải mẹo làm đẹp: một LM **buộc** phải gán xác suất > 0 cho mọi thứ, và né điều đó
chính là lỗi #2 của thước cũ.

## Held-out

⚠️ Thước cũ tự thú trong docstring: *"held-out sạch tương đối. Sổ nợ leakage: để có số TUYỆT
ĐỐI sạch cần reserved-shard lúc train"*. Ta làm đúng: `--holdout` trỏ file KHÔNG có trong
manifest train. Mặc định `data/real_vi/essays.jsonl` (vi-alpaca, không nằm trong
`config/train_*.yaml`). `--check-leak` đối chiếu n-gram với corpus train và báo % trùng.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind

_ROOT = Path(__file__).resolve().parents[1]


def load_heldout(path: Path, n: int, seed: int) -> list[str]:
    texts: list[str] = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o, str):
            texts.append(o)
        elif isinstance(o, dict):
            if o.get("text"):
                texts.append(o["text"])
            for s in o.get("sentences", []) or []:
                if s.strip():
                    texts.append(s.strip())
    random.Random(seed).shuffle(texts)
    return texts[:n]


def encode(state, text: str) -> list[int]:
    """Tokenize ĐÚNG như train. TK_041: KHÔNG `.split()` + `encode_tokens` (đẻ id mới,
    phá vocab đóng, và cho chuỗi chưa từng tồn tại lúc train — bug #3 của TK_041)."""
    return state.vocab.encode_prompt(text)


def build_counts(state) -> tuple[dict[int, dict[int, float]], np.ndarray, float]:
    """Bigram + unigram từ W_seq (đồ thị THẬT, không đọc lại corpus)."""
    W = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    V = len(state.vocab)
    uni = np.zeros(V, dtype=np.float64)
    bi: dict[int, dict[int, float]] = {}
    for i, j, w in W.items():
        if i >= V or j >= V:
            continue
        bi.setdefault(i, {})[j] = bi.setdefault(i, {}).get(j, 0.0) + w
        uni[j] += w
    total = float(uni.sum()) or 1.0
    return bi, uni, total


def eval_bits(state, encoded: list[tuple[list[int], int]], lam: float) -> dict:
    """P(next|w0) = lam·P_bigram + (1-lam)·P_unigram  → CE trên MỌI vị trí (không bỏ qua).

    ⚠️ `encoded` phải được tokenize XONG TRƯỚC khi gọi (xem `main`). Bản đầu tính
    `V = len(vocab)` rồi mới encode → trên vocab legacy `encode_prompt` ĐẺ id mới → `gold >= V`
    → **bị bỏ qua**, tức tái lập ĐÚNG lỗi #2 của thước cũ mà file này sinh ra để chống. Từ
    OOV trong held-out là **thất bại THẬT** và phải trả giá bằng bits (sàn add-1), không được
    biến mất khỏi mẫu số.
    """
    bi, uni, total = build_counts(state)
    V = len(state.vocab)
    p_uni = (uni + 1.0) / (total + V)  # add-1 → không bao giờ 0
    p_floor = 1.0 / (total + V)  # id chưa từng thấy (OOV legacy): vẫn phải có xác suất

    sum_bits = 0.0
    n_pos = 0
    n_bytes = 0
    n_zero_bigram = 0
    n_oov = 0
    rank_bits = 0.0  # thước CŨ, tính song song để so
    for ids, nb in encoded:
        if len(ids) < 2:
            continue
        n_bytes += nb
        for i in range(1, len(ids)):
            w0, gold = ids[i - 1], ids[i]
            if gold >= V:  # id ngoài mảng — TÍNH, không bỏ
                sum_bits += -math.log2(p_floor)
                rank_bits += math.log2(V)
                n_pos += 1
                n_oov += 1
                continue
            # OOV THẬT = token chưa từng thấy lúc train (uni=0), KHÔNG phải "id >= V".
            # Bản đầu đếm `gold >= V` → trên legacy `encode_prompt` ĐẺ id mới nên id luôn < V
            # → OOV luôn báo 0.0% = **check không thể kích hoạt**. Bits vẫn đúng (từ lạ nhận
            # sàn add-1) nhưng bộ đếm mù — đúng lớp lỗi im lặng của TK_041 §5.2.
            if uni[gold] == 0:
                n_oov += 1
            row = bi.get(w0)
            if row:
                z = sum(row.values())
                p_bi = row.get(gold, 0.0) / z if z > 0 else 0.0
            else:
                p_bi = 0.0
            if p_bi == 0.0:
                n_zero_bigram += 1
            p = lam * p_bi + (1.0 - lam) * float(p_uni[gold])
            sum_bits += -math.log2(max(p, 1e-12))
            # thước CŨ để đối chiếu: hạng của gold trong bigram row
            if row:
                rank = 1 + sum(1 for v in row.values() if v > row.get(gold, 0.0))
            else:
                rank = V // 2
            rank_bits += math.log2(rank)
            n_pos += 1
    return {
        "bits_per_token": sum_bits / max(1, n_pos),
        "bits_per_byte": sum_bits / max(1, n_bytes),
        "n_pos": n_pos,
        "n_bytes": n_bytes,
        "pct_zero_bigram": 100.0 * n_zero_bigram / max(1, n_pos),
        "pct_oov": 100.0 * n_oov / max(1, n_pos),
        "old_rank_bits": rank_bits / max(1, n_pos),
    }


def check_leak(texts: list[str], train_paths: list[Path], k: int = 8) -> float:
    """% câu held-out có k-gram trùng corpus train → phát hiện rò."""
    grams: set[tuple[str, ...]] = set()
    for p in train_paths:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = o if isinstance(o, str) else (o.get("text") or "")
                w = t.split()
                for i in range(len(w) - k):
                    grams.add(tuple(w[i : i + k]))
    hit = 0
    for t in texts:
        w = t.split()
        if any(tuple(w[i : i + k]) in grams for i in range(max(0, len(w) - k))):
            hit += 1
    return 100.0 * hit / max(1, len(texts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--holdout", default="data/real_vi/essays.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=None, help="None = sweep")
    ap.add_argument("--check-leak", action="store_true")
    args = ap.parse_args()

    texts = load_heldout(_ROOT / args.holdout, args.n, args.seed)
    print(f"held-out: {len(texts)} đơn vị từ {args.holdout}")
    if args.check_leak:
        tr = [
            _ROOT / "data/text_corpus" / n
            for n in ("grade_lessons.jsonl", "history_articles.jsonl", "paragraph_units.jsonl")
        ]
        print(f"⚠ RÒ: {check_leak(texts, tr):.1f}% câu held-out có 8-gram trùng corpus train")

    print(f"load {args.checkpoint} …", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        state, _ = load_checkpoint(args.checkpoint)
    print(f"vocab={len(state.vocab):,} cha={len(state.parents.parents):,}")

    # Encode TOÀN BỘ trước (vocab legacy có thể mọc) → V ổn định khi chấm.
    encoded = [(encode(state, t), len(t.encode("utf-8"))) for t in texts]
    print(f"vocab sau khi encode held-out: {len(state.vocab):,}")

    lams = [args.lam] if args.lam is not None else [0.5, 0.7, 0.85, 0.95, 0.99]
    print(f"\n{'lam':>6} {'bits/token':>11} {'bits/BYTE':>10} {'%bigram=0':>10} {'%OOV':>6} {'(rank cũ)':>10}")
    best = None
    for lam in lams:
        r = eval_bits(state, encoded, lam)
        print(
            f"{lam:>6.2f} {r['bits_per_token']:>11.3f} {r['bits_per_byte']:>10.3f} "
            f"{r['pct_zero_bigram']:>9.1f}% {r['pct_oov']:>5.1f}% {r['old_rank_bits']:>10.3f}"
        )
        if best is None or r["bits_per_byte"] < best[1]["bits_per_byte"]:
            best = (lam, r)
    lam, r = best
    print(f"\n=== BASE-LM (bigram+unigram, KHÔNG agenda) — lam*={lam} ===")
    print(f"  bits/token : {r['bits_per_token']:.3f}")
    print(f"  bits/BYTE  : {r['bits_per_byte']:.3f}   ← đại lượng so được xuyên tokenizer")
    print(f"  vị trí chấm: {r['n_pos']:,} (MỌI vị trí — không bỏ qua ca thất bại)")
    print(f"  bigram = 0 : {r['pct_zero_bigram']:.1f}% vị trí → phải nhờ unigram backoff")
    print(f"  OOV        : {r['pct_oov']:.1f}% vị trí (BPE phải = 0 — vocab đóng)")
    print("\n  ⚠ KHÔNG so số này với bits/byte tiếng Anh (byte tiếng Việt 'rẻ' hơn — lỗi phạm trù).")
    print("    Dùng để so CÁC MÔ HÌNH trên CÙNG held-out: bigram → hạng-thấp (G2) → PC (G4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
