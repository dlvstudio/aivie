#!/usr/bin/env python3
"""
TK_042 §7.22 — KV CACHE cho v2: sinh nhanh, ĐẦU RA GIỐNG HỆT (đo cả đúng-đắn lẫn tốc độ).

    PYTHONPATH=src python -u scripts/probe_kv_cache.py --ckpt out/lm_v2_42m.pt

## Vì sao (câu hỏi tác giả 2026-07-19: "chạy runtime / sinh nhanh")

Không cache: mỗi token tính LẠI toàn bộ cửa sổ ctx (O(ctx)/token, lặp 99% việc). KV cache:
prefill prompt một lần, decode từng token O(1). Nhanh 5-20× mà KHÔNG đổi một chữ.

## Điều kiện phân định (đúng-đắn TRƯỚC tốc độ)

1. ĐÚNG-ĐẮN (cứng): greedy có-cache PHẢI ra token id GIỐNG HỆT greedy không-cache. Lệch dù 1
   token ⇒ cache SAI (nghi RoPE offset — token mới phải xoay theo vị trí past_len, không phải
   0; xem rope() trong neural_lm.py). Đây là bài kiểm chống lỗi im lặng TK_041 §5.2.
2. TỐC ĐỘ: tokens/giây có-cache vs không-cache, cùng prompt, cùng số token.

Nếu (1) FAIL thì (2) vô nghĩa — dừng, sửa cache. KHÔNG khoe tốc độ khi đầu ra đã khác.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.neural_lm import NeuralLM  # noqa: E402
from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402


def gen_naive(lm, ids0, n):
    """Vòng argmax KHÔNG cache — tính lại cả cửa sổ mỗi bước (đường tham chiếu)."""
    m = lm.model
    m.eval()
    ids = list(ids0)
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor(ids[-lm.ctx :], dtype=torch.long, device=lm.device)[None]
            out = m(x)[0, -1]
            ids.append(int(out.argmax()))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/lm_v2_42m.pt")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok)
    lm = NeuralLM.load(_ROOT / args.ckpt, V)
    print(f"LM {sum(p.numel() for p in lm.model.parameters())/1e6:.1f}M · ctx={lm.ctx} · dev={lm.device}")

    prompts = ["Nhà Trần là", "Hỏi: Thủ đô Việt Nam là gì ? Đáp:", "Lê Lợi khởi nghĩa"]

    print("\n─── 1. ĐÚNG-ĐẮN: greedy có-cache == không-cache? ───")
    all_ok = True
    for p in prompts:
        ids0 = tok.encode(p)
        a = gen_naive(lm, ids0, args.n)
        b = lm.generate_cached(ids0, args.n)
        ok = a == b
        all_ok &= ok
        mark = "✓ GIỐNG HỆT" if ok else "✗ LỆCH"
        print(f"  {p!r:42} {mark}")
        if not ok:
            # chỉ ra token đầu tiên lệch — chẩn đoán
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"    lệch từ vị trí {i}: naive={tok.decode([x])!r} cache={tok.decode([y])!r}")
                    break
    if not all_ok:
        print("\n✗ CACHE SAI — dừng, không đo tốc độ. Nghi RoPE offset (rope() trong neural_lm.py).")
        return 1
    print("  → tất cả GIỐNG HỆT ✓ (cache đúng)")

    print("\n─── 2. TỐC ĐỘ: tokens/giây ───")
    ids0 = tok.encode(prompts[0])
    for _ in range(1):  # warmup
        gen_naive(lm, ids0, 8)
        lm.generate_cached(ids0, 8)
    t0 = time.time(); gen_naive(lm, ids0, args.n); t_naive = time.time() - t0
    t0 = time.time(); lm.generate_cached(ids0, args.n); t_cache = time.time() - t0
    print(f"  KHÔNG cache : {args.n/t_naive:6.1f} tok/s  ({t_naive:.2f}s)")
    print(f"  CÓ cache    : {args.n/t_cache:6.1f} tok/s  ({t_cache:.2f}s)")
    print(f"  ⇒ nhanh {t_naive/t_cache:.1f}× · đầu ra GIỐNG HỆT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
