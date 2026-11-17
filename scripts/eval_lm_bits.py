#!/usr/bin/env python3
"""
EVAL LM v2 trên held-out — bits/token + bits/byte + (tuỳ chọn) sinh mẫu decode-polish.

    PYTHONPATH=src python -u scripts/eval_lm_bits.py --ckpt out/lm_v2_v11_qa.pt \\
        --holdout data/sft/qa_heldout.jsonl

## Vì sao script RIÊNG (không dùng eval_bits.py)

`eval_bits.py` nạp checkpoint **ĐỒ THỊ** (`GraphState.load` → pickle) — chĩa nó vào LM `.pt`
(dict cfg+state_dict) = `UnpicklingError` (M4 đã vấp, xem eval_v10_fp32.txt). Đây là thước cho
**LM nơ-ron**, CÙNG công thức bits (`-log2 P(gold)` mọi vị trí) + CÙNG `load_heldout`/seed để so
được với số trong `train_neural_lm.py`.

Dùng để lập BẢNG so 2×2 (v10 vs v11 · essays vs qa_heldout) — trả lời "thêm data (v11 180M) có
NÂNG chất lượng QA không" trên ruler MIỀN ĐÍCH, tách khỏi lệch-miền của essays.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from nn_aivie.core.tokenizer import BPETokenizer, DAP_LABEL, SPECIAL_EOS, normalize

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from eval_bits import load_heldout  # noqa: E402 — CÙNG held-out/seed/công thức


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", required=True, help="checkpoint v2 (dict cfg+state_dict)")
    ap.add_argument("--holdout", default="data/sft/qa_heldout.jsonl")
    ap.add_argument("--n-holdout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--answer-only", action="store_true",
                    help="chỉ chấm token SAU `Đáp:` (né confound context-leak: v11/v12 thuộc lòng "
                         "context wiki ⇒ bits toàn-đơn-vị tụt GIẢ). Dùng khi so cross-model.")
    ap.add_argument("--gen-samples", type=int, default=6, help="số prompt sinh đọc tay (0=bỏ)")
    ap.add_argument("--gen-rep-penalty", type=float, default=1.2)
    ap.add_argument("--gen-no-repeat", type=int, default=3)
    ap.add_argument("--gen-temp", type=float, default=0.0)
    ap.add_argument("--gen-top-p", type=float, default=1.0)
    args = ap.parse_args()

    import torch

    from nn_aivie.core.neural_lm import NeuralLM

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok.tokens)
    lm = NeuralLM.load(_ROOT / args.ckpt, V)  # v2: đọc cfg tường minh, strict load
    model, ctx, dev = lm.model, lm.ctx, lm.device
    is_lp = getattr(model, "returns_logprob", False)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"ckpt {args.ckpt}: {n_par/1e6:.1f}M · ctx {ctx} · device={dev}")

    texts = load_heldout(_ROOT / args.holdout, args.n_holdout, args.seed)
    heldout = [tok.encode(t) for t in texts]
    nbytes = sum(len(normalize(t).encode("utf-8")) for t in texts)
    print(f"held-out {Path(args.holdout).name}: {len(texts)} đơn vị · {nbytes:,} byte · "
          f"{sum(len(i) for i in heldout):,} token")

    dap_id = tok.token_to_id[DAP_LABEL]
    s = 0.0
    n = 0
    n_skip = 0
    model.eval()
    with torch.no_grad():
        for ids in heldout:
            if len(ids) < 2:
                continue
            m = min(len(ids), ctx)
            start = 0  # chấm từ vị trí dự đoán token[start+1]
            if args.answer_only:
                head = ids[:m]
                if dap_id not in head:
                    n_skip += 1
                    continue
                start = len(head) - 1 - head[::-1].index(dap_id)  # index `Đáp:` cuối cùng
            x = torch.tensor(ids[:ctx], dtype=torch.long, device=dev)[None]
            out = model(x)[0].float()
            lp = out if is_lp else torch.log_softmax(out, -1)
            rows = torch.arange(start, m - 1, device=dev)
            cols = torch.tensor(ids[start + 1:m], dtype=torch.long, device=dev)
            s += (-lp[rows, cols].sum().item()) / math.log(2)
            n += m - 1 - start
    if args.answer_only:
        print(f"→ {s/max(1,n):.3f} bits/token (CHỈ đáp, sau `Đáp:`)  "
              f"({n:,} vị trí · bỏ {n_skip} đơn vị thiếu Đáp:)")
    else:
        print(f"→ {s/max(1,n):.3f} bits/token · {s/nbytes:.3f} bits/byte  ({n:,} vị trí)")

    if args.gen_samples > 0:
        eos_id = tok.token_to_id[SPECIAL_EOS]
        prompts = [
            "Hỏi: Thủ đô của Việt Nam là gì",
            "Hỏi: Ai là người sáng lập nhà Lý",
            "Hỏi: Mặt trời mọc ở hướng nào",
            "Hỏi: Nước sôi ở nhiệt độ bao nhiêu",
            "Hỏi: Ai trồng lúa",
            "Hỏi: Quốc hội do ai bầu",
        ][: args.gen_samples]
        print("── sinh (decode polish: rep-penalty + no-repeat-ngram) ──")
        for p in prompts:
            ids = lm.generate(
                tok.encode(f"{p} {DAP_LABEL}"), 60, eos_id=eos_id,
                temperature=args.gen_temp, top_p=args.gen_top_p,
                repetition_penalty=args.gen_rep_penalty, no_repeat_ngram=args.gen_no_repeat,
                seed=args.seed,
            )
            print(f"   {p}\n     → {tok.decode(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
