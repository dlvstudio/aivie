#!/usr/bin/env python3
"""
DIAG quy-trách-nhiệm KÊNH — vì sao token X thắng ở bước kế? (TK_045 Tier-1, cho M4)

    ⚠️ CHẠY M4. PYTHONPATH=src python -u scripts/diag_channel_attrib.py \
        --checkpoint checkpoints/network_v12_sent.pkl --log-file out/diag_channel_attrib.log

## Vì sao (wrong-anchor "Nhà Lý dời đô về" → "Nhà Lý thành lập năm 109")

`matrix_core_logits` trộn ~10 kênh (bag/succ+path/l2r/agenda/h/eos/know/qk/stack). Khi output sai,
KHÔNG biết kênh nào đẩy. Diag in **giá trị RAW từng kênh** cho top-8 ứng viên tại bước kế + kiểm
2 giả thuyết:
  H1: `par_succ.max()==0` (path/succ KHÔNG fire) → bag+fluency tự quyết → cổng bag-midsent là NO-OP,
      phải sửa chỗ khác (act cha / act_floor / best_k≥2 / top_k_parents).
  H2: token sai CÓ succ>0 (cha thật sự "tiếp" bằng nó) → cổng không đủ, cần siết path-lead.
Kết quả chọn hướng vá — đừng đoán.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.matrix_core import (  # noqa: E402
    ensure_parent_bipartite, matrix_core_logits, parent_coactivation_logits,
)

PROMPTS = [
    "Nhà Lý dời đô về",                 # ca GÃY chính
    "Trận Bạch Đằng năm",               # ca gãy 2
    "Ngô Quyền là vị vua sáng lập nhà Ngô, được",   # ca ĐÚNG (control)
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--log-file", default="out/diag_channel_attrib.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True); logf.write(str(m) + "\n"); logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    vocab, ps = state.vocab, state.parents
    sur = vocab.token_of
    cfg = HCAConfig()
    n = len(vocab)
    log(f"[{time.time()-t0:.0f}s] ✓ loaded · parents {len(ps):,}")

    for p in PROMPTS:
        log(f"\n═══ {p!r} ═══")
        hist = list(vocab.encode_prompt(p))
        w0 = int(hist[-1])
        bag, succ, conf, act = parent_coactivation_logits(
            state, hist, vocab_size=n, lam=float(getattr(cfg, "ar_ctx_lambda", 0.85)),
            ctx_len=0, gamma=float(getattr(cfg, "ar_matrix_parent_gamma", 1.5)),
            succ_boost=float(getattr(cfg, "ar_matrix_parent_succ_boost", 3.0)),
            top_k_parents=int(getattr(cfg, "ar_matrix_parent_topk", 48)),
            gidf_hi=float(getattr(cfg, "ar_matrix_parent_gidf_hi", 4.0)),
        )
        log(f"  w0={sur(w0)!r} · conf={conf:.3f} · bag.max={bag.max():.4g} "
            f"· succ+path.max={succ.max():.4g}  ← H1 nếu ≈0")
        # cha đang kích (act) — xem cha ĐÚNG có được kích không
        if act.size:
            top_pi = np.argsort(-act)[:5]
            bi = ensure_parent_bipartite(state)
            log("  top-5 cha (act · emit đầu):")
            for pi in top_pi:
                if act[pi] <= 0:
                    continue
                em = bi.emits[pi][:14] if pi < len(bi.emits) else []
                txt = " ".join(sur(int(t)) for t in em)
                log(f"    act={act[pi]:.4g}  pi={int(pi)}  {txt!r}")
        # logits cuối + phân rã
        logits = matrix_core_logits(state, hist, config=cfg)
        cand = np.argsort(-logits)[: args.top]
        log(f"  TOP-{args.top} ứng viên (logit · bag · succ+path):")
        for j in cand:
            j = int(j)
            log(f"    {sur(j)!r:18} logit={logits[j]:7.3f}  bag={bag[j]:8.4g}  succ={succ[j]:8.4g}")
    log(f"\n[{time.time()-t0:.0f}s] → succ.max≈0 ở ca gãy ⇒ H1 (path không fire): sửa act cha/"
        "act_floor/best_k≥2/top_k. succ>0 ở token SAI ⇒ H2: siết path-lead. Dán log về.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
