#!/usr/bin/env python3
"""
DIAGNOSTIC GROUNDING — vì sao retrieval trôi entity (Thăng Long→Gia Long)? (TK_042).

    ⚠️ CHẠY M4 — checkpoint đồ thị 2GB → ~20GB RAM (TK_040).
    PYTHONPATH=src python -u scripts/probe_grounding.py --checkpoint checkpoints/network_v11.pkl

## Vì sao (đo 2026-07-20: composition ✓, grounding ✗ ans_recall 0.28, ent_keep 0.00)

Logic grounding CÓ (best_match: focus + two-tier curated + entity-filter) mà vẫn trôi. Probe
KHÔNG vá — nó PHƠI BÀY chỗ gãy để vá đúng: mỗi prompt in
  1. q_content + gidf (token nào đặc thù)
  2. FOCUS top_focus_tokens chọn gì  ← nếu neo vào token CHUNG ('Long') thay đặc thù ('Thăng') = gãy ở đây
  3. curated top-1 (có fact trong miền không, hay rơi wiki)
  4. allp top-5: score · priority · token QUERY mà cha phủ · text  ← cha SAI thắng nhờ token CHUNG nào
  5. best_match pick cuối

## Đọc (chẩn đoán, quyết cách vá)

* focus = token generic/chung (BPE piece 'Long') ⇒ lỗi TOP_FOCUS (fix: cụm đa-token làm neo).
* cha sai phủ ĐÚNG 1 token va-chạm ('Long' trong Gia Long) ⇒ thiếu gate CỤM (cần Thăng+Long kề).
* curated rỗng/thấp → rơi wiki ⇒ khoảng trống coverage curated (scale-drift, khối wiki át).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.generate_hebbian import _completion_from_parent, _content_ids  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

PROMPTS = [
    ("Hỏi: Thăng Long là gì", "TRÔI: Thăng Long→Gia Long (chung 'Long')"),
    ("Hỏi: Trần Hưng Đạo là ai", "TRÔI: →Lý Chiêu Hoàng (cùng thời Trần)"),
    ("Hỏi: Công an làm gì", "TRÔI: →Chiến dịch/quân sự"),
    ("Hỏi: Nhà Trần là gì", "TRÔI: →Hồ Quý Ly"),
    ("Hỏi: Chính phủ làm gì", "CONTROL trúng ✓"),
    ("Hỏi: Nhà Lý là gì", "CONTROL ~đúng (Lý Thái Tổ)"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    log(f"load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    ps, vocab, E = state.parents, state.vocab, state.embedding
    if ps is None:
        raise SystemExit("✗ thiếu parents")
    sur = vocab.token_of
    log(f"✓ loaded · parents {len(ps):,}")
    # HANG lần trước (3771c5e): treo NGAY sau load, trước prompt đầu. Nghi đường top_focus_tokens
    # MỚI (alpha=0.3 → _fb_layer_ref/fold_gdf) — best_match cố tình ép alpha=-1.0 (legacy đã kiểm,
    # chạy được trong retrieval_compose). Ép legacy toàn cục + LOG TỪNG lời gọi (flush) để lần này
    # hoặc chạy, hoặc chỉ ĐÚNG chỗ treo.
    ps.focus_pos_alpha = -1.0
    log("  focus_pos_alpha ép = -1.0 (đường legacy như best_match)")

    def pinfo(pi, q_content):
        rec = ps.parents[pi]
        covered = [t for t in q_content if t in set(rec.match_ids)]
        cov = " ".join(f"{sur(t)!r}({ps._gidf(t):.1f})" for t in covered)
        txt = decode_surface(state, list(_completion_from_parent(ps, pi)))[:85]
        return f"prio={getattr(rec,'priority',1.0):.2f} phủ[{cov}] · {txt!r}"

    for prompt, note in PROMPTS:
        log(f"\n═══ {prompt}   [{note}] ═══")
        try:
            log("  · encode_prompt …")
            qids = list(vocab.encode_prompt(prompt))
            q_content = list(dict.fromkeys(_content_ids(qids, vocab, skip=set())))
            log("  q_content: " + " ".join(f"{sur(t)!r}({ps._gidf(t):.1f})" for t in q_content))
            log("  · top_focus_tokens …")
            focus = ps.top_focus_tokens(tuple(q_content), k=2, vocab=vocab)
            log("  FOCUS: " + " ".join(f"{sur(t)!r}({ps._gidf(t):.1f})" for t in focus))

            log("  · rank_matches(curated) …")
            curated = ps.rank_matches(qids, vocab, E, min_priority=0.9, top_n=1)
            if curated:
                log(f"  curated#1 (score {curated[0][1]:.3f}): {pinfo(curated[0][0], q_content)}")
            else:
                log("  curated#1: RỖNG → rơi xuống pool wiki")

            log("  · rank_matches(allp) …")
            allp = ps.rank_matches(qids, vocab, E, top_n=args.top)
            log(f"  allp top-{args.top}:")
            for pi, sc in allp:
                log(f"    {sc:.3f} · {pinfo(pi, q_content)}")

            log("  · best_match …")
            pick = ps.best_match(qids, vocab, E)
            log(f"  → best_match PICK: {pinfo(pick[0], q_content)}" if pick else "  → best_match: None")
        except Exception as e:
            import traceback
            log(f"  ✗ LỖI: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    log("\n═══ ĐỌC: FOCUS neo token đặc thù hay CHUNG? cha sai phủ token va-chạm nào? "
        "curated có fact trong miền hay rơi wiki? → quyết vá (cụm-neo / gate-cụm / two-tier). ═══")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
