#!/usr/bin/env python3
"""
DIAGNOSTIC focus-bug — vì sao "Trần Hưng Đạo là ai" → focus 'ai' → bài AI/X-quang? (cho M4).

    ⚠️ CHẠY M4. PYTHONPATH=src python -u scripts/probe_focus_bug.py \
        --checkpoint checkpoints/network_v11.pkl --log-file out/probe_focus_bug.log

## Giả thuyết (đo `out/probe_retrieval_compose_v11_fix4.txt`)

`top_focus_tokens` = `entity_register_ids[:k]`. Với "Trần Hưng Đạo là ai" nó chọn 'là'(1.89)'ai'(3.69)
— KHÔNG phải thực-thể. Hai tầng nghi:
  (1) **thực-thể NỔI TIẾNG gidf THẤP**: "Trần Hưng Đạo" xuất hiện khắp corpus → idf thấp → thua từ-hỏi.
  (2) **"ai"(who) FOLD ≡ "AI"(trí-tuệ)**: focus 'ai' → `case_equiv`/`facts_about` kéo bài AI/X-quang.

Probe KHÔNG vá — PHƠI BÀY: mỗi prompt in per-token (gidf, fold_gidf, gdf, vị-trí, FB-opener?,
case-equiv surfaces) + entity_register ranking + focus + facts_about#1. So ca GÃY vs ca ĐÚNG.
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
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import _completion_from_parent, _content_ids  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

PROMPTS = [
    ("Hỏi: Trần Hưng Đạo là ai", "GÃY: focus 'ai' → bài AI/X-quang (fold ai≡AI?)"),
    ("Hỏi: Hà Nội là gì", "GÃY: → Ca trù (focus lệch?)"),
    ("Hỏi: Nhà Trần là gì", "ĐÚNG (control) — focus 'Nhà Trần'"),
    ("Hỏi: Thăng Long là gì", "ĐÚNG (control)"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--log-file", default="out/probe_focus_bug.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True); logf.write(str(m) + "\n"); logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    ps, vocab = state.parents, state.vocab
    cfg = HCAConfig()
    sur = vocab.token_of
    ps.focus_pos_alpha = -1.0 if not getattr(ps, "_vocab_ref", None) else ps.focus_pos_alpha
    fb = getattr(ps, "_fb_layer_ref", None)
    log(f"[{time.time()-t0:.0f}s] ✓ loaded · parents {len(ps):,}")

    def fold_gidf(t):
        try:
            return ps._fold_gidf(t, vocab)
        except Exception:
            return ps._gidf(t)

    def is_fb(t):
        return bool(fb is not None and fb.row(t))

    def caseq(t):
        try:
            eq = ps.case_equiv_ids(t, vocab)
            return " ".join(sorted(sur(x) for x in eq))
        except Exception:
            return sur(t)

    for prompt, note in PROMPTS:
        log(f"\n═══ {prompt}   [{note}] ═══")
        qids = list(vocab.encode_prompt(prompt))
        qc = list(dict.fromkeys(_content_ids(qids, vocab, skip=set())))
        log("  per-token content (gidf · fold_gidf · gdf · FB? · case-equiv):")
        for t in qc:
            log(f"    {sur(t)!r:14} gidf={ps._gidf(t):5.2f} fold={fold_gidf(t):5.2f} "
                f"gdf={ps._gdf.get(t,0):>5} fb={is_fb(t)!s:5} caseq={caseq(t)!r}")
        reg = ps.entity_register_ids(tuple(qc))
        focus = ps.top_focus_tokens(tuple(qc), k=2, vocab=vocab)
        fa = ps.facts_about(tuple(focus), top_k=1, vocab=vocab,
                            theme_weight=cfg.compose_theme_weight)
        fatop = decode_surface(state, list(_completion_from_parent(ps, fa[0])))[:80] if fa else "∅"
        log(f"  entity_register[:5] = {' '.join(repr(sur(t)) for t in reg[:5])}")
        log(f"  FOCUS = {' '.join(repr(sur(t)) for t in focus)}")
        log(f"  facts_about(focus)#1 = {fatop!r}")
    log(f"\n[{time.time()-t0:.0f}s] → GÃY: thực-thể gidf < từ-hỏi? 'ai' caseq chứa 'AI'? "
        "→ điểm vá (demote từ-hỏi / chặn fold chéo-lớp / neo cụm thực-thể).")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
