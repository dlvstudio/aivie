#!/usr/bin/env python3
"""
PROBE MULTI-ZONE (multi-head attention) — zone hoá + Hopfield retrieve fact mỗi zone (cho M4).

    ⚠️ CHẠY M4. PYTHONPATH=src python -u scripts/probe_multizone.py \
        --checkpoint checkpoints/network_v11.pkl --log-file out/probe_multizone.log

## Đo (RUNBOOK_STYLE / bàn multi-zone 2026-07-22)

Gen hiện = retrieve-1-parent + copy (không ghép). Ý tác giả: zone = vùng chú ý → mỗi zone một head,
Hopfield-attention retrieve fact riêng, rồi ghép. `src/nn_aivie/hca/multizone.py`.

Probe KHÔNG wire vào generate — ĐO trước phần graph làm được:
  * Zone hoá query có tách đúng vùng nội-dung không? (Trần Quốc Toản | gen Z | …)
  * Hopfield re-rank (softmax cosine) chọn fact có ĐÚNG/khớp zone hơn facts_about thô không?
  * Zone STYLE ("hài hước") cosine THẤP / rỗng → xác nhận style KHÔNG retrieve (thuộc LM).
So `top-cos` (attention) — cao = fact bám zone; thấp = zone chức-năng/style.
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
from nn_aivie.hca.generate_hebbian import _completion_from_parent  # noqa: E402
from nn_aivie.hca.multizone import zone_retrieve  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

PROMPTS = [
    "Hỏi: So sánh Nhà Lý và Nhà Trần",
    "Hỏi: Kể về Trần Hưng Đạo và trận Bạch Đằng",
    "bằng văn phong hài hước, hãy miêu tả trận Bạch Đằng của Ngô Quyền",
    "Viết một đoạn văn hài hước về Trần Quốc Toản bóp nát quả cam và thế hệ gen Z",
    "kể một câu chuyện buồn về mùa thu Hà Nội",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--gidf-min", type=float, default=4.0, help="ngưỡng entity-gate (cap+gidf)")
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--cos-min", type=float, default=0.5, help="ngưỡng zone CONTENT (top-cos)")
    ap.add_argument("--log-file", default="out/probe_multizone.log")
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
    sur = vocab.token_of
    log(f"[{time.time()-t0:.0f}s] ✓ loaded · parents {len(ps):,}")

    for p in PROMPTS:
        log(f"\n═══ {p} ═══")
        qids = list(vocab.encode_prompt(p))
        zones = zone_retrieve(state, qids, gidf_min=args.gidf_min, beta=args.beta)
        content_zones = [z for z in zones if z.is_content(args.cos_min)]
        log(f"  {len(zones)} zone → {len(content_zones)} CONTENT (top-cos ≥ {args.cos_min}):")
        for z in zones:
            names = " ".join(repr(sur(t)) for t in z.tokens)
            tag = "CONTENT" if z.is_content(args.cos_min) else "bỏ (style/chức-năng)"
            log(f"  ── zone [{tag}] {names}  (gidf≤{z.max_gidf:.1f} · top-cos={z.top_cos:.2f})")
            for pi, cos, w in z.facts:
                txt = decode_surface(state, list(_completion_from_parent(ps, pi)))[:75]
                log(f"       cos={cos:.2f} w={w:.2f} → {txt!r}")
        # ghép NAIVE (chỉ CONTENT zone, nối top-fact) — xem input cho compose ra sao
        stitched = " ".join(
            decode_surface(state, list(_completion_from_parent(ps, z.facts[0][0])))[:70]
            for z in content_zones if z.facts
        )
        log(f"  GHÉP-NAIVE (content zones): {stitched[:220]!r}")
    log(f"\n[{time.time()-t0:.0f}s] → zone CONTENT tách đúng + Hopfield chọn fact khớp? style-zone cos "
        "thấp (xác nhận HOW≠retrieve). Ghép-naive là input cho compose/LM. Dán log về.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
