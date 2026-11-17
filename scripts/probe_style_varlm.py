#!/usr/bin/env python3
"""
PROBE VĂN-PHONG: VarLM (biến-bậc) vs bigram trên đường COMPOSED — local-style trong graph? (cho M4).

    ⚠️ CHẠY M4. PYTHONPATH=src python -u scripts/probe_style_varlm.py \
        --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_varlm.log

## Câu hỏi (bàn 2026-07-22: "học văn phong không cần LM?")

Style = mô-hình-chuỗi. Graph ĐÃ có VarOrderLM (PPM ctx 1..4, `core/varlm.py`) trong kênh micro —
"LM Hebbian nội-tại", KHÔNG backprop (Luật-0). Câu hỏi tầng-0: **VarLM (nhớ cụm dài) có làm đường
COMPOSED mượt hơn bigram không?** Nếu có → higher-order n-gram = local-style, đáng SFT-ingest styled
(tầng 1). Nếu không → local-style cũng cần LM backprop.

Probe A/B mỗi prompt (ép composed: tắt verbatim fallback):
  * BIGRAM  : `ar_varlm_weight=0`  (chỉ W_seq bậc-1)
  * VARLM   : `ar_varlm_weight=1`  (PPM biến-bậc — nhớ cụm dài)
ĐỌC TAY (G15): VARLM có ít salad `. , và`/lặp/cụt hơn, cụm dài tự nhiên hơn? (cần varlm sidecar
`<ckpt>.varlm.pkl` — thiếu = VarLM no-op, hai cột giống nhau → build_varlm.py trước.)
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
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

# prompt CÓ NHÃN style (tác giả chỉ ra: prompt phải chứa "hài hước" thì mới test được conditioning) +
# buộc composed (chủ-đề trộn style → không chép-1-parent). So chủ-đề ĐÃ THẤY (trong styled_sft) vs MỚI.
PROMPTS = [
    "Viết hài hước về Trần Hưng Đạo:",          # chủ-đề CÓ trong styled_sft → test retrieval+local
    "Viết hài hước về Nhà Trần:",                # có trong styled_sft
    "Viết hài hước về Sông Hồng:",               # chủ-đề MỚI (không trong styled) → test transfer
    "Kể một cách tự nhiên về mùa đông Hà Nội:",  # MỚI
    "Viết hài hước về việc đi làm:",             # MỚI
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-file", default="out/probe_style_varlm.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True); logf.write(str(m) + "\n"); logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    has_varlm = getattr(state, "varlm", None) is not None
    log(f"[{time.time()-t0:.0f}s] ✓ loaded · parents {len(state.parents):,} · varlm sidecar={has_varlm}")
    if not has_varlm:
        log("⚠ KHÔNG có varlm sidecar (<ckpt>.varlm.pkl) → VarLM no-op, 2 cột SẼ GIỐNG nhau. "
            "Chạy scripts/build_varlm.py trước.")

    def mk(varlm_w):
        c = HCAConfig()
        c.ar_verbatim_fallback = False        # ép composed (né chép-1-parent)
        c.ar_tail_verbatim_fallback = False
        c.essay_ar_min_cov = 0.0
        c.ar_varlm_weight = varlm_w
        return c

    cfg_bi, cfg_var = mk(0.0), mk(1.0)

    def gen(p, c):
        return decode_surface(state, generate(state, p, max_len=args.max_len, config=c,
                                              seed=args.seed, multi=True))

    log("\n═══ VĂN-PHONG composed: BIGRAM vs VARLM — đọc tay salad/cụm dài ═══")
    for p in PROMPTS:
        log(f"\n▶ {p}")
        log(f"  BIGRAM : {gen(p, cfg_bi)!r}")
        log(f"  VARLM  : {gen(p, cfg_var)!r}")
    log(f"\n[{time.time()-t0:.0f}s] → VARLM ít salad/cụm-dài tự-nhiên hơn BIGRAM ⇒ higher-order = "
        "local-style, đáng SFT-ingest styled (RUNBOOK_STYLE §SFT). Giống nhau ⇒ n-gram không đủ.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
