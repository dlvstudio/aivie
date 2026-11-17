#!/usr/bin/env python3
"""
DIAGNOSTIC văn-phong — style ở đường COMPOSED (A3, KHÔNG chép verbatim) có salad không? (cho M4).

    ⚠️ CHẠY M4. PYTHONPATH=src python -u scripts/probe_style_composed.py \
        --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_composed.log

## Vì sao (bàn 2026-07-22: Hopfield-cho-văn-phong)

Sinh known-entity nhìn mượt — NHƯNG phần lớn là **chép verbatim** (crisp block dội cha). Câu hỏi:
đường **COMPOSED/A3** (sinh từng-token, ghép đa-fact — không chép nguyên) có bị **word-salad
`. , và`** (TK_017 §12) không? Nếu tệ rõ → **phrase-Hopfield** (truy hồi cả cụm tự nhiên) là lever
thật. Nếu đã ổn → style không phải bài toán.

Probe KHÔNG vá — so 2 chế độ mỗi prompt, ĐỌC TAY:
  * VERBATIM (mặc định, lưới `ar_verbatim_fallback` BẬT) = chép mượt (baseline "đẹp")
  * COMPOSED (tắt verbatim + `essay_ar_min_cov=0`) = ép A3 token-by-token → phơi salad thật

Đo đọc-tay (G15 — không metric tự động cho style): composed có đọc trôi không, hay `. , và`/lặp/cụt.
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

# PHỨC TẠP đa-nội-dung — KHÔNG chép-1-parent được → BUỘC ghép/compose (v2 2026-07-22):
# probe v1 dùng single-fact ("Nhà Trần là gì") → gen chép 1 parent verbatim, composed==verbatim,
# KHÔNG chạm đường ghép. Prompt dưới đây trộn ≥2 thực-thể / yêu-cầu style → không parent nào phủ trọn.
PROMPTS = [
    # đa-thực-thể (buộc ghép 2 chùm fact)
    "Hỏi: So sánh Nhà Lý và Nhà Trần",
    "Hỏi: Kể về Trần Hưng Đạo và trận Bạch Đằng",
    "Hỏi: Thăng Long và Hà Nội có liên quan gì",
    # style + nội-dung (ý tưởng tác giả — style là HOW, nội-dung là WHAT)
    "bằng văn phong hài hước, hãy miêu tả trận Bạch Đằng của Ngô Quyền",
    "Viết một đoạn văn hài hước về Trần Quốc Toản bóp nát quả cam và thế hệ gen Z",
    "kể một câu chuyện buồn về mùa thu Hà Nội",
    # control single-fact (phải == verbatim, chứng minh đường ghép mới khác)
    "Hỏi: Nhà Trần là gì",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--max-len", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-file", default="out/probe_style_composed.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True); logf.write(str(m) + "\n"); logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    log(f"[{time.time()-t0:.0f}s] ✓ loaded · parents {len(state.parents):,}")

    cfg_v = HCAConfig()                       # VERBATIM (lưới an toàn BẬT) — baseline "đẹp"
    cfg_c = HCAConfig()                       # COMPOSED — ép A3 token-by-token
    cfg_c.ar_verbatim_fallback = False
    cfg_c.ar_tail_verbatim_fallback = False
    cfg_c.essay_ar_min_cov = 0.0              # cov luôn ≥0 → không bao giờ dội verbatim

    def gen(prompt, cfg):
        return decode_surface(state, generate(state, prompt, max_len=args.max_len,
                                              config=cfg, seed=args.seed, multi=True))

    log("\n═══ VĂN-PHONG: VERBATIM (chép) vs COMPOSED (A3) — đọc tay salad? ═══")
    for p in PROMPTS:
        log(f"\n▶ {p}")
        log(f"  VERBATIM : {gen(p, cfg_v)!r}")
        log(f"  COMPOSED : {gen(p, cfg_c)!r}")
    log(f"\n[{time.time()-t0:.0f}s] → ĐỌC TAY điểm-xuất-phát composition (prompt phức, buộc ghép):")
    log("   • gen CHỈ chép-1-parent rồi cụt (bỏ zone khác)? = composition CHƯA có → cần build multi-zone.")
    log("   • gen GHÉP nhiều fact nhưng SALAD ('. , và'/lặp/nhảy chủ đề)? = có ghép thô, cần phrase-Hopfield.")
    log("   • gen TRÔI sang entity khác (Bạch Đằng→Nhà Trần)? = grounding drift ở đa-thực-thể.")
    log("   • style 'hài hước/buồn' → có ảnh hưởng gì không, hay bị bỏ qua (=HOW không retrieve được).")
    log("   COMPOSED khác VERBATIM ⇒ đường A3 ĐÃ chạm; == ⇒ vẫn chép-1-parent (single-fact control).")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
