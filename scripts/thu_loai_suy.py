"""Dùng thử cạnh loại suy khái niệm (§171 §173–§179).

Nạp net_ls_concept_conf.pkl (A∪B cấp-từ, đích sạch), sinh cho từng prompt và IN
CẢ hai bản: §171 TẮT vs BẬT. Loại suy chỉ nổi lên khi mạng KHÔNG chắc (p_top1
thấp) — rõ nhất ở prompt CỤT/mở như «sắt và», «phở và», «lụa và».

    PYTHONPATH=src python scripts/thu_loai_suy.py "sắt và" "phở và" "kẽm và"
    # không tham số → dùng bộ prompt mẫu

Cờ (env / config mặc định đã hợp lý):
    HCA_BETA=0.6   độ mạnh tiêm loại suy
    HCA_CONF=0.35  chỉ tiêm khi p_top1 < ngưỡng này (cao = tiêm nhiều hơn)
    HCA_MAXLEN=16  số token sinh
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

# tắt §170 (W_skip scale) để CÔ LẬP đóng góp §171 khi so tắt/bật
import nn_aivie.hca.w_skip as _WS  # noqa: E402
_WS.ANALOGY_TO_WSKIP_SCALE = 0.0

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import generate_hebbian as G  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

CKPT = os.environ.get("HCA_CKPT", "checkpoints/net_ls_concept_conf.pkl")
BETA = float(os.environ.get("HCA_BETA", "0.6"))
CONF = float(os.environ.get("HCA_CONF", "0.35"))
MAXLEN = int(os.environ.get("HCA_MAXLEN", "16"))

MAU = [
    "sắt và", "kẽm và", "chì và", "natri và",   # hoá học (neo A)
    "phở và", "lụa và", "gan và",               # món ăn / dệt / bộ phận (neo B)
    "sắt là kim loại",                          # ca vàng: BẬT phải GIỮ nguyên
]


def main() -> int:
    prompts = sys.argv[1:] or MAU
    t0 = time.time()
    st, _ = load_checkpoint(CKPT)
    v = st.vocab
    print(f"[{time.time()-t0:.0f}s] nạp {CKPT}\n", flush=True)

    def gen(p: str, conf: float) -> str:
        cfg = HCAConfig()
        cfg.hub_chain_boost = 1.0
        cfg.ar_w_skip_reinforced_beta = BETA
        cfg.ar_w_skip_reinforced_conf = conf
        n = len(v.encode_prompt(p))
        ids = [int(x) for x in G.generate(st, p, max_len=MAXLEN, config=cfg, seed=0)]
        return v.detokenize(ids[n:]).strip()

    for p in prompts:
        off = gen(p, 0.0)      # §171 TẮT
        on = gen(p, CONF)      # §171 BẬT
        mark = "  (đổi)" if on != off else ""
        print(f"«{p}»{mark}", flush=True)
        print(f"   TẮT: {off}", flush=True)
        print(f"   BẬT: {on}", flush=True)
        print(flush=True)
    print(f"[{time.time()-t0:.0f}s] xong", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
