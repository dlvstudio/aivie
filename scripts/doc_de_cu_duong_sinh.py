"""ĐỌC BẰNG MẮT thứ chuỗi ba bước ĐỀ CỬ cho đường sinh (§67 D).

Mọi thước từ §60 tới §66 đều là thước TRUNG GIAN (hạng của `khỉ`, ②top1, độ
sạch cạnh). Mục tiêu thật của tác giả là SINH: tốt hơn, sáng tạo hơn, vẫn đúng
tri thức. Chưa mục nào bật `hub_chain_boost` trên `classrich2` rồi đọc xem nó
đề cử gì.

Script này in thẳng đề cử của `hub_chain_proposals` cho vài prompt — không
chấm điểm, chỉ để đọc.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/doc_de_cu_duong_sinh.py \
        checkpoints/classrich2.pkl
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import generate_hebbian as G  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
PROMPT = [
    "nghịch ngợm như loài",
    "trung thành như loài",
    "chó là",
    "mèo là một loài",
    "hổ là loài",
    "cây bàng là",
]


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classrich2.pkl"
    cfg = HCAConfig()
    cfg.hub_chain_boost = 4.0                 # BẬT đường sinh
    if "nodt" in sys.argv:                    # A/B bản vá §66
        cfg.hub_chain_hat_dong_thuan = False
    cfg.hub_chain_kenh_vi_ngu = "vingu" in sys.argv
    for a in sys.argv[2:]:
        if a.startswith("tt="):
            cfg.hub_chain_nhan_truc_tiep = float(a[3:])
        if a.startswith("kt="):
            cfg.hub_chain_ke_truc_tiep = float(a[3:])
    print(f"  hạt_đồng_thuận={cfg.hub_chain_hat_dong_thuan} · "
          f"kênh_vị_ngữ={cfg.hub_chain_kenh_vi_ngu}")
    state, _ = load_checkpoint(ck)
    v = state.vocab
    print(f"\n### {ck} · {len(state.parents.parents)} cha · boost={cfg.hub_chain_boost}\n")
    for pr in PROMPT:
        ids = list(v.encode_prompt(pr))
        de = G.hub_chain_proposals(state, ids, cfg)
        top = sorted(de.items(), key=lambda kv: -kv[1])[:10]
        print(f"  «{pr} ___»")
        print(f"     {[(H._surf(v, t), round(float(w), 2)) for t, w in top]}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
