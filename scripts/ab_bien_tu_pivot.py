"""A/B §62 — biên TỪ cho pivot — trên `classrich`, đo bằng cạnh và bằng ca.

Trên toy, §62 vô hiệu: khi có span WS thì guard cũ ở `_ung_vien_trai` (bỏ ứng
viên mà span của nó CHỨA pivot) đã chặn `bao` của «bao gồm» rồi.

Chỗ §62 làm được việc mà guard cũ không: **vế PHẢI**. Với «X gọi là Y», pivot
`gọi` khiến quét phải dừng ngay ở `là` (`cs in piv → break`) ⇒ cạnh MẤT. Đếm
trên classrich: `gọi` 1 001 lần, 67,8% nằm trong «gọi là».

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/ab_bien_tu_pivot.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classrich.pkl"
    sp = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/classrich.ws_spans.json"
    cfg = HCAConfig()
    state, _ = load_checkpoint(ck)
    v = state.vocab
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    wsp = H.load_ws_spans(sp)
    print(f"\n### {ck} · {len(state.parents.parents)} cha · {len(wsp or ())} span\n")

    for bien in (False, True):
        idx = H.build_hub_spread_index(
            state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
            ws_spans=wsp, bien_tu_pivot=bien,
            head_rule=cfg.hub_chain_head_rule,
            head_rule_trai=cfg.hub_chain_head_rule_trai,
            label_gidf_q=cfg.hub_chain_gidf_q,
            label_indeg_q=cfg.hub_chain_indeg_q,
            chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm)
        n_c = sum(len(d) for d in idx.via_pivot.values())
        n_s = sum(len(d) for d in idx.via_pivot_sach.values())
        n_n = len({d for dd in idx.via_pivot.values() for d in dd})
        # cạnh sinh bởi pivot `gọi` — thứ §62 nhắm tới
        goi = set(H.resolve_surface_ids(v, "gọi") or [])
        n_goi = sum(1 for dd in idx.via_pivot.values()
                    for _d, (_w, m) in dd.items() if int(m) in goi)
        print(f"  biên_từ={str(bien):<5}  cạnh={n_c:>6}  bể③={n_s:>6}"
              f"  nhãn={n_n:>5}  cạnh qua `gọi`={n_goi:>4}")
        del idx
        gc.collect()
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
