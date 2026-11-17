"""Bước ① được nuôi bằng gì? — đo KÊNH THUỘC TÍNH trên dữ liệu thật (§68).

§53 dựng `via_attr` = loại cạnh THỨ HAI, trích từ pivot SO SÁNH (`như`/`giống`/
`tựa`): «nghịch ngợm NHƯ loài chó» → `nghịch ngợm → chó`. Đó là toàn bộ nguồn
typed của bước ①.

§67.A đo được ① trả về VẬT MANG đúng nhưng hiếm khi có 2 vật mang cùng lớp.
Câu chưa hỏi: **kênh này có bao nhiêu cạnh trên dữ liệu thật, và bao nhiêu
thuộc tính có ≥2 vật mang?** Nếu nó thưa thì mọi bản vá cho ① đều vá trên một
kênh gần rỗng — cùng loại chẩn đoán với §66.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/do_kenh_thuoc_tinh.py
"""
from __future__ import annotations

import gc
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
SS = {"như", "giống", "tựa"}


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classrich2.pkl"
    cfg = HCAConfig()
    state, _ = load_checkpoint(ck)
    v, ps = state.vocab, state.parents
    S = [str(v.token_of(i)).replace(MARK_SPACE, "").strip() for i in range(len(v))]
    wsp = H.load_ws_spans(ck.replace(".pkl", ".ws_spans.json"))
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        ws_spans=wsp, head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, head_gidf_q=cfg.hub_chain_head_gidf_q,
        label_indeg_q=cfg.hub_chain_indeg_q,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm,
        kenh_vi_ngu=("vingu" in sys.argv))
    print(f"  cờ: kênh_vị_ngữ={'vingu' in sys.argv}")
    print(f"\n### {ck} · {len(ps.parents)} cha\n")

    # 1. câu có pivot SO SÁNH
    dem = Counter()
    for rec in ps.parents:
        for t in rec.emit_ids:
            if S[int(t)] in SS:
                dem[S[int(t)]] += 1
    print(f"  câu chứa pivot so sánh: {dict(dem)}  ·  tổng {sum(dem.values())}")

    # 2. kênh via_attr
    n_c = sum(len(d) for d in idx.via_attr.values())
    hai = [s for s, d in idx.via_attr.items() if len(d) >= 2]
    ba = [s for s, d in idx.via_attr.items() if len(d) >= 3]
    print(f"  `via_attr`: {len(idx.via_attr)} thuộc tính · {n_c} cạnh")
    print(f"     có ≥2 vật mang: {len(hai)} ({100*len(hai)/max(1,len(idx.via_attr)):.1f}%)"
          f"  ·  có ≥3: {len(ba)}")

    # 3. so với kênh PHÂN LOẠI
    n_p = sum(len(d) for d in idx.via_pivot.values())
    print(f"  `via_pivot` (để so): {len(idx.via_pivot)} chủ ngữ · {n_p} cạnh")

    # 4. thuộc tính nào có nhiều vật mang nhất — đọc bằng mắt
    top = sorted(idx.via_attr.items(), key=lambda kv: -len(kv[1]))[:10]
    print("\n  10 thuộc tính nhiều vật mang nhất:")
    for s_, d in top:
        def sf(t: int) -> str:
            return idx.label_surf.get(int(t)) or (
                H._surf(v, int(t)) if int(t) < len(v) else f"<{t}>")
        print(f"     {sf(s_):<14} {len(d):>3} vật mang: {[sf(x) for x in list(d)[:6]]}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
