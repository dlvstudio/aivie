"""TK_052 — đọc CA ĐẠT và CA TRƯỢT kèm CÂU TRAIN, ánh xạ về ví dụ «con khỉ».

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/doc_ca_loai_suy.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import generate_hebbian as G  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()


def main() -> int:
    cfg = HCAConfig()
    cfg.hub_chain_boost = 4.0
    cfg.hub_chain_nhan_truc_tiep = 0.0     # CHUỖI THUẦN
    cfg.hub_chain_ke_truc_tiep = 0.0
    cfg.hub_chain_gom_tu = 2               # chế độ cho 70,3%
    st, _ = load_checkpoint("checkpoints/loai_suy.pkl")
    v, ps = st.vocab, st.parents
    S = [str(v.token_of(i)).replace(MARK_SPACE, "").strip() for i in range(len(v))]
    khoa = json.load(open("out/loai_suy_khoa.json"))

    wsp = H.load_ws_spans("checkpoints/loai_suy.ws_spans.json")
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    idx = H.build_hub_spread_index(
        st, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        ws_spans=wsp, head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, head_gidf_q=cfg.hub_chain_head_gidf_q,
        label_indeg_q=cfg.hub_chain_indeg_q,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm)

    def sf(t: int) -> str:
        return idx.label_surf.get(int(t)) or (
            H._surf(v, int(t)) if int(t) < len(v) else "")

    # câu train chứa thuộc tính
    def cau_cua(w: str, n: int = 3) -> list[str]:
        ra = []
        for rec in ps.parents:
            txt = " ".join(S[int(t)] for t in rec.emit_ids
                           if S[int(t)] and not S[int(t)].startswith("<"))
            if txt.startswith(w + " ") or f" {w} " in txt:
                if txt not in ra:
                    ra.append(txt)
            if len(ra) >= n:
                break
        return ra

    dat, truot = [], []
    for k in khoa:
        ids = [int(t) for t in v.encode_prompt(f"{k['attr']} như loài")]
        de = G.hub_chain_proposals(st, ids, cfg)
        top = [sf(t) for t, _ in sorted(de.items(), key=lambda kv: -kv[1])[:10]]
        (dat if k["m3"].split()[0] in top else truot).append((k, top))
    print(f"\n### {len(dat)} ĐẠT · {len(truot)} TRƯỢT trên {len(khoa)} bộ\n")

    for ten, ds in (("✓ ĐẠT", dat[:3]), ("✗ TRƯỢT", truot[:3])):
        for k, top in ds:
            print(f"  {ten} — ánh xạ ca «con khỉ»:")
            print(f"     thuộc tính  {k['attr']!r:<12} ↔ `nghịch ngợm`")
            print(f"     vật mang    {k['m1']!r}, {k['m2']!r}  ↔ `chó`, `mèo`")
            print(f"     lớp         {k['lop']!r:<12} ↔ `động vật`")
            print(f"     GIỮ LẠI     {k['m3']!r:<12} ↔ `khỉ`  (chưa từng đứng cạnh thuộc tính)")
            print(f"     CÂU TRAIN:")
            for c in cau_cua(k["attr"], 2):
                print(f"        {c[:150]}")
            print(f"     ĐỀ CỬ  → {top[:8]}")
            vm = [sf(d) for i in (H.resolve_surface_ids(v, k['attr']) or [])
                  for d in idx.via_attr.get(int(i), {})]
            print(f"     ① via_attr[{k['attr']}] = {vm[:6]}")
            nh = [sf(d) for i in (H.resolve_surface_ids(v, k['m1']) or [])
                  for d in idx.via_pivot.get(int(i), {})]
            print(f"     ② via_pivot[{k['m1']}] = {nh[:6]}\n")
    print(f"[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
