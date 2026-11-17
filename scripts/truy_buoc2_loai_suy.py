"""TK_052 vòng 2 — vì sao ② hỏng dù cạnh `m→C` tồn tại 94,5%?

Vòng 1: thu hồi m3 bão hoà 27,6% (đối chứng 0,7%), nhưng LỚP ĐÚNG chỉ vào
top-10 ở 1,4% bộ. Tức ③ ra thành viên đúng mà ② không hề chọn đúng lớp — hai
bước không khớp. Script này chạy TAY từng bước trên vài bộ để xem đứt ở đâu.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/truy_buoc2_loai_suy.py
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
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/loai_suy.pkl"
    n_in = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 6
    cfg = HCAConfig()
    state, _ = load_checkpoint(ck)
    v = state.vocab
    wsp = (None if "nospan" in sys.argv
           else H.load_ws_spans(ck.replace(".pkl", ".ws_spans.json")))
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        ws_spans=wsp, head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, head_gidf_q=cfg.hub_chain_head_gidf_q,
        label_indeg_q=cfg.hub_chain_indeg_q,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm)
    khoa = json.load(open("out/loai_suy_khoa.json"))
    print(f"\n### {ck} · {len(state.parents.parents)} cha · {len(khoa)} bộ · "
          f"span={'TẮT' if wsp is None else len(wsp)}\n")

    def sf(t: int) -> str:
        return idx.label_surf.get(int(t)) or (
            H._surf(v, int(t)) if int(t) < len(v) else "")

    dem = {"①_ra_m": 0, "②_ra_C": 0, "③_ra_m3": 0, "n": 0}
    for k in khoa:
        ctx = list(v.encode_prompt(f"{k['attr']} như loài"))
        aid = H.resolve_surface_ids(v, k["attr"]) or []
        if not aid:
            continue
        dem["n"] += 1
        s1 = H.score_hubs_from(state, idx, aid[:4], topk=12, gidf_min=0.0,
                               channel="typed_first", kenh_thuoc_tinh=True)
        hat_all = [h for h in s1 if h.surface != k["attr"]]
        ty = [h for h in hat_all if h.typed]
        hat = (ty if ty else hat_all)[:4]
        want_m = {k["m1"], k["m2"], k["m1"].split()[0], k["m2"].split()[0]}
        co_m = [h.surface for h in hat if h.surface in want_m
                or (want_m & set(h.surface.split()))]
        if co_m:
            dem["①_ra_m"] += 1

        s2 = H.score_hubs_from(state, idx, [h.tid for h in hat], topk=10,
                               gidf_min=0.0, channel="typed_first",
                               rank=cfg.hub_chain_rank, loc_pos=True,
                               ctx_ids=ctx) if hat else []
        cam = {h.surface for h in hat} | {k["attr"]}
        ok = [h for h in s2 if h.surface not in cam]
        want_c = {k["lop"], k["lop"].split()[0]}
        vi_c = next((i for i, h in enumerate(ok, 1)
                     if h.surface in want_c or (want_c & set(h.surface.split()))),
                    None)
        if vi_c == 1:
            dem["②_ra_C"] += 1

        ra: list[str] = []
        if ok:
            nb = H.resolve_surface_ids(v, ok[0].surface) or [ok[0].tid]
            ra = [m.surface for m in H.reverse_peers(
                state, idx, [ok[0].tid] + [t for t in nb if t != ok[0].tid][:3],
                topk=20, gidf_min=0.0, channel="typed_first",
                ctx_ids=ctx, ke_ctx=cfg.hub_chain_ke_ctx)]
        _r = [x.split()[0] for x in ra]
        if k["m3"].split()[0] in _r:
            dem["③_ra_m3"] += 1
        if k["m3"].split()[0] in _r[:10]:
            dem["③_top10"] = dem.get("③_top10", 0) + 1
        if k["m3"].split()[0] in _r[:5]:
            dem["③_top5"] = dem.get("③_top5", 0) + 1

        if dem["n"] <= n_in:
            print(f"  A={k['attr']!r} lớp={k['lop']!r} m=[{k['m1']},{k['m2']}] m3={k['m3']!r}")
            print(f"     ① hạt = {[(h.surface, h.typed) for h in hat]}   TRÚNG m: {co_m}")
            print(f"     ② nhãn = {[h.surface for h in ok[:5]]}   hạng của lớp đúng: {vi_c}")
            print(f"     ③ = {ra[:6]}")
    n = max(1, dem["n"])
    print(f"\n  {dem['n']} bộ · ① ra vật mang đúng {100*dem['①_ra_m']/n:.1f}%"
          f" · ② ra lớp đúng ở hạng 1 {100*dem['②_ra_C']/n:.1f}%"
          f" · ③ ra m3 (top-20) {100*dem['③_ra_m3']/n:.1f}%"
          f" · top-10 {100*dem.get('③_top10',0)/n:.1f}%"
          f" · top-5 {100*dem.get('③_top5',0)/n:.1f}%")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
