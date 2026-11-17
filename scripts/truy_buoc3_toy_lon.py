"""Truy bước ③ trên checkpoint toy lớn — vì sao `khỉ` rơi khỏi top-30.

§59 kết luận trần ≥59 769 vì ① đã được vá. Chấm lại bằng `cham_toy_lon.py`
(bản có vá §59) cho thấy ② VẪN ĐÚNG ở 99 362 và 245 321 cha — nút thắt dời
sang ③. Script này in TOÀN BỘ bể ③ kèm trọng số để đọc thứ tự thật.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/truy_buoc3_toy_lon.py \
        out/noise_big/toy_noise_100000.pkl
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
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()


def main() -> int:
    duong = sys.argv[1] if len(sys.argv) > 1 else "out/noise_big/toy_noise_100000.pkl"
    cfg = HCAConfig()
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    state, _ = load_checkpoint(duong)
    v = state.vocab
    _sp = Path(duong).parent / (Path(duong).stem + ".ws_spans.json")
    wsp = (H.load_ws_spans(str(_sp))
           if (_sp.exists() and len(sys.argv) > 2 and sys.argv[2] == "ws") else None)
    print(f"  span WS: {len(wsp) if wsp else 0}")
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        use_isg=False, ws_spans=wsp, head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, label_indeg_q=cfg.hub_chain_indeg_q,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm)
    print(f"\n### {duong} · {len(state.parents.parents)} cha\n")

    ctx = list(v.encode_prompt("nghịch ngợm như loài"))
    moi = H.resolve_surface_ids(v, "nghịch") + H.resolve_surface_ids(v, "ngợm")
    s1 = H.score_hubs_from(state, idx, moi, topk=12, gidf_min=0.0,
                           channel="typed_first", kenh_thuoc_tinh=True)
    cand = [h for h in s1 if h.surface not in ("ngợm", "như", "nghịch")]
    ty = [h for h in cand if h.typed]
    hat = (ty if ty else cand)[:4]
    print(f"① hạt = {[(h.surface, h.typed) for h in hat]}")

    s2 = H.score_hubs_from(state, idx, [h.tid for h in hat], topk=20,
                           gidf_min=0.0, channel="typed_first",
                           rank=cfg.hub_chain_rank, ctx_ids=ctx)
    cam = {h.surface for h in hat} | {"nghịch", "ngợm"}
    ok = [h for h in s2 if h.surface not in cam]
    print(f"② nhãn = {[(h.surface, round(h.score, 3)) for h in ok[:8]]}")
    if not ok:
        return 1

    # `khỉ` NẰM Ở NHÃN NÀO — nếu ② chọn một MẢNH khác của cùng cụm lớp
    # (`vật` thay vì `động` của «động vật») thì bể ③ mất sạch đích.
    kids = H.resolve_surface_ids(v, "khỉ")
    be0 = idx.via_pivot_sach or idx.via_pivot
    nhan_khi = {H._surf(v, int(d)): round(float(w), 2)
                for k in kids for d, (w, _m) in be0.get(int(k), {}).items()}
    print(f"   nhãn của `khỉ` = {nhan_khi}")
    for h in ok[:5]:
        bt = set(H.resolve_surface_ids(v, h.surface) or [h.tid]) | {h.tid}
        co = any(d in bt for k in kids for d in be0.get(int(k), {}))
        print(f"     ② #{ok.index(h)+1} {h.surface:<8} chứa khỉ: {co}")

    nhan = ok[0]
    nb = H.resolve_surface_ids(v, nhan.surface) or [nhan.tid]
    nhan_ids = [nhan.tid] + [t for t in nb if t != nhan.tid][:3]
    ms = H.reverse_peers(state, idx, nhan_ids, topk=200, gidf_min=0.0,
                         channel="typed_first")
    print(f"\n③ bể `{nhan.surface}` = {len(ms)} thành viên (topk=200)")
    print("   hạng  bề mặt        w      typed")
    for i, m in enumerate(ms[:12], 1):
        print(f"   {i:>4}  {m.surface:<12}  {m.weight:.4f}  {m.typed}")
    for i, m in enumerate(ms, 1):
        if m.surface == "khỉ":
            print(f"   ...\n   {i:>4}  {m.surface:<12}  {m.weight:.4f}  {m.typed}  ← ĐÍCH")
            break
    else:
        print("   khỉ KHÔNG có trong 200")

    # Trọng số có phân biệt được không, hay ABC đang quyết?
    dem = Counter(round(m.weight, 4) for m in ms if m.typed)
    print(f"\n   phân bố w (typed): {dem.most_common(6)}")
    bang = [m.surface for m in ms if m.typed and round(m.weight, 4)
            == max(dem, key=lambda k: dem[k])]
    print(f"   {len(bang)} thành viên CÙNG w trội — 12 đầu: {bang[:12]}")

    # Câu train sinh cạnh rác: đọc `mid` (pivot) của vài thành viên
    be = idx.via_pivot_sach or idx.via_pivot
    want = set(nhan_ids)
    print("\n   pivot (mid) của cạnh → nhãn:")
    for m in ms[:6] + [x for x in ms if x.surface == "khỉ"]:
        for hid in want:
            d = be.get(m.tid, {})
            if hid in d:
                w, mid = d[hid]
                print(f"     {m.surface:<12} w={w:.3f} mid="
                      f"{H._surf(v, int(mid)) if mid is not None else None!r}")
                break
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
