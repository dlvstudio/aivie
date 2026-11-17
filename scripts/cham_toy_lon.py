"""Chấm checkpoint toy bằng ĐÚNG cấu hình mặc định hiện tại (TK_046 §58).

`probe_chain3_noise_curve.py` dựng chuỗi theo bản CŨ (chưa có kênh thuộc tính
§53, `rank="phieu_x_ctx"` §43, luật vế trái §58). Phần đắt của nó là TRAIN và
checkpoint được giữ lại, nên script này chấm lại rời — không phải train lần nữa.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/cham_toy_lon.py out/noise_big
"""
from __future__ import annotations

import gc
import re
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
    thu_muc = Path(sys.argv[1] if len(sys.argv) > 1 else "out/noise_big")
    cfg = HCAConfig()
    if len(sys.argv) > 2:          # A/B kề-ngữ-cảnh (§60): 0 = tắt
        cfg.hub_chain_ke_ctx = float(sys.argv[2])
    # argv[3]: "0" = KHÔNG nạp span WS · "1" = nạp · argv[4]: biên-từ-pivot §62
    dung_ws = (len(sys.argv) <= 3) or sys.argv[3] != "0"
    bien = (len(sys.argv) <= 4) or sys.argv[4] != "0"
    tu_du = (len(sys.argv) <= 5) or sys.argv[5] != "0"      # §63
    if len(sys.argv) > 6:                                   # §64 head_gidf_q
        cfg.hub_chain_head_gidf_q = float(sys.argv[6])
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    ck = sorted(thu_muc.glob("toy_noise_*.pkl"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    print(f"\n### CHẤM {len(ck)} checkpoint bằng mặc định §58\n")
    print(f"  head_rule={cfg.hub_chain_head_rule} · trai={cfg.hub_chain_head_rule_trai}"
          f" · rank={cfg.hub_chain_rank} · lọc gidf={cfg.hub_chain_gidf_q}"
          f"/indeg={cfg.hub_chain_indeg_q}\n")
    print(f"  cờ: ke_ctx={cfg.hub_chain_ke_ctx} · span={dung_ws} · "
          f"biên_từ={bien} · từ_đủ={tu_du} · "
          f"head_gidf_q={cfg.hub_chain_head_gidf_q}")
    print(f"  {'cha':>7}  {'nhãn':>8}  {'khỉ hạng':>9}  {'③ top-6'}")
    for p in ck:
        state, _ = load_checkpoint(str(p))
        v = state.vocab
        # span WS nếu đã bake — §62 cần nó để định BIÊN TỪ của pivot
        _sp = p.with_suffix("").with_suffix(".ws_spans.json")
        _sp = p.parent / (p.stem + ".ws_spans.json")
        wsp = H.load_ws_spans(str(_sp)) if (dung_ws and _sp.exists()) else None
        idx = H.build_hub_spread_index(
            state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
            use_isg=False, ws_spans=wsp, bien_tu_pivot=bien,
            gop_manh_tu=("gopmanh" in sys.argv),
            kenh_vi_ngu=("vingu" in sys.argv),
            head_gidf_q=cfg.hub_chain_head_gidf_q,
            head_rule=cfg.hub_chain_head_rule,
            head_rule_trai=cfg.hub_chain_head_rule_trai,
            label_gidf_q=cfg.hub_chain_gidf_q,
            label_indeg_q=cfg.hub_chain_indeg_q,
            chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm)
        ctx = list(v.encode_prompt("nghịch ngợm như loài"))
        moi = (H.resolve_surface_ids(v, "nghịch")
               + H.resolve_surface_ids(v, "ngợm"))
        s1 = H.score_hubs_from(state, idx, moi, topk=60, gidf_min=0.0,
                               channel="typed_first", kenh_thuoc_tinh=True)
        # HẠT GIỐNG CHỈ-TYPED (§59) — cùng luật với `hub_chain_proposals`:
        # có hit typed thì chỉ dùng typed, không có mới rơi về cooc.
        _cand = [h for h in s1 if h.surface not in ("ngợm", "như", "nghịch")]
        _ty = [h for h in _cand if h.typed]
        hat = (_ty if _ty else _cand)[:4]
        if (cfg.hub_chain_hat_dong_thuan and len(_ty) < 2
                and "nodt" not in sys.argv):                          # §66
            _dt = H.chon_hat_dong_thuan(idx, v, _cand, topk=4)
            if _dt:
                hat = _dt
        s2 = H.score_hubs_from(
            state, idx, [h.tid for h in hat], topk=20, gidf_min=0.0,
            channel="typed_first", rank=cfg.hub_chain_rank,
            uu_tien_tu_du=tu_du, ctx_ids=ctx) if hat else []
        cam = {h.surface for h in hat} | {"nghịch", "ngợm"}
        ok = [h for h in s2 if h.surface not in cam]
        ms: list[str] = []
        if ok:
            nb = H.resolve_surface_ids(v, ok[0].surface) or [ok[0].tid]
            ms = [m.surface for m in H.reverse_peers(
                state, idx, [ok[0].tid] + nb[:3], topk=30, gidf_min=0.0,
                channel="typed_first", ctx_ids=ctx,
                ke_ctx=cfg.hub_chain_ke_ctx)]
        hg = (ms.index("khỉ") + 1) if "khỉ" in ms else 0
        print(f"  {len(state.parents.parents):>7}  "
              f"{str(ok[0].surface if ok else None):>8}  {hg or '—':>9}  {ms[:6]}"
              f"   hạt={[h.surface for h in hat]}")
        del state, idx
        gc.collect()
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
