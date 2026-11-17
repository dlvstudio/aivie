"""Thử OFFLINE: khi PHIẾU hoà, nhãn là TỪ ĐỦ (span WS) thắng nhãn MẢNH.

Bật span WS trên toy 99 362 cho bức tranh mới (§63):

    chó → {vật 1,9 · phổ biến · chủ yếu · gia súc · cưng · động vật}
    mèo → {đề nghị · cưng · động vật · đặc sản · Hà Nội · thiếu nhi}
    khỉ → {động vật}

`động vật` nay là MỘT nhãn nguyên khối (đường 2 «gộp mảnh» tự giải), và nó
hoà 2 phiếu với `cưng` — nhưng thua vì `score` 2,7 < 15,6. `score` của id-span
và của token KHÔNG cùng thang đo (đúng bệnh §18.H).

Thử khoá phụ: TỪ ĐỦ thắng MẢNH. Không phải danh sách từ — hỏi span WS.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/probe_nhan_tu_du.py
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
    cfg = HCAConfig()
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    thu_muc = Path(sys.argv[1] if len(sys.argv) > 1 else "out/noise_big")
    print(f"\n### {thu_muc} · nhãn TỪ ĐỦ thắng MẢNH khi hoà phiếu\n")
    print(f"  {'cha':>7}  {'② CŨ':>10}  {'② MỚI':>10}  {'khỉ':>4}  ③ top-6")
    for p in sorted(thu_muc.glob("toy_noise_*.pkl"),
                    key=lambda q: int("".join(c for c in q.stem if c.isdigit()))):
        sp = p.parent / (p.stem + ".ws_spans.json")
        if not sp.exists():
            continue
        state, _ = load_checkpoint(str(p))
        v = state.vocab
        wsp = H.load_ws_spans(str(sp))
        idx = H.build_hub_spread_index(
            state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
            use_isg=False, ws_spans=wsp, head_rule=cfg.hub_chain_head_rule,
            head_rule_trai=cfg.hub_chain_head_rule_trai,
            label_gidf_q=cfg.hub_chain_gidf_q, label_indeg_q=cfg.hub_chain_indeg_q,
            chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm)
        ctx = list(v.encode_prompt("nghịch ngợm như loài"))
        moi = (H.resolve_surface_ids(v, "nghịch")
               + H.resolve_surface_ids(v, "ngợm"))
        s1 = H.score_hubs_from(state, idx, moi, topk=12, gidf_min=0.0,
                               channel="typed_first", kenh_thuoc_tinh=True)
        cand = [h for h in s1 if h.surface not in ("ngợm", "như", "nghịch")]
        ty = [h for h in cand if h.typed]
        hat = (ty if ty else cand)[:4]
        s2 = H.score_hubs_from(state, idx, [h.tid for h in hat], topk=20,
                               gidf_min=0.0, channel="typed_first",
                               rank=cfg.hub_chain_rank, ctx_ids=ctx) if hat else []
        cam = {h.surface for h in hat} | {"nghịch", "ngợm"}
        ok = [h for h in s2 if h.surface not in cam]

        # PHIẾU: hạt nào bầu nhãn nào (score_hubs_from không trả ra)
        be = idx.via_pivot
        phieu: dict[int, int] = {}
        for h in hat:
            for i in (H.resolve_surface_ids(v, h.surface) or [h.tid])[:4]:
                for dst in be.get(int(i), {}):
                    phieu[int(dst)] = phieu.get(int(dst), 0) + 1

        def tu_du(t: int) -> int:
            """1 nếu nhãn là TỪ ĐỦ (span WS ≥2 token), 0 nếu là mảnh token."""
            return 1 if len(idx.label_span.get(int(t), ())) >= 2 else 0

        # KHOÁ THỨ BA: lớp THẬT có nhiều thành viên. `label_indeg` = số chủ ngữ
        # KHÁC NHAU trỏ tới nhãn. §24/§55 từng dùng nó làm BỘ LỌC và nó cắt mất
        # cạnh vàng; ở đây nó chỉ là khoá PHỤ khi phiếu đã hoà (bài học 48).
        def indeg(t: int) -> int:
            return int(idx.label_indeg.get(int(t), 0))
        moi_xep = sorted(ok, key=lambda h: (-phieu.get(int(h.tid), 0),
                                            -tu_du(int(h.tid)),
                                            -indeg(int(h.tid)), -h.score))
        cu_xep = sorted(ok, key=lambda h: (-phieu.get(int(h.tid), 0),
                                           -tu_du(int(h.tid)), -h.score))
        ra: list[str] = []
        if moi_xep:
            n0 = moi_xep[0]
            nb = H.resolve_surface_ids(v, n0.surface) or [n0.tid]
            ra = [m.surface for m in H.reverse_peers(
                state, idx, [n0.tid] + [t for t in nb if t != n0.tid][:3],
                topk=30, gidf_min=0.0, channel="typed_first",
                ctx_ids=ctx, ke_ctx=cfg.hub_chain_ke_ctx)]
        hg = (ra.index("khỉ") + 1) if "khỉ" in ra else 0
        print(f"  {len(state.parents.parents):>7}  "
              f"{(ok[0].surface if ok else '—'):>10}  "
              f"{(moi_xep[0].surface if moi_xep else '—'):>10}  "
              f"[{(cu_xep[0].surface if cu_xep else '—')}] "
              f"{hg or '—':>4}  {ra[:6]}")
        del state, idx
        gc.collect()
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
