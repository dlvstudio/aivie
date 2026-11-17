"""Thử OFFLINE: GỘP MẢNH ở bước ② — nhãn là CỤM, không phải token.

Chẩn đoán (§60): ở 148 875 và 245 321 cha, ② chọn `bao`/`vật` còn `động` —
nhãn DUY NHẤT chứa `khỉ` — nằm hạng 4. `vật` và `động` là HAI MẢNH của cùng
một cụm «động vật»; phiếu của lớp bị chẻ đôi rồi mỗi nửa thua rác.

§38 đã gộp mảnh ở tầng THÀNH VIÊN (`Bóng`+`Bóng đá`+`đá`); đây là cùng bệnh ở
tầng NHÃN. Ghép hai nhãn khi chúng đứng KỀ nhau thành cụm thật trong đồ thị.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/probe_gop_manh_nhan.py \
        out/noise_big/toy_noise_150000.pkl
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
    duong = sys.argv[1]
    n_gop = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cfg = HCAConfig()
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    state, _ = load_checkpoint(duong)
    v = state.vocab
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        use_isg=False, head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, label_indeg_q=cfg.hub_chain_indeg_q,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm)
    print(f"\n### {duong} · {len(state.parents.parents)} cha · gộp trong top-{n_gop}\n")

    ctx = list(v.encode_prompt("nghịch ngợm như loài"))
    moi = H.resolve_surface_ids(v, "nghịch") + H.resolve_surface_ids(v, "ngợm")
    s1 = H.score_hubs_from(state, idx, moi, topk=12, gidf_min=0.0,
                           channel="typed_first", kenh_thuoc_tinh=True)
    cand = [h for h in s1 if h.surface not in ("ngợm", "như", "nghịch")]
    ty = [h for h in cand if h.typed]
    hat = (ty if ty else cand)[:4]
    s2 = H.score_hubs_from(state, idx, [h.tid for h in hat], topk=20,
                           gidf_min=0.0, channel="typed_first",
                           rank=cfg.hub_chain_rank, ctx_ids=ctx)
    cam = {h.surface for h in hat} | {"nghịch", "ngợm"}
    ok = [h for h in s2 if h.surface not in cam]
    print(f"② CŨ  = {[(h.surface, round(h.score, 1)) for h in ok[:n_gop]]}")

    bt = {h.surface: [int(x) for x in
                      (H.resolve_surface_ids(v, h.surface) or [h.tid])[:6]]
          for h in ok[:n_gop]}

    def _ke(a: str, b: str) -> float:
        """`a b` có phải CỤM thật không — đứng kề, cùng chiều, trong đồ thị."""
        w = 0.0
        for x in bt[a]:
            for y in bt[b]:
                e = idx.fwd.get(x, {}).get(y)
                if e is not None and int(e[1]) <= 1:
                    w = max(w, float(e[0]))
        return w

    top = [h.surface for h in ok[:n_gop]]
    cap: list[tuple[float, str, str]] = []
    for a in top:
        for b in top:
            if a == b:
                continue
            w = _ke(a, b)
            if w > 0:
                cap.append((w, a, b))
    cap.sort(reverse=True)
    print(f"   cặp KỀ tìm được: {[(f'{a} {b}', round(w, 1)) for w, a, b in cap[:6]]}")

    diem = {h.surface: float(h.score) for h in ok[:n_gop]}
    gop: dict[str, float] = dict(diem)
    thanh_phan: dict[str, list[str]] = {s: [s] for s in diem}
    for w, a, b in cap:
        if a in gop and b in gop and a != b:
            ten = f"{a} {b}"
            gop[ten] = diem[a] + diem[b]
            thanh_phan[ten] = [a, b]
            gop.pop(a, None)
            gop.pop(b, None)
    xs = sorted(gop.items(), key=lambda kv: -kv[1])
    print(f"② MỚI = {[(k, round(w, 1)) for k, w in xs[:5]]}")

    # ③ trên nhãn top-1 MỚI: hợp bể của mọi mảnh
    ten, _ = xs[0]
    ids: list[int] = []
    for m in thanh_phan[ten]:
        ids += bt[m][:4]
    ms = [p.surface for p in H.reverse_peers(
        state, idx, ids, topk=30, gidf_min=0.0, channel="typed_first",
        ctx_ids=ctx, ke_ctx=cfg.hub_chain_ke_ctx)]
    hg = (ms.index("khỉ") + 1) if "khỉ" in ms else 0
    print(f"\n③ nhãn `{ten}` · khỉ hạng {hg or '—'} · top-8 {ms[:8]}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
