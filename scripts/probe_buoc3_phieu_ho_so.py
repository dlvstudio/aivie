"""Thử OFFLINE: xếp hạng ③ bằng PHIẾU trên HỒ SƠ NHÃN của hạt giống.

Chẩn đoán (§60): ở 99 362 cha, 62 thành viên typed của nhãn `động` đều có
w = 0,9500 — trọng số hết phân biệt, thứ tự rơi về ABC, `khỉ` đứng 36 nên
rơi khỏi `topk=30`. Cạnh vàng KHÔNG mất.

Ý: bước ② đã có sẵn danh sách nhãn xếp hạng của hạt (`động·cưng·vật·đặc·
súc…`) — đó là HỒ SƠ của lớp. Thành viên nào dính nhiều nhãn trong hồ sơ đó
thì gần hạt hơn thành viên chỉ dính đúng nhãn top-1 (`động` = hang).

Script này KHÔNG sửa cơ chế — nó tính lại thứ hạng ngoài luồng để xem có
đáng vá không.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/probe_buoc3_phieu_ho_so.py \
        out/noise_big/toy_noise_100000.pkl
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
    duong = sys.argv[1] if len(sys.argv) > 1 else "out/noise_big/toy_noise_100000.pkl"
    n_ho_so = int(sys.argv[2]) if len(sys.argv) > 2 else 8
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
    print(f"\n### {duong} · {len(state.parents.parents)} cha · hồ sơ {n_ho_so} nhãn\n")

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
    if not ok:
        print("② rỗng")
        return 1
    nhan = ok[0]
    print(f"① hạt  = {[h.surface for h in hat]}")
    print(f"② nhãn = {nhan.surface} · hồ sơ = "
          f"{[h.surface for h in ok[:n_ho_so]]}")

    nb = H.resolve_surface_ids(v, nhan.surface) or [nhan.tid]
    nhan_ids = {nhan.tid} | {t for t in nb[:4]}

    # HỒ SƠ = mọi biến thể bề mặt của top-N nhãn ở ②, TRỪ nhãn top-1.
    ho_so: dict[int, float] = {}
    for h in ok[1:n_ho_so]:
        for t in (H.resolve_surface_ids(v, h.surface) or [h.tid])[:4]:
            ho_so[int(t)] = max(ho_so.get(int(t), 0.0), float(h.score))
    ho_so = {t: w for t, w in ho_so.items() if t not in nhan_ids}

    # NGỮ CẢNH cho ③: token nội dung của prompt (`loài` là thứ đáng giá nhất —
    # thành viên của lớp hay đứng ngay sau nó, tên hang/tên người thì không).
    # ⚠️ MỘT bề mặt có NHIỀU id BPE và `fwd` khoá theo TỪNG id (§23.1) — tra
    # bằng id thô của `encode_prompt` là bắt hụt. Gom mọi biến thể.
    ctx_bm: list[str] = []
    for t in ctx:
        s = H._surf(v, int(t))
        if len(s) > 1 and s not in H._FUNC and s not in ctx_bm:
            ctx_bm.append(s)
    ctx_nd = [[int(x) for x in (H.resolve_surface_ids(v, s) or [])][:6]
              for s in ctx_bm]
    print(f"   ngữ cảnh ③ = {list(zip(ctx_bm, [len(g) for g in ctx_nd]))}"
          " (bề mặt, số biến thể)")

    be = idx.via_pivot_sach or idx.via_pivot
    hang: list[tuple[float, float, float, str, int]] = []
    for src, dsts in be.items():
        if not (nhan_ids & dsts.keys()):
            continue
        surf = H._surf(v, int(src))
        if not surf or len(surf) <= 1 or surf in H._FUNC:
            continue
        if surf in cam:
            continue
        w = max(float(dsts[h][0]) for h in nhan_ids if h in dsts)
        phieu = sum(1.0 for t in ho_so if t in dsts)
        # đồng hiện W_skip giữa thành viên và từng token ngữ cảnh, hai chiều
        bt_src = [int(src)] + [int(x) for x in
                               (H.resolve_surface_ids(v, surf) or [])[:4]]
        kc = 0.0
        for nhom in ctx_nd:
            if any(b in idx.fwd.get(a, {})
                   for c in nhom for s2_ in bt_src
                   for a, b in ((c, s2_), (s2_, c))):
                kc += 1.0
        # KỀ: ứng viên có đứng ngay sau token ngữ cảnh CUỐI (`loài`) không —
        # đúng cái ô mà đường sinh sắp điền.
        ke = 0.0
        for nhom in reversed(ctx_nd):
            for c in nhom:
                for s2_ in bt_src:
                    e = idx.fwd.get(c, {}).get(s2_)
                    if e is not None and int(e[1]) <= 1:
                        ke = max(ke, float(e[0]))
            if ke:
                break
        hang.append((phieu, kc / max(1, len(ctx_nd)), w, surf, int(src), ke))

    print(f"\n③ bể = {len(hang)} thành viên")
    for ten, key in (
            ("CŨ  (w, ABC)", lambda r: (-r[2], r[3])),
            ("hồ sơ nhãn", lambda r: (-r[0], -r[2], r[3])),
            ("ngữ cảnh đồng hiện", lambda r: (-r[1], -r[2], r[3])),
            ("KỀ sau ngữ cảnh", lambda r: (-r[5], -r[2], r[3])),
            ("kề × hồ sơ", lambda r: (-r[5], -r[0], -r[2], r[3]))):
        xs = sorted(hang, key=key)
        top = [f"{r[3]}·{r[5]:.1f}" for r in xs[:8]]
        vt = next((i for i, r in enumerate(xs, 1) if r[3] == "khỉ"), None)
        print(f"   {ten:<18} khỉ hạng {vt or '—':>4}   top-8 {top}")
    print(f"   có cạnh KỀ: {sum(1 for r in hang if r[5] > 0)}/{len(hang)}")

    # ĐỐI CHỨNG — nếu chỉ hỏi «cái gì đứng ngay sau `loài`?» mà đã ra `khỉ` thì
    # tín hiệu KỀ đang GÁNH cả bài toán và chuỗi ②③ vô dụng. Phải bác.
    doi: dict[str, float] = {}
    for c in ctx_nd[-1]:
        for dst, (w_, d_) in idx.fwd.get(c, {}).items():
            if int(d_) > 1:
                continue
            s = H._surf(v, int(dst))
            if not s or len(s) <= 1 or s in H._FUNC or s in cam:
                continue
            doi[s] = max(doi.get(s, 0.0), float(w_))
    xs = sorted(doi.items(), key=lambda kv: (-kv[1], kv[0]))
    vt = next((i for i, kv in enumerate(xs, 1) if kv[0] == "khỉ"), None)
    print(f"\n   ĐỐI CHỨNG chỉ-KỀ (bỏ chuỗi ②③): bể {len(xs)} · khỉ hạng "
          f"{vt or '—'} · top-8 {[k for k, _ in xs[:8]]}")

    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
