"""§89 — chấm CỔNG §88.16 vế 2 và 3 trên checkpoint có corpus loại suy.

Vế 1 (TRUNG THÀNH ≥ 78,9%, PHỦ QUYẾT) chấm bằng `ab_thu_tu_cha_hoat.py`.
Script này chấm hai vế còn lại:

  VẾ 2  bốn prompt LOẠI SUY (0/4 suốt cả phiên) phải có ≥1 câu đề cử đúng LỚP
  VẾ 3  thu hồi «con khỉ» trên 44 bộ lớp CỤ THỂ, so với ĐỐI CHỨNG HOÁN VỊ

Đối chứng hoán vị: trộn ngẫu nhiên ánh xạ thuộc tính → bộ, rồi chấm lại. Nếu thu
hồi thật ≈ thu hồi hoán vị thì cơ chế KHÔNG dùng cạnh thuộc tính, chỉ ăn may
theo tần suất lớp. TK_052 vòng 3 đo được 84,8% vs 0,0% trên corpus tổng hợp.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/cham_cong_loai_suy.py \
        checkpoints/network_v12_sent_ls.pkl
"""
from __future__ import annotations

import gc
import json
import random
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
CT = {"chim", "thú", "cây", "cá", "loài", "hoa", "quả", "con", "giống",
      "kim loại", "hành tinh", "tỉnh", "thành phố", "sông", "núi",
      "cây trồng", "cây leo", "chất lỏng"}
LOAI_SUY = ["nghịch ngợm như loài", "chó và mèo đều là",
            "chó nghịch ngợm, mèo nghịch ngợm, còn", "hổ giống mèo vì",
            "khoẻ như", "nhanh như", "khôn như"]


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_ls.pkl"
    cfg = HCAConfig()
    cfg.hub_chain_boost = 1.0
    cfg.hub_chain_loc_dich = "loc" in sys.argv     # §93
    for _a in sys.argv[2:]:                        # §101
        if _a.startswith("kenh="):
            cfg.hub_chain_channel = _a[5:]
        if _a.startswith("gt="):                   # §109
            cfg.hub_chain_gom_tu = int(_a[3:])
        if _a.startswith("aw="):                   # §107
            cfg.hub_chain_attr_w_min = float(_a[3:])
        if _a.startswith("g2="):                   # §105
            cfg.hub_chain_gidf_min_2 = float(_a[3:])
        if _a.startswith("htop="):
            cfg.hub_chain_hub_top = int(_a[5:]); cfg.hub_chain_hub_lam_de_cu = True
        if _a.startswith("k1="):                   # §103
            cfg.hub_chain_channel_1 = _a[3:]
        if _a.startswith("k2="):
            cfg.hub_chain_channel_2 = _a[3:]
    state, _ = load_checkpoint(ck)
    v = state.vocab
    print(f"\n### {ck} · {len(state.parents.parents)} cha"
          f" · ①={cfg.hub_chain_channel_1 or cfg.hub_chain_channel}"
          f" ②={cfg.hub_chain_channel_2 or cfg.hub_chain_channel}"
          f" · aw={cfg.hub_chain_attr_w_min} gt={cfg.hub_chain_gom_tu}")

    # ── VẾ 2 — bốn prompt loại suy, đọc bằng MẮT
    print("\n═══ VẾ 2 · prompt LOẠI SUY")
    for p in LOAI_SUY:
        try:
            ids = G.generate(state, p, max_len=20, config=cfg, seed=0)
            print(f"   «{p}» ▸ {v.detokenize(ids)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   «{p}» ▸ !! {type(e).__name__}: {e}")

    # ── VẾ 3 — thu hồi «con khỉ», thật vs hoán vị
    kh = json.load(open("out/loai_suy_that_khoa.json"))
    # §102 — `tatca` dùng CẢ 198 bộ. 44 bộ lớp-cụ-thể là quá ít: §101 đo cooc
    # 2,3% = ĐÚNG MỘT bộ, không tách được khỏi 0,0% của đối chứng.
    if "tatca" in sys.argv:
        bo = [k for k in kh if k["khi"]]
    else:
        bo = [k for k in kh if k["lop"].split()[0] in CT or k["lop"] in CT]
    print(f"\n═══ VẾ 3 · thu hồi trên {len(bo)} bộ lớp CỤ THỂ")

    def cham(bo_ds, ten: str) -> None:
        d1 = d5 = d10 = n = 0
        vd: list[str] = []
        for k in bo_ds:
            A, khi = k["thuoc_tinh"], set(k["khi"])
            if not khi:
                continue
            ids = [int(t) for t in v.encode_prompt(f"{A} như")]
            if not ids:
                continue
            de = G.hub_chain_proposals(state, ids, cfg)
            if not de:
                continue
            n += 1
            top = [H._surf(v, int(t)).strip()
                   for t, _ in sorted(de.items(), key=lambda kv: -kv[1])[:10]]
            if top[:1] and top[0] in khi:
                d1 += 1
            if any(x in khi for x in top[:5]):
                d5 += 1
            if any(x in khi for x in top[:10]):
                d10 += 1
            if len(vd) < 5:
                vd.append(f"      «{A} như ___» khỉ={sorted(khi)[:4]} → {top[:6]}")
        if not n:
            print(f"   {ten}: 0 bộ chấm được")
            return
        print(f"   {ten}: {n} bộ · top-1 {100*d1/n:.1f}%"
              f" · top-5 {100*d5/n:.1f}% · top-10 {100*d10/n:.1f}%")
        for x in vd:
            print(x)

    cham(bo, "THẬT      ")
    # ĐỐI CHỨNG: trộn ánh xạ thuộc tính → khỉ, giữ nguyên phân bố lớp
    rng = random.Random(0)
    khis = [k["khi"] for k in bo]
    rng.shuffle(khis)
    hv = [dict(k, khi=x) for k, x in zip(bo, khis)]
    cham(hv, "HOÁN VỊ   ")

    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
