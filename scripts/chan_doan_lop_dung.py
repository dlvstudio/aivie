"""§100 — PHÉP CHẨN ĐOÁN: nếu ② có nhãn lớp ĐÚNG thì chuỗi có chạy không?

Bốn vòng (§90/§91/§93/§94) đều dừng ở «không tìm được nguồn phân loại tốt».
KHÔNG vòng nào chứng minh rằng CÓ nguồn tốt thì cơ chế sẽ chạy — đó là một giả
định chưa ai kiểm, và nếu nó SAI thì mọi công đi tìm WordNet tiếng Việt là vô ích
ngay từ đầu.

CÁCH ĐO — thay ②③ bằng cây chẩn đoán, GIỮ ① là cơ chế thật:

    ①  thuộc tính → vật mang    `via_attr` THẬT (32 256 cạnh, cơ chế của dự án)
    ②  vật mang → lớp           CÂY CHẨN ĐOÁN (mô hình sinh, 99/119 từ)
    ③  lớp → thành viên         cây đó ĐẢO
    chấm với khoá `out/loai_suy_that_khoa.json` — KHÔNG ĐỤNG VÀO

Cây chẩn đoán KHÔNG phải tài nguyên sản phẩm; nó chỉ tồn tại để phép đo này trả
lời được có/không.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ THẬT phải vượt ĐỐI CHỨNG HOÁN VỊ rõ rệt (≥2×) ở top-10.                   ║
║  · ĐẠT  ⇒ nút thắt ĐÚNG là nhãn lớp; đầu tư vào nguồn thật là đáng.       ║
║  · TRƯỢT ⇒ ĐÓNG hướng «con khỉ»: kể cả có nhãn lớp hoàn hảo, ① không đủ.  ║
║ Hai kết cục đều có giá; không có kết cục «không kết luận được».           ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/chan_doan_lop_dung.py \
        checkpoints/network_v12_sent_ls.pkl
"""
from __future__ import annotations

import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
CT = {"chim", "thú", "cây", "cá", "loài", "hoa", "quả", "con", "giống",
      "kim loại", "cây trồng", "cây leo", "chất lỏng"}


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_ls.pkl"
    cay = json.loads(Path("data/dict/cay_chan_doan.json").read_text("utf-8"))["lop"]
    tv: dict[str, set[str]] = defaultdict(set)
    for w, c in cay.items():
        tv[c].add(w)
    print(f"\n### cây chẩn đoán: {len(cay)} từ · {len(tv)} lớp "
          f"({', '.join(f'{c}:{len(s)}' for c, s in sorted(tv.items()))})")

    cfg = HCAConfig()
    state, _ = load_checkpoint(ck)
    v = state.vocab
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        head_rule=cfg.hub_chain_head_rule, head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, head_gidf_q=cfg.hub_chain_head_gidf_q,
        label_indeg_q=cfg.hub_chain_indeg_q, gop_manh_tu=cfg.hub_chain_gop_manh_tu,
        pos_map=H.load_pos_map(cfg.hub_chain_pos_dict),
        kenh_vi_ngu=bool(cfg.hub_chain_kenh_vi_ngu),
        chu_ngu_ket_hop_q=float(cfg.hub_chain_chu_gia_q))
    va = getattr(idx, "via_attr", None) or {}

    def sf(t: int) -> str:
        return (idx.label_surf.get(int(t))
                or (H._surf(v, int(t)) if int(t) < len(v) else "")).strip()

    kh = json.loads(Path("out/loai_suy_that_khoa.json").read_text("utf-8"))
    bo = [k for k in kh if k["lop"].split()[0] in CT or k["lop"] in CT]
    print(f"### {len(bo)} bộ lớp cụ thể · khoá GIỮ NGUYÊN\n")

    def cham(bo_ds, ten: str, cay=None, tv=None) -> None:
        cay = cay if cay is not None else globals()['_CAY']
        tv = tv if tv is not None else globals()['_TV']
        d1 = d5 = d10 = n = 0
        vd: list[str] = []
        for k in bo_ds:
            A, khi = k["thuoc_tinh"], set(k["khi"])
            if not khi:
                continue
            # ① cơ chế THẬT: thuộc tính → vật mang
            mang = {sf(d) for i in (H.ids_theo_be_mat(idx, v, A) or [])
                    for d in va.get(int(i), {})}
            mang = {x for x in mang if x}
            if not mang:
                continue
            # ②③ CÂY CHẨN ĐOÁN: vật mang → lớp → thành viên khác
            de: list[str] = []
            for b in mang:
                c = cay.get(b)
                if c:
                    de.extend(sorted(tv[c] - mang - {b}))
            if not de:
                continue
            n += 1
            top = list(dict.fromkeys(de))[:10]
            if top[:1] and top[0] in khi:
                d1 += 1
            if any(x in khi for x in top[:5]):
                d5 += 1
            if any(x in khi for x in top[:10]):
                d10 += 1
            if len(vd) < 6:
                vd.append(f"      «{A} như ___» mang={sorted(mang)[:3]}"
                          f" khỉ={sorted(khi)[:3]} → {top[:6]}")
        if not n:
            print(f"   {ten}: 0 bộ chấm được")
            return
        print(f"   {ten}: {n} bộ · top-1 {100*d1/n:.1f}%"
              f" · top-5 {100*d5/n:.1f}% · top-10 {100*d10/n:.1f}%")
        for x in vd:
            print(x)

    globals()["_CAY"], globals()["_TV"] = cay, tv
    cham(bo, "THẬT           ", cay, tv)
    # ĐỐI CHỨNG A — hoán vị KHOÁ. YẾU: cây chỉ có 5 lớp và `cây` chiếm 49/97 từ
    # nên khoá xáo thường rơi trúng cùng lớp ⇒ nền ăn may bị thổi lên.
    rng = random.Random(0)
    khis = [k["khi"] for k in bo]
    rng.shuffle(khis)
    cham([dict(k, khi=x) for k, x in zip(bo, khis)], "ĐC-A xáo khoá  ", cay, tv)
    # ĐỐI CHỨNG B — xáo NHÃN LỚP. Đây mới là null ĐÚNG cho câu hỏi «ánh xạ lớp
    # có mang thông tin không»: giữ nguyên phân bố cỡ lớp, chỉ gán sai từ→lớp.
    ws = list(cay)
    cs = [cay[w] for w in ws]
    rng.shuffle(cs)
    cay_xao = dict(zip(ws, cs))
    tv_xao: dict[str, set[str]] = defaultdict(set)
    for w, c in cay_xao.items():
        tv_xao[c].add(w)
    cham(bo, "ĐC-B xáo nhãn  ", cay_xao, tv_xao)

    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
