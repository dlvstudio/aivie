"""§92 — ĐỀ XUẤT CÓ KIỂM CHỨNG: đổi mục tiêu thay vì vá thêm cơ chế.

§89 đo được ① chạy (33 974 cạnh `via_attr`) mà ② vỡ vì nhãn lớp chung chung.
§88.12 chứng minh ② KHÔNG sửa được bằng dữ liệu: điều kiện «≥2 thành viên cùng
lớp mang cùng thuộc tính» đòi đúng thứ ngôn ngữ tự nhiên tránh nói (thành ngữ cố
tình chọn vật mẫu khác lớp: `nhanh như cắt` là chim, `nhanh như sóc` là thú —
trên 50 thuộc tính chỉ 2 đạt, cả hai rác).

ĐỔI MỤC TIÊU. Không đòi chuỗi THU HỒI thành viên bị giấu. Đòi nó ĐỀ XUẤT, và
mỗi đề xuất phải KIỂM CHỨNG ĐƯỢC bằng ba cạnh đã có trong đồ thị:

    ①  A → b        `via_attr`   (`nhanh như cắt` — đã chứng thực)
    ②  b → C        `via_pivot`  (`cắt là chim` — ĐỌC THẲNG, không suy ra)
    ③  C → m        `via_pivot` đảo  (`cú là chim`, `cò là chim` — đã chứng thực)
    và (A, m) CHƯA từng đứng cạnh nhau  ⇒ đề xuất MỚI

Khác biệt cốt lõi với ② cũ: lớp lấy từ cạnh `b là C` CỦA CHÍNH VẬT MANG ĐÓ — một
sự thật đã phát biểu — thay vì tìm lớp giải thích được nhiều vật mang. Một vật
mang là đủ để GỢI Ý; nó không đủ để KHẲNG ĐỊNH, và ta không khẳng định.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ (a) ≥50% đề xuất đọc bằng MẮT là hợp lý về ngữ nghĩa, và                  ║
║ (b) lớp dùng để bắc cầu phải CỤ THỂ — nếu phần lớn là `đồ`/`phần`/`cái`   ║
║     thì đổi mục tiêu KHÔNG cứu được gì, kết luận vẫn là §89.              ║
║ Trượt ⇒ âm tính, ghi lại, KHÔNG vá thêm cơ chế.                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/de_xuat_co_kiem_chung.py \
        checkpoints/network_v12_sent_ls.pkl
"""
from __future__ import annotations

import gc
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
# thuộc tính có thật trong corpus §88, đọc được bằng mắt
THU = ["nhanh", "khoẻ", "khôn", "nhọn", "ngọt", "trắng", "tròn", "mềm",
       "đen", "đẹp", "lớn", "ngắn", "dày", "mỏng", "cứng", "xốp"]


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_ls.pkl"
    cfg = HCAConfig()
    state, _ = load_checkpoint(ck)
    v = state.vocab
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        head_rule=cfg.hub_chain_head_rule, head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, head_gidf_q=cfg.hub_chain_head_gidf_q,
        label_indeg_q=cfg.hub_chain_indeg_q, gop_manh_tu=cfg.hub_chain_gop_manh_tu,
        pos_map=H.load_pos_map(cfg.hub_chain_pos_dict),
        # ⚠️ HAI THAM SỐ NÀY BẢN ĐẦU THIẾU (§92.1). `hub_chain_proposals` truyền
        # cả `kenh_vi_ngu` lẫn `chu_ngu_ket_hop_q` (generate_hebbian.py ~2863).
        # Thiếu `chu_ngu_ket_hop_q` ⇒ không lọc chủ ngữ ⇒ via_pivot đầy rác
        # (`Nước là .<EOS>`, `con là 3`, `Ngọc là 2`) và mọi đề xuất thành rác.
        # Lần thứ BA trong phiên probe dựng sai cấu hình (§85, §87, nay §92).
        kenh_vi_ngu=bool(cfg.hub_chain_kenh_vi_ngu),
        chu_ngu_ket_hop_q=float(cfg.hub_chain_chu_gia_q))
    va = getattr(idx, "via_attr", None) or {}
    vp = idx.via_pivot

    def sf(t):
        return (idx.label_surf.get(int(t))
                or (H._surf(v, int(t)) if int(t) < len(v) else "")).strip()

    # ③ đảo: lớp → thành viên
    tv: dict[str, set[str]] = defaultdict(set)
    for s, ds in vp.items():
        a = sf(s)
        for d in ds:
            if a:
                tv[sf(d)].add(a)

    print(f"\n### {ck} · via_attr {sum(len(d) for d in va.values())} cạnh"
          f" · via_pivot {sum(len(d) for d in vp.values())} cạnh"
          f" · {len(tv)} lớp có thành viên\n")

    dem_lop: Counter = Counter()
    n_co_bac = n_tong = 0
    for A in THU:
        ids = H.ids_theo_be_mat(idx, v, A) or []
        mang: set[str] = set()
        for i in ids:
            for d in va.get(int(i), {}):
                if (x := sf(d)):
                    mang.add(x)
        if not mang:
            print(f"── «{A} như ___»  ① KHÔNG có vật mang")
            continue
        n_tong += 1
        dong = []
        for b in sorted(mang)[:6]:
            lops = set()
            for i in (H.ids_theo_be_mat(idx, v, b) or []):
                for d in vp.get(int(i), {}):
                    if (c := sf(d)):
                        lops.add(c)
            for C in sorted(lops):
                anh = sorted(tv.get(C, set()) - mang - {b})
                if anh:
                    dem_lop[C] += 1
                    dong.append((b, C, anh[:5]))
        if dong:
            n_co_bac += 1
        print(f"── «{A} như ___»  vật mang: {sorted(mang)[:6]}")
        for b, C, anh in dong[:4]:
            print(f"     qua «{b} là {C}»  ⇒ đề xuất: "
                  + ", ".join(f"«{A} như {m}»" for m in anh[:4]))
        if not dong:
            print("     (không bắc được cầu nào)")

    print(f"\n  {n_co_bac}/{n_tong} thuộc tính bắc được cầu")
    print(f"  LỚP dùng để bắc cầu, phổ biến nhất: {dem_lop.most_common(12)}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
