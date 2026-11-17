"""§84 A/B — bậc-2 lấy từ CHA ĐANG HOẠT thay vì gộp toàn cục.

§83 nối `ParentStore.seq_next` (gộp trên CẢ 450 050 cha) vào đường sinh thật.
Trật tự lên thật — 89,3% → 98,0% ở w=2 — nhưng đổi bằng TRI THỨC:

    w=0  chó là gia súc thuộc nhóm ăn thịt, nuôi để giữ nhà hay đi săn .
    w=2  chó là gia đình và bạn bè và súc tích cực trong cuộc sống của chúng ta

Vì gộp toàn cục thì `có thể được sử dụng để tạo ra một` thống trị. Bản này chỉ
đọc chuỗi của CHA ĐANG HOẠT (`_active_parent_set` — đúng câu nguồn mà prompt đã
truy ra), nên trật tự lẽ ra không phải trả giá bằng tri thức.

THƯỚC MỚI — §83.6 bài học 124: thước TRẬT TỰ thưởng cho VĂN MẪU trôi chảy, nên
nó phong quán quân cho đúng mức xoá sạch tri thức. Vòng này thêm thước bắt được
chính điều đó:

    TRUNG THÀNH = tỉ lệ TỪ sinh ra có mặt trong CÂU NGUỒN THẬT của prompt
                  (`data/text_corpus/dinh_nghia.jsonl`)

`chó là gia đình và bạn bè` ⇒ thấp. `chó là gia súc thuộc nhóm ăn thịt` ⇒ cao.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ Một mức ĐÚNG HƯỚNG chỉ khi ĐỒNG THỜI:                                    ║
║   (a) TRUNG THÀNH **không giảm** so với mức 0 — vế PHỦ QUYẾT. Mức nào    ║
║       tụt trung thành thì BỊ LOẠI bất kể trật tự đẹp tới đâu (§83.3).    ║
║   (b) TRẬT TỰ tăng, và                                                    ║
║   (c) LẶP không tăng, và                                                  ║
║   (d) 4 ca vàng không hỏng khi đọc bằng MẮT.                             ║
║ Không mức nào qua ⇒ ÂM TÍNH, để mặc định tắt, ghi lại.                   ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/ab_thu_tu_cha_hoat.py \
        checkpoints/network_v12_sent_dn.pkl
"""
from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import generate_hebbian as G  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
MAX = 24
# §126 — bộ hạt giống mặc định. `hat=N` trên dòng lệnh để đổi số hạt.
SEEDS = list(range(6))
NGUON = "data/text_corpus/dinh_nghia.jsonl"

# §155 — BỘ PROMPT THEO NHÓM. Bản cũ có 19/20 là khuôn «X là» nên cổng §151
# KHÔNG thấy 0,15 cắt «muối là» thành «sản phẩm axit» (SAI) và bốn prompt hoá
# RỖNG. Cùng lỗi §113 ở tầng khác: ở đó THƯỚC mù với thứ tự, ở đây BỘ PROMPT mù
# với thể loại. Chấm theo từng nhóm để hỏng ở nhóm nào cũng lộ.
NHOM: dict[str, list[str]] = {
    "ĐỊNH NGHĨA": [
        "chó là", "mèo là", "sóc là", "hổ là", "sắt là", "đồng là",
        "chim sẻ là", "lúa là", "muối là", "cây bàng là", "con tê tê là",
        "nước là", "mặt trời là", "khỉ là", "bướm là",
    ],
    "HỎI": [
        "thủ đô của Việt Nam là", "vì sao trời mưa?",
        "một năm có bao nhiêu", "ai là người",
    ],
    "LOẠI SUY": [
        "nghịch ngợm như loài", "chó và mèo đều là", "sắt và đồng đều là",
        "chó nghịch ngợm, mèo nghịch ngợm, còn", "hổ giống mèo vì",
        "khoẻ như", "nhanh như",
    ],
    "MỞ": ["mùa xuân", "học sinh", "con người cần", "buổi sáng hôm nay"],
}
PROMPT = [p for ps in NHOM.values() for p in ps]
VANG = ["chó là", "sắt là", "hổ là", "lúa là"]


def _tu(s: str) -> list[str]:
    """§112 — BỎ token thuần CHỮ SỐ, áp ĐỐI XỨNG cho nguồn và sinh.

    §111.4: `w=1` sinh «khoảng cách lên tới 6» — `6` là mảnh của `650` bị `max_len`
    cắt, nên thước TRỪ ĐIỂM đúng phần sinh chính xác hơn. Thước này đo «mạng có
    phát ra TRI THỨC của câu nguồn không»; con số không phải tri thức đang xét.
    Bỏ ở CẢ HAI vế nên không thiên vị cấu hình nào.

    Luật khai TRƯỚC khi chạy lại — không phải chỉnh cho vừa số (bài học 164).
    """
    return [w for w in re.findall(r"\w+", s.lower())
            if w and not w.isdigit()]


def nap_nguon() -> dict[str, set[str]]:
    """prompt → tập TỪ của mọi câu định nghĩa thật bắt đầu bằng prompt đó."""
    ra: dict[str, set[str]] = {p: set() for p in PROMPT}
    q = Path(NGUON)
    if not q.exists():
        return ra
    for ln in q.open(encoding="utf-8"):
        try:
            t = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(t, str):
            t = str(t)
        low = t.lower()
        for p in PROMPT:
            if low.startswith(p.lower()):
                ra[p] |= set(_tu(t))
    return ra


def main() -> int:
    global SEEDS
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_dn.pkl"
    for _a in sys.argv[2:]:
        if _a.startswith("hat="):
            SEEDS = list(range(int(_a[4:])))
    state, _ = load_checkpoint(ck)
    v = state.vocab
    ps = state.parents
    ps._ensure_seq_index()          # noqa: SLF001 — `_seq_bi` làm thước TRẬT TỰ
    bi = ps._seq_bi                 # noqa: SLF001
    nguon = nap_nguon()
    co_nguon = [p for p in PROMPT if nguon[p]]
    print(f"\n### {ck} · {len(ps.parents)} cha · bigram-cha {len(bi)} hàng")
    print(f"### có câu nguồn thật: {len(co_nguon)}/{len(PROMPT)} prompt "
          f"→ {sorted(co_nguon)}")

    if "cuaso" in sys.argv:
        # §111 — cửa sổ thứ tự agenda. §82 đo được mạng truy ĐÚNG câu nguồn rồi
        # mất trật tự; agenda hoá ra HOÀN HẢO, lỗi ở chỗ mỗi bước được chọn bất
        # kỳ trong `ag_win` mục kế tiếp.
        muc = [(f"window={w}", dict(ar_w_skip_agenda_order_window=w))
               for w in (4, 3, 2, 1)]
    elif "motlan" in sys.argv:
        # §140 — hs=1,0 là vùng DUY NHẤT chuỗi thắng được (§138). Hai ràng buộc
        # riêng và kết hợp: cổng vị trí (§139) + một-lần-mỗi-từ (§140).
        _n = dict(hub_chain_bang_tv_w=0.5, ar_w_skip_chain_hoi_sinh=1.0)
        muc = [("mốc (hs=0)", dict(hub_chain_bang_tv_w=0.5)),
               ("hs=1,0 trần", dict(_n)),
               ("hs=1,0 +vị trí", dict(_n, ar_w_skip_chain_vi_tri=True)),
               ("hs=1,0 +một lần", dict(_n, ar_w_skip_chain_mot_lan=True)),
               ("hs=1,0 +CẢ HAI", dict(_n, ar_w_skip_chain_vi_tri=True,
                                       ar_w_skip_chain_mot_lan=True))]
    elif "dotin" in sys.argv:
        # §151 — DỪNG THEO ĐỘ TIN. Cổng: «mặt trời» hết đuôi rác (ĐẠT, đo bằng
        # mắt); ca vàng không cụt sớm; ba thước không xấu quá biên độ §125.
        muc = [(f"độ_tin={x}", dict(ar_w_skip_do_tin_min=x))
               for x in (0.0, 0.05, 0.10, 0.15)]
    elif "vitri" in sys.argv:
        # §139 — cổng VỊ TRÍ có làm «vùng hữu ích» (hồi_sinh cao) trả nổi giá?
        muc = [("mốc (hồi_sinh=0)", dict(hub_chain_bang_tv_w=0.5)),
               ("hs=0,5 không cổng",
                dict(hub_chain_bang_tv_w=0.5, ar_w_skip_chain_hoi_sinh=0.5)),
               ("hs=0,5 CÓ cổng",
                dict(hub_chain_bang_tv_w=0.5, ar_w_skip_chain_hoi_sinh=0.5,
                     ar_w_skip_chain_vi_tri=True)),
               ("hs=1,0 không cổng",
                dict(hub_chain_bang_tv_w=0.5, ar_w_skip_chain_hoi_sinh=1.0)),
               ("hs=1,0 CÓ cổng",
                dict(hub_chain_bang_tv_w=0.5, ar_w_skip_chain_hoi_sinh=1.0,
                     ar_w_skip_chain_vi_tri=True))]
    elif "hoisinh" in sys.argv:
        # §138 — GIÁ của việc nới §12.144: quét thang hồi sinh.
        muc = [(f"hồi_sinh={x}",
                dict(ar_w_skip_chain_hoi_sinh=x, hub_chain_bang_tv_w=0.5))
               for x in (0.0, 0.25, 0.5, 1.0)]
    elif "bangtv" in sys.argv:
        # §136 — đảo-bảng gieo ③. Cổng SINH CÂU: không được xấu đi so với mốc.
        muc = [(f"bang_tv_w={x}", dict(hub_chain_bang_tv_w=x))
               for x in (0.0, 2.0)]
    elif "loc" in sys.argv:
        # §93 — vế PHỦ QUYẾT cho bộ lọc đích: TRUNG THÀNH không được tụt 78,9%
        muc = [("gốc (mốc)", dict(hub_chain_loc_dich=False)),
               ("LỌC ĐÍCH", dict(hub_chain_loc_dich=True))]
    elif "gop" in sys.argv:
        # §87 — CỔNG: TRUNG THÀNH đầu ra phải TĂNG. §86 chỉ đo hạng truy hồi
        # (trung gian); §83/§84 đã dạy hai lần rằng chỉ số trung gian đẹp không
        # suy ra đầu ra tốt. Đây mới là phép thử quyết định.
        muc = [("gốc (mốc)", dict(retrieval_gop_hoa_thuong=False)),
               ("GỘP hoa/thường", dict(retrieval_gop_hoa_thuong=True)),
               ("GỘP + bậc-2 cha hoạt w=1 top_n=3",
                dict(retrieval_gop_hoa_thuong=True,
                     ar_w_skip_order2_weight=1.0,
                     ar_w_skip_order2_cha_hoat=True,
                     ar_w_skip_order2_top_n=3))]
    elif "topn" in sys.argv:
        # §84.2 — w=1 cố định, quét ĐỘ RỘNG tập cha đang hoạt. Ca vàng ở
        # top_n=20 lộ cha RÁC lọt vào (`hổ là phần 1 trong chương trình lớp 5`,
        # `… và Benz thể`), nên nghi ngờ tập quá rộng chứ không phải cơ chế sai.
        muc = [("TẮT (mốc)", dict(ar_w_skip_order2_weight=0.0))] + [
            (f"CHA HOẠT w=1 top_n={k}",
             dict(ar_w_skip_order2_weight=1.0, ar_w_skip_order2_cha_hoat=True,
                  ar_w_skip_order2_top_n=k))
            for k in (1, 2, 3, 5, 10)]
    elif "reinforced" in sys.argv:
        # §178 — §171 (SUM hồi sinh loại suy khái niệm) TẮT vs BẬT. Cổng phủ
        # quyết như mọi lần: TRUNG THÀNH + PHỦ KHÔNG được tụt; nếu nhóm LOẠI SUY
        # giàu hơn mà hai thước giữ ⇒ đúng hướng. Tắt §170 (W_skip scale) để cô
        # lập §171. Cần checkpoint có cạnh REINFORCED (net_ls_concept_conf.pkl).
        import nn_aivie.hca.w_skip as _WS
        _WS.ANALOGY_TO_WSKIP_SCALE = 0.0
        _b = dict(ar_w_skip_reinforced_beta=0.6)
        muc = [("§171 TẮT (conf=0)", dict(_b, ar_w_skip_reinforced_conf=0.0)),
               ("§171 conf=0.20", dict(_b, ar_w_skip_reinforced_conf=0.20)),
               ("§171 conf=0.35", dict(_b, ar_w_skip_reinforced_conf=0.35))]
    else:
        muc = [("TẮT (mốc)", dict(ar_w_skip_order2_weight=0.0)),
               ("toàn cục w=1 (§83)", dict(ar_w_skip_order2_weight=1.0)),
               ("CHA HOẠT w=1", dict(ar_w_skip_order2_weight=1.0,
                                     ar_w_skip_order2_cha_hoat=True)),
               ("CHA HOẠT w=2", dict(ar_w_skip_order2_weight=2.0,
                                     ar_w_skip_order2_cha_hoat=True)),
               ("CHA HOẠT w=4", dict(ar_w_skip_order2_weight=4.0,
                                     ar_w_skip_order2_cha_hoat=True))]

    for ten, co in muc:
        cfg = HCAConfig()
        cfg.hub_chain_boost = 1.0
        for k, x in co.items():
            setattr(cfg, k, x)
        # MỖI LẦN CHẠY TỰ IN DÒNG CỜ CỦA CHÍNH NÓ.
        print(f"\n═══ {ten}  ·  gop={cfg.retrieval_gop_hoa_thuong}"
              f" · w={cfg.ar_w_skip_order2_weight}"
              f" · cha_hoat={cfg.ar_w_skip_order2_cha_hoat}"
              f" · top_n={cfg.ar_w_skip_order2_top_n}"
              f" · boost={cfg.hub_chain_boost}", flush=True)
        # §126 — chấm trên NHIỀU HẠT GIỐNG. §125 đo biên độ qua 6 hạt:
        # TRUNG THÀNH 2,6 · TRẬT TỰ 3,6 · LẶP 4,8 — mà `seed=0` lại là hạt TỐT
        # NHẤT về LẶP. Chấm một hạt không làm kết quả tất định, chỉ làm nó
        # THIÊN LỆCH theo hạt đó: §112 đổi mặc định bằng chênh +2,2 (< biên độ
        # 2,6) và §119 loại bằng chênh −2,6 (= biên độ) — cả hai vô nghĩa.
        # Cổng chỉ có nghĩa khi chênh lệch LỚN HƠN biên độ, nên phải in kèm.
        # §155 — THÊM «PHỦ» (recall) và «RỖNG». TRUNG THÀNH là TỈ LỆ
        # (đúng/tổng-sinh) nên CẮT SÂU HƠN thì tự tăng — §153 đo được +1,5 chỉ
        # phản ánh cắt sâu, không phải nội dung tốt hơn. PHỦ đi ngược chiều:
        # (từ nguồn có mặt trong sinh)/(tổng từ nguồn) ⇒ cắt là TỤT. Hai thước
        # cùng nhau mới bắt được mất nội dung.
        tt_h, tq_h, lp_h, pu_h, ro_h = [], [], [], [], []
        for _sd in SEEDS:
            tt_co = tt_mau = 0
            lap_tu = lap_mau = 0
            tt_tt = tt_tm = 0
            pu_co = pu_mau = 0
            n_rong = 0
            for p in PROMPT:
                n_pr = len(v.encode_prompt(p))
                ids = [int(t) for t in G.generate(state, p, max_len=MAX,
                                                  config=cfg, seed=_sd)]
                sinh = ids[n_pr:]
                if sinh:
                    lap_tu += len(sinh) - len(set(sinh))
                    lap_mau += len(sinh)
                for a, b in zip(ids[max(0, n_pr - 1):],
                                ids[max(0, n_pr - 1) + 1:]):
                    tt_mau += 1
                    tt_co += int(b in bi.get(a, ()))
                _ws = set(_tu(v.detokenize(sinh)))
                if not _ws:
                    n_rong += 1
                if nguon[p]:
                    for w in _tu(v.detokenize(sinh)):
                        tt_tm += 1
                        tt_tt += int(w in nguon[p])
                    for w in nguon[p]:          # §155 — PHỦ: chiều ngược lại
                        pu_mau += 1
                        pu_co += int(w in _ws)
                if p in VANG and _sd == SEEDS[0]:
                    print(f"   VÀNG «{p}» ▸ {v.detokenize(ids)}", flush=True)
            tt_h.append(100 * tt_tt / max(1, tt_tm))
            tq_h.append(100 * tt_co / max(1, tt_mau))
            lp_h.append(100 * lap_tu / max(1, lap_mau))
            pu_h.append(100 * pu_co / max(1, pu_mau))
            ro_h.append(float(n_rong))

        def _bc(ten: str, xs: list[float]) -> str:
            tb = sum(xs) / len(xs)
            bd = max(xs) - min(xs)
            return (f"{ten} {tb:5.1f}%±{bd / 2:.1f}"
                    f" [{min(xs):.1f}–{max(xs):.1f}]")

        print(f"   ── {len(SEEDS)} hạt · {_bc('TRUNG THÀNH', tt_h)}"
              f"  · {_bc('PHỦ', pu_h)}"
              f"  · {_bc('TRẬT TỰ', tq_h)}"
              f"  · {_bc('LẶP', lp_h)}"
              f"  · RỖNG {sum(ro_h) / len(ro_h):.1f}/{len(PROMPT)}", flush=True)

    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
