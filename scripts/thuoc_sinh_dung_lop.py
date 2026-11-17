"""THƯỚC SINH v3 — đề cử có ĐÚNG LỚP của chủ ngữ không? (§70)

La bàn cũ («② chọn đúng lớp so với chân lý») đo một đại lượng TRUNG GIAN, và
§67.D cho thấy nó nói dối được: `«cây bàng là ___»` → `Chó·Dê·Sư·Báo` — chuỗi
có thể đúng lớp mà đầu ra vẫn sai loài.

Thước này chấm thẳng thứ đường sinh đưa ra:

    prompt  = «<thành viên> là»              (câu định nghĩa thật)
    chuẩn   = lớp của thành viên đó + các thành viên KHÁC của lớp
    đạt     = trong top-k đề cử có lớp đúng HOẶC một đồng-thành-viên

Tách hai nhóm — thứ mà mọi thước trước gộp chung:

    NHỚ  — thành viên ĐÃ có cạnh `via_pivot` tới lớp (chuỗi chỉ cần tra lại)
    SUY  — thành viên CHƯA có cạnh đó (chuỗi phải suy ra) ← đây mới là «con khỉ»

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/thuoc_sinh_dung_lop.py \
        checkpoints/classrich2.pkl [vingu] [n]
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import generate_hebbian as G  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classrich2.pkl"
    n_ca = int([a for a in sys.argv[2:] if a.isdigit()][0]) if any(
        a.isdigit() for a in sys.argv[2:]) else 300
    cfg = HCAConfig()
    cfg.hub_chain_boost = 4.0
    for _a in sys.argv[2:]:
        if _a.startswith("boost="):
            cfg.hub_chain_boost = float(_a[6:])
    cfg.hub_chain_kenh_vi_ngu = "vingu" in sys.argv
    cfg.hub_chain_hat_dong_thuan = "dt" in sys.argv
    for a in sys.argv[2:]:
        if a.startswith("cd="):
            cfg.hub_chain_cong_don = int(a[3:])
        if a.startswith("tt="):
            cfg.hub_chain_nhan_truc_tiep = float(a[3:])
        if a == "gopmanh":
            cfg.hub_chain_gop_manh_tu = True
        if a.startswith("gt="):
            cfg.hub_chain_gom_tu = int(a[3:])
        if a.startswith("nb="):
            cfg.hub_chain_n_bien_the = int(a[3:])
        if a.startswith("kt="):
            cfg.hub_chain_ke_truc_tiep = float(a[3:])
        if a.startswith("tv="):
            cfg.hub_chain_tv_truc_tiep = float(a[3:])
        if a.startswith("ke="):        # §60 kề ngữ cảnh ở ③
            cfg.hub_chain_ke_ctx = float(a[3:])
        if a.startswith("hq="):        # §64 ngưỡng chọn đầu
            cfg.hub_chain_head_gidf_q = float(a[3:])
    state, _ = load_checkpoint(ck)
    v = state.vocab
    ten = ("out/chan_ly_v2_classrich2.json" if "classrich2" in ck
           else "out/chan_ly_v2_classrich.json")
    lop = json.load(open(ten))["lop"]
    print(f"\n### {ck} · {len(state.parents.parents)} cha · "
          f"vị_ngữ={cfg.hub_chain_kenh_vi_ngu} · cộng_dồn={cfg.hub_chain_cong_don}"
          f" · trực_tiếp={cfg.hub_chain_nhan_truc_tiep}"
          f" · kề={cfg.hub_chain_ke_ctx} · hq={cfg.hub_chain_head_gidf_q}\n")

    # chỉ số để biết ca là NHỚ hay SUY
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    wsp = H.load_ws_spans(ck.replace(".pkl", ".ws_spans.json"))
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        ws_spans=wsp, head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, head_gidf_q=cfg.hub_chain_head_gidf_q,
        label_indeg_q=cfg.hub_chain_indeg_q, kenh_vi_ngu=cfg.hub_chain_kenh_vi_ngu,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm,
        gop_manh_tu=cfg.hub_chain_gop_manh_tu,
        lop_trang=(set(json.load(open("out/lop_tu_dien.json")))
                   if "lopdt" in sys.argv else None))

    def sf(t: int) -> str:
        return idx.label_surf.get(int(t)) or (
            H._surf(v, int(t)) if int(t) < len(v) else "")

    # KIỂM ĐỘ BỀN: chỉ chấm trên lớp SINH VẬT/CỤ THỂ. Lớp chuẩn của nhiều ca
    # là `phần`/`khoảng`/`bộ phận` — nếu cơ chế chỉ đang đề cử nhãn chung chung
    # thì nó khớp chuẩn chung chung mà không biết gì. Tập con này không có chỗ
    # cho kiểu ăn may đó.
    _CT = {"chim", "thú", "cây", "cá", "loài", "hoa", "quả", "con", "giống",
           "kim loại", "hành tinh", "tỉnh", "thành phố", "sông", "núi"}
    chi_ct = "cuthe" in sys.argv
    ca: list[tuple[str, str, set[str]]] = []
    for c, ms in sorted(lop.items(), key=lambda kv: -len(kv[1])):
        ms = [m for m in ms if H.resolve_surface_ids(v, m)]
        if len(ms) < 3 or (chi_ct and c.split()[0] not in _CT):
            continue
        for m in ms[:6]:
            ca.append((m, c, set(ms) - {m}))
    ca = ca[:n_ca]
    print(f"  {len(ca)} ca · {len({c for _, c, _ in ca})} lớp\n")

    ket = {"NHỚ": [0, 0], "SUY": [0, 0]}
    top5 = {"NHỚ": 0, "SUY": 0}
    vi_du: list[str] = []
    truot: list[int] = []
    for m, c, ban in ca:
        ids = [int(t) for t in v.encode_prompt(f"{m} là")]
        # NHỚ hay SUY: thành viên đã có cạnh tới lớp chưa?
        # ⚠️ SO KHỚP BỀ MẶT CHÍNH XÁC là chưa đủ: cạnh trỏ tới BIẾN THỂ của
        # lớp («cây thân gỗ» vs `cây`) sẽ bị gán nhầm là SUY, và ca đó thực ra
        # chỉ là NHỚ. Tách nghiêm: coi là NHỚ nếu cạnh trỏ tới bất kỳ nhãn nào
        # CHIA TỪ với lớp chuẩn.
        _ctok = set(c.lower().split())
        co_canh = False
        for i in (H.resolve_surface_ids(v, m) or [])[:4]:
            for d in idx.via_pivot.get(int(i), {}):
                _sd = sf(d).lower()
                if _sd == c.lower() or (_ctok & set(_sd.split())):
                    co_canh = True
        nhom = "NHỚ" if co_canh else "SUY"
        de = G.hub_chain_proposals(state, ids, cfg)
        top = [sf(t) for t, _ in sorted(de.items(), key=lambda kv: -kv[1])[:10]]
        dung = {c, c.split()[0]} | ban
        # CHẨN ĐOÁN ca trượt: lớp đúng có nằm trong cạnh trực tiếp của X không?
        # «không TÌM được» (có cạnh, bị xếp sau) vs «không CÓ» (thiếu cạnh).
        _co = False
        for i in (H.resolve_surface_ids(v, m) or [])[:4]:
            for d in idx.via_pivot.get(int(i), {}):
                if sf(d) in dung:
                    _co = True
        ket[nhom][1] += 1
        if top and top[0] in dung:
            ket[nhom][0] += 1
        if any(x in dung for x in top[:5]):
            top5[nhom] += 1
        else:
            truot.append(1 if _co else 0)
        if len(vi_du) < 6 and nhom == "SUY":
            vi_du.append(f"     «{m} là ___» lớp={c!r} → {top[:6]}")

    for k in ("NHỚ", "SUY"):
        d, t = ket[k]
        if t:
            print(f"  {k}: {t:>4} ca · top-1 ĐÚNG LỚP {100*d/t:>5.1f}% · "
                  f"top-5 {100*top5[k]/t:>5.1f}%")
    tong_t = sum(x[1] for x in ket.values())
    tong_d = sum(x[0] for x in ket.values())
    tong5 = sum(top5.values())
    print(f"  TỔNG: {tong_t} ca · top-1 {100*tong_d/max(1,tong_t):.1f}% · "
          f"top-5 {100*tong5/max(1,tong_t):.1f}%")
    if truot:
        print(f"\n  CA TRƯỢT top-5: {len(truot)} · trong đó CÓ cạnh trực tiếp"
              f" tới lớp đúng mà vẫn trượt: {sum(truot)} "
              f"({100*sum(truot)/len(truot):.1f}%) ⇒ lỗi XẾP HẠNG; "
              f"còn lại {len(truot)-sum(truot)} là THIẾU CẠNH")
    print("\n  6 ca SUY (đọc bằng mắt):")
    print("\n".join(vi_du))
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
