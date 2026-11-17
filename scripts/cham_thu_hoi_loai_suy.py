"""TK_052 — đo TỈ LỆ THU HỒI thành viên bị giữ lại («con khỉ» của mỗi bộ).

382 bộ, mỗi bộ có đúng cấu trúc ca vàng và một thành viên `m3` KHÔNG BAO GIỜ
đứng cùng thuộc tính. Prompt «<A> như loài ___» có ra `m3` không?

CỔNG (nêu trước, TK_052):
    ≥50% bộ có m3 trong top-10  ⇒ H2: cơ chế làm được, dữ liệu tự nhiên không
                                  có hình dạng đó
    <10%                        ⇒ H1: cơ chế sai, ca thử vàng là hiện vật n=1
    10–50%                      ⇒ chưa phân định

Chấm CHUỖI BA BƯỚC THUẦN (`tt=0 kt=0`) — §73 đã cho thấy đường thẳng của §71/§72
làm được việc mà chuỗi không làm, nên phải TẮT chúng thì mới đo đúng cơ chế.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/cham_thu_hoi_loai_suy.py
"""
from __future__ import annotations

import gc
import json
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


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/loai_suy.pkl"
    cfg = HCAConfig()
    cfg.hub_chain_boost = 4.0
    # CHUỖI THUẦN: tắt đường thẳng §71/§72 (mặc định TK_046 bật chúng).
    thuan = "duongthang" not in sys.argv
    if thuan:
        cfg.hub_chain_nhan_truc_tiep = 0.0
        cfg.hub_chain_ke_truc_tiep = 0.0
    for _a in sys.argv[2:]:
        if _a.startswith("nb="):
            cfg.hub_chain_n_bien_the = int(_a[3:])
        if _a.startswith("gt="):
            cfg.hub_chain_gom_tu = int(_a[3:])
        if _a == "gopmanh":
            cfg.hub_chain_gop_manh_tu = True
        if _a == "giumanh":
            cfg.hub_chain_bo_manh = False
        if _a.startswith("k1="):
            cfg.hub_chain_topk_1 = int(_a[3:])
    _kf = ("out/so_sanh_that_khoa_cuthe.json" if "cuthe" in sys.argv
           else "out/so_sanh_that_khoa.json" if "that" in sys.argv
           else "out/loai_suy_khoa.json")
    khoa = json.load(open(_kf))
    state, _ = load_checkpoint(ck)
    v = state.vocab
    print(f"\n### {ck} · {len(state.parents.parents)} cha · {len(khoa)} bộ · "
          f"chuỗi {'THUẦN' if thuan else '+ đường thẳng'}\n")

    idx_surf = {}
    d1 = d5 = d10 = 0
    n = 0
    lop_dung = 0
    vi_du: list[str] = []
    doi_chung = [0, 0]
    d30 = [0, 0]
    for k in khoa:
        _pr = (f"{k['attr']} như" if "khongloai" in sys.argv
               else f"{k['attr']} như loài")
        ids = [int(t) for t in v.encode_prompt(_pr)]
        if not ids:
            continue
        n += 1
        de = G.hub_chain_proposals(state, ids, cfg)
        top = []
        for t, _w in sorted(de.items(), key=lambda kv: -kv[1])[:30]:
            s = idx_surf.get(int(t))
            if s is None:
                s = H._surf(v, int(t)) if int(t) < len(v) else ""
                idx_surf[int(t)] = s
            top.append(s)
        # m3 có thể là CỤM: sinh tự hồi quy nên đích là TOKEN ĐẦU của nó
        dich = k["m3"].split()[0]
        if top[:1] == [dich]:
            d1 += 1
        if dich in top[:5]:
            d5 += 1
        if dich in top[:10]:
            d10 += 1
        if dich in top[:30]:
            d30[0] += 1
        d30[1] += 1
        if k["lop"].split()[0] in top[:10]:
            lop_dung += 1
        # ĐỐI CHỨNG: m3 của bộ KHÁC. Nếu tỉ lệ ngang nhau thì đề cử KHÔNG phụ
        # thuộc prompt và con số thu hồi là ăn may.
        gia = khoa[(khoa.index(k) + 37) % len(khoa)]["m3"].split()[0]  # noqa
        if gia in top[:10]:
            doi_chung[0] += 1
        doi_chung[1] += 1
        if len(vi_du) < 8:
            vi_du.append(f"     «{_pr} ___» giữ lại={k['m3']!r} "
                         f"lớp={k['lop']!r} → {top[:6]}")
    print(f"  {n} bộ chấm được")
    print(f"  THU HỒI m3:  top-1 {100*d1/max(1,n):>5.1f}%  ·  "
          f"top-5 {100*d5/max(1,n):>5.1f}%  ·  top-10 {100*d10/max(1,n):>5.1f}%")
    print(f"  top-30: {100*d30[0]/max(1,d30[1]):.1f}%  ·  bể đề cử trung bình "
          f"{sum(1 for _ in ()) or ''}")
    print(f"  ĐỐI CHỨNG m3 của bộ KHÁC trong top-10: "
          f"{100*doi_chung[0]/max(1,doi_chung[1]):.1f}%")
    print(f"  (đối chiếu) LỚP đúng trong top-10: {100*lop_dung/max(1,n):.1f}%")
    print("\n  8 bộ đầu:")
    print("\n".join(vi_du))
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
