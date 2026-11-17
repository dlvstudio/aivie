"""§147 — NƯỚNG CẠNH c→Y: cho `c` THỪA HƯỞNG mô tả của anh em cùng lớp.

Người dùng: «phải có cơ chế để c nối với y, hoặc y nối với c».

§143 đo được: đường sinh đọc KHO CÂU CHA, không đọc `TypedAdjacency`. Nên «cạnh
c→Y» mà nó THẤY = `c` và `Y` cùng nằm trong một câu cha. §144 mới nướng
`c ↔ a` và `c ↔ nhãn lớp` (ngữ cảnh chung) — KHÔNG mang nội dung sang, và §146
đo được chuyển giao 0/60.

Cơ chế ở đây thực hiện đúng mẫu người dùng:

    X … a, X … b, X … c   ← X = CÙNG LỚP (bảng §128, top-1 84,9%)
    a … Y, b … Y          ← Y = nội dung ĐỊNH NGHĨA THẬT của anh em
    ⟹ nướng câu «c là <định nghĩa thừa hưởng>»   ← cạnh c…Y THẬT trong kho câu

Chọn câu thừa hưởng: lấy định nghĩa của anh em có ĐỘ TRÙNG CAO NHẤT với `Y`
(giao ≥2 anh em) rồi thay chủ ngữ. Túi-từ thì sai ngữ pháp («kiếm là binh khí
binh khí dùng cán dài»); thừa hưởng câu thì đúng ngữ pháp VÀ giữ đúng thuộc
tính đồng thuận.

⚠️ CƠ CHẾ NÀY BỊA NỘI DUNG CHO `c`. Đo được ca sai: `lao` nhận `virus` từ
`cúm`/`sởi`, trong khi lao do vi khuẩn Koch. Đòi ≥2 đồng thuận KHÔNG chặn được
vì cả hai anh em cùng sai kiểu. Đây là rủi ro cố hữu của loại suy (người dùng đã
nêu: «dù c có thể yếu hơn») ⇒ priority THẤP, và phải chấm bằng 60 từ giữ riêng.

    PYTHONPATH=src python scripts/nuong_canh_cy.py CKPT --ra CKPT_MOI
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402

T0 = time.time()
HU = {"là", "và", "có", "của", "một", "các", "những", "để", "trong", "cho",
      "không", "hay", "với", "được", "thì", "này", "đó", "như", "khi", "ở"}


def _tu(s: str) -> set[str]:
    return {x for x in re.findall(r"[\wÀ-ỹ]+", (s or "").lower())
            if x not in HU and len(x) > 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--dong-thuan", type=int, default=2,
                    help="số anh em tối thiểu phải cùng có một từ thì mới tính vào Y")
    ap.add_argument("--anh-em-toi-da", type=int, default=5)
    ap.add_argument("--uu-tien", type=float, default=0.70,
                    help="THẤP hơn câu thật (1.0) và câu lớp §144 (0.80)")
    ap.add_argument("--chi-giu-rieng", default="",
                    help="chỉ nướng cho các từ trong tệp này (để chấm sạch)")
    a = ap.parse_args()

    # định nghĩa THẬT — nguồn của Y
    dn: dict[str, str] = {}
    for ln in Path("data/text_corpus/dictionary.jsonl").open(encoding="utf-8"):
        try:
            t = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(t, str):
            continue
        m = re.search(r"^Từ:\s*(.+)$", t, re.M)
        g = re.search(r"^Giải nghĩa:\s*(.+)$", t, re.M)
        if m and g:
            dn.setdefault(m.group(1).strip().lower(),
                          re.sub(r"\([^)]*\)", " ", g.group(1)).split(":")[0].strip())

    bang = json.loads(Path("data/dict/nhan_lop.json").read_text("utf-8"))
    lop2tu: dict[str, list[str]] = defaultdict(list)
    for w, ls in bang.items():
        for x in ls:
            p = x.split()
            if len(p) >= 2:
                lop2tu[" ".join(p[:2])].append(w)
                break

    chi = set()
    if a.chi_giu_rieng:
        chi = set(json.loads(Path(a.chi_giu_rieng).read_text("utf-8")))
        print(f"### chỉ nướng cho {len(chi)} từ giữ riêng")

    state, _m = load_checkpoint(a.ckpt)
    ps, vocab = state.parents, state.vocab
    truoc = len(ps.parents)
    ps.seg.begin_batch()
    n = 0
    vd: list[str] = []
    for lop, tv in sorted(lop2tu.items()):
        tv = sorted(set(tv))
        for c in tv:
            if chi and c not in chi:
                continue
            ae = [x for x in tv if x != c and x in dn][:a.anh_em_toi_da]
            if len(ae) < a.dong_thuan:
                continue
            cnt: Counter[str] = Counter()
            for x in ae:
                cnt.update(_tu(dn[x]))
            Y = {w for w, k in cnt.items() if k >= a.dong_thuan}
            if len(Y) < 3:
                continue
            # câu THỪA HƯỞNG = định nghĩa anh em trùng Y nhiều nhất
            best = max(ae, key=lambda x: len(_tu(dn[x]) & Y))
            mo_ta = dn[best].strip().rstrip(".")
            if not mo_ta or len(mo_ta) < 6:
                continue
            cau = f"{c} là {mo_ta}."
            ids = [int(t) for t in vocab.encode_prompt(cau)]
            if not ids:
                continue
            ps.add_example(ids, vocab, dap_id=None,
                           priority=a.uu_tien, defer_index=True)
            n += 1
            if len(vd) < 5:
                vd.append(f"    {cau[:96]}")
    ps.seg.end_batch(ps.parents)
    ps.rebuild_hierarchical_index()
    print(f"### nướng {n:,} câu c→Y · cha {truoc:,} → {len(ps.parents):,}")
    for x in vd:
        print(x)
    save_checkpoint(state, a.ra, bake_derived=True)
    print(f"### lưu → {a.ra}\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
