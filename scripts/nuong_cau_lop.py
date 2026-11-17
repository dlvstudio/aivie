"""§143 — NƯỚNG CÂU LỚP vào kho cha, để đường sinh QUÉT như mọi kết nối khác.

Phát hiện §143.1: `vote_logits` → `W_skip`, mà `build_w_skip` dựng từ
`ps.parents` — KHO CÂU CHA, **không** từ `TypedAdjacency`. Nên thêm cạnh vào
bảng kề KHÔNG ảnh hưởng gì tới sinh câu. «Kết nối trong mạng» mà đường sinh
thấy = **cặp từ cùng xuất hiện trong một câu cha, trong khoảng cách d_max**.

⇒ Muốn `sắt`↔`kẽm` là kết nối THẬT (không suy lúc sinh) thì chúng phải CÙNG
NẰM trong một câu cha. Script này sinh câu lớp từ bảng nhãn §128 và nạp qua
`add_example` — đúng đường ingest chuẩn, nên bản ghi có `slot_chain`,
`match_ids` hợp lệ như mọi câu thật.

Khuôn câu: «A, B, C và D đều là <LỚP>» — đúng khuôn loại suy `chó và mèo đều
là`, đặt mọi thành viên trong cùng cửa sổ d_max VÀ nối chúng với nhãn lớp.

§142 đã đo: đồng hiện bậc hai cho HÀNG XÓM CHỦ ĐỀ (`muối → salad/rau`), không
cho ANH EM CÙNG LỚP. Khuôn định nghĩa mã hoá quan hệ lớp TRỰC TIẾP (§128:
84,9% vs 4,1%) — nên nguồn phải là BẢNG, không phải đồng hiện.

    PYTHONPATH=src python scripts/nuong_cau_lop.py CKPT --ra CKPT_MOI
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402

T0 = time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--bang", default="data/dict/nhan_lop.json")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--moi-cau", type=int, default=6,
                    help="số thành viên mỗi câu (giữ trong cửa sổ d_max)")
    ap.add_argument("--toi-thieu", type=int, default=2,
                    help="lớp phải có ít nhất ngần này thành viên")
    ap.add_argument("--hoan-vi", type=int, default=2,
                    help="số lượt hoán vị thêm cho lớp lớn (phủ cặp XA)")
    ap.add_argument("--uu-tien", type=float, default=0.80,
                    help="priority: thấp hơn câu thật (curated=1.0)")
    a = ap.parse_args()

    bang = json.loads(Path(a.bang).read_text("utf-8"))
    lop2tu: dict[str, list[str]] = defaultdict(list)
    for w, ls in bang.items():
        for x in ls:
            p = x.split()
            if len(p) >= 2:                 # nhãn GHÉP: cụ thể hơn `thú`/`cây`
                lop2tu[" ".join(p[:2])].append(w)
                break                        # mỗi từ vào ĐÚNG một lớp
    lop2tu = {k: sorted(set(v)) for k, v in lop2tu.items()
              if len(set(v)) >= a.toi_thieu}
    print(f"### {len(lop2tu)} lớp có ≥{a.toi_thieu} thành viên")

    state, _meta = load_checkpoint(a.ckpt)
    ps, vocab = state.parents, state.vocab
    truoc = len(ps.parents)
    ps.seg.begin_batch()
    n_cau = 0
    # §144 — CỬA SỔ TRƯỢT + HOÁN VỊ. Bản §143 cắt KHỐI RỜI `range(0,n,moi_cau)`
    # nên `sắt` (khối 1) và `đồng` (khối 2) KHÔNG đồng hiện lần nào — chính phép
    # chia khối phá đúng quan hệ đang cần tạo (đo: 0 câu chung ⇒ «sắt và đồng
    # đều là» → «cuộc họp»). Trượt chồng 50% phủ cặp gần; hoán vị phủ cặp xa.
    import random as _rd
    rng = _rd.Random(0)
    buoc = max(1, a.moi_cau // 2)
    for lop, tv in sorted(lop2tu.items()):
        nhom_ds: list[list[str]] = []
        for i in range(0, max(1, len(tv) - 1), buoc):
            nhom_ds.append(tv[i:i + a.moi_cau])
        for _ in range(a.hoan_vi if len(tv) > a.moi_cau else 0):
            x = list(tv); rng.shuffle(x)
            nhom_ds += [x[i:i + a.moi_cau] for i in range(0, len(x), a.moi_cau)]
        for nhom in nhom_ds:
            if len(nhom) < 2:
                continue
            cau = f"{', '.join(nhom[:-1])} và {nhom[-1]} đều là {lop}."
            ids = [int(t) for t in vocab.encode_prompt(cau)]
            if not ids:
                continue
            ps.add_example(ids, vocab, dap_id=None,
                           priority=a.uu_tien, defer_index=True)
            n_cau += 1
            if n_cau <= 3:
                print(f"    ví dụ: {cau}")
    ps.seg.end_batch(ps.parents)
    ps.rebuild_hierarchical_index()
    print(f"\n### nạp {n_cau:,} câu lớp · cha {truoc:,} → {len(ps.parents):,}")
    save_checkpoint(state, a.ra, bake_derived=True)
    print(f"### lưu → {a.ra}")
    print(f"[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
