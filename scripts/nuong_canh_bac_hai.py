"""§141 — NƯỚNG CẠNH BẬC HAI vào checkpoint (không suy lúc sinh).

Mẫu người dùng nêu:

    X … a       a … Y
    X … b       b … Y      ⟹  c → Y   (và Y → c)
    X … c

Vì sao phải NƯỚNG chứ không suy lúc sinh: §136–§139 đo được đề cử suy-lúc-sinh
vào ở HẠNG HAI — nhánh hồi sinh đặt chúng ở thang TRUNG VỊ (~0,03) trong khi
cạnh thật ở ~40, nên chúng không bao giờ thắng; vặn trọng số đủ mạnh để thắng
thì câu vỡ (`chó là gia gia gia gia gia súc`). Cạnh nướng sẵn là công dân HẠNG
NHẤT: cùng bảng kề, cùng thang, đường sinh quét như mọi cạnh khác, KHÔNG phải
sửa vòng sinh.

QUY MÔ (đo §141): bao đóng thô 227,7 triệu cạnh = 185× bậc một ⇒ không nướng
thẳng được. Hai ràng buộc cắt xuống mức nướng được:

    trần bậc TG   K≥   cạnh bậc 2   so bậc 1
        73 (p90)   2    1.459.488      1,2×    ← mặc định
       179 (p95)   2    4.524.124      3,7×
       674 (p99)   2   18.230.314     14,8×

  (1) TRUNG GIAN phải mang thông tin — hư từ (`của`, `và`, `là`) có bậc ra tới
      9 631 so với trung vị 6; cho chúng làm trung gian thì mọi từ nối mọi từ.
      Cắt theo PHÂN VỊ bậc ra — dữ liệu tự quyết, không phải danh sách tay
      (Luật 0).
  (2) K trung gian PHÂN BIỆT — đúng mẫu người dùng: `a` và `b` cùng dẫn tới `Y`
      thì mới suy `c → Y`. Một trung gian là trùng hợp, hai mới là quy luật.

    PYTHONPATH=src python scripts/nuong_canh_bac_hai.py CKPT [--tran-q 0.90]
        [--k 2] [--tran-bac-ra 64] [--ra out/canh_bac_hai.pkl]
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402

T0 = time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tran-q", type=float, default=0.90,
                    help="phân vị bậc ra: trung gian bậc cao hơn bị BỎ")
    ap.add_argument("--k", type=int, default=2,
                    help="số trung gian PHÂN BIỆT tối thiểu")
    ap.add_argument("--tran-bac-ra", type=int, default=64,
                    help="giữ tối đa ngần này đích/nguồn (mạnh nhất trước)")
    ap.add_argument("--gdf-max-p", type=float, default=0.05,
                    help="bỏ token có mặt trong hơn ngần này TỈ LỆ câu cha (hư từ)")
    ap.add_argument("--cau-toi-da", type=int, default=12,
                    help="giữ tối đa ngần này token nội dung mỗi câu cha")
    ap.add_argument("--gdf-min", type=int, default=20,
                    help="SÀN: token có mặt trong ít hơn ngần này câu cha bị bỏ")
    ap.add_argument("--pmi-mu", type=float, default=1.0,
                    help="mũ chuẩn hoá theo tần suất đích: 0 = đếm thô, 1 = PMI")
    ap.add_argument("--ra", default="out/canh_bac_hai.pkl")
    a = ap.parse_args()

    state, _ = load_checkpoint(a.ckpt)
    ps = state.parents
    # §141b — QUAN HỆ NỀN: đồng hiện trong CÙNG CÂU CHA, không phải bigram.
    # Bản đầu dùng `_seq_bi` (liền kề) và cạnh ra TOÀN HƯ TỪ: `chó → để/đến/
    # trên/xuống`, `nước → chân/hơi/mất`, mẫu ngẫu nhiên `Nh → là/"/)`. Ghép
    # hai chặng LIỀN KỀ cho ra «cách nhau ~2 vị trí» = khuôn NGỮ PHÁP, không
    # phải tương tự phân bố. Mẫu người dùng viết `X … a` — cùng ngữ cảnh ở
    # khoảng cách BẤT KỲ, tức đồng hiện trong câu.
    import math
    N = max(1, len(ps.parents))
    def _gidf(t: int) -> float:
        return math.log(N / (1 + ps._gdf.get(int(t), 0)))   # noqa: SLF001
    # §141c — ngưỡng theo `gdf` TUYỆT ĐỐI, không phải phân vị của phân bố
    # `gidf` trên TỪ VỰNG. Bản trước lấy phân vị 35 của gidf và ra 10,02, tức
    # chỉ giữ token có mặt trong ≤20/450 303 câu cha ⇒ `chó` (gdf 1 316),
    # `sắt` (892), thậm chí `kẽm` (66) đều BỊ LOẠI, chỉ còn token siêu hiếm.
    # Phân bố gidf trên từ vựng lệch mạnh về phía hiếm nên phân vị của NÓ không
    # phải công cụ cắt hư từ. (§128 dùng phân vị thành công vì lấy trên đúng
    # quần thể cần cắt — từ đứng đầu lời giải nghĩa.)
    gdf_max = int(a.gdf_max_p * N)
    print(f"### bỏ token có mặt trong > {gdf_max:,} câu cha"
          f" ({a.gdf_max_p:.0%} của {N:,}) — áp cho CẢ trung gian LẪN đích")
    bi: dict[int, set] = {}
    for i, rec in enumerate(ps.parents):
        if i % 100000 == 0:
            print(f"    dựng đồng hiện {i}/{N}", flush=True)
        ws = [int(x) for x in getattr(rec, "match_ids", ()) or ()]
        # §141e — SÀN tần suất: PMI mũ 1,0 đẩy ngược sang từ SIÊU HIẾM nên
        # mảnh BPE (`ugula`,`öpp`, gdf 2–10) nổi lên trên `thiếc`/`quặng`.
        # Đếm thô nổi từ PHỔ BIẾN, PMI nổi từ SIÊU HIẾM — hai bệnh đối xứng;
        # lời giải là PMI CÓ SÀN, không phải chọn một trong hai.
        ws = [x for x in ws
              if a.gdf_min <= ps._gdf.get(x, 0) <= gdf_max][:a.cau_toi_da]  # noqa: SLF001
        for x in ws:
            s = bi.setdefault(x, set())
            for y in ws:
                if y != x:
                    s.add(y)
    deg = {u: len(x) for u, x in bi.items()}
    ds = sorted(deg.values())
    tran = ds[min(len(ds) - 1, int(a.tran_q * len(ds)))]
    print(f"### {len(bi)} nguồn · bậc ra trung vị {ds[len(ds) // 2]}"
          f" · max {ds[-1]} · TRẦN trung gian = {tran} (phân vị {a.tran_q})")

    ra: dict[int, dict[int, int]] = {}
    n_canh = 0
    for i, (u, dsts) in enumerate(bi.items()):
        if i % 5000 == 0:
            print(f"    {i}/{len(bi)}  ({n_canh:,} cạnh)", flush=True)
        c: dict[int, int] = defaultdict(int)
        for x in dsts:
            x = int(x)
            if deg.get(x, 0) > tran:        # (1) trung gian là HUB ⇒ bỏ
                continue
            for w in (bi.get(x, ()) or ()):
                c[int(w)] += 1
        # (2) ≥ K trung gian phân biệt; bỏ cạnh ĐÃ CÓ ở bậc 1 và tự-vòng
        co = set(int(x) for x in dsts)
        # §141d — XẾP HẠNG bằng PMI, không phải ĐẾM THÔ. Đếm thô luôn nổi từ
        # phổ biến: lượt trước cho `chó → phương/nữ/thần`, `sắt → ta/đề/bản`,
        # vì `ta`,`bản`,`sự` đồng hiện với mọi thứ. Chia cho tần suất đích thì
        # `chó → mèo` (hiếm mà gắn chặt) mới thắng `chó → ta` (phổ biến mà rời
        # rạc). Lượt nướng #2 vô tình mạch lạc chỉ vì nó CHỈ giữ từ siêu hiếm.
        kept = [(w, n / (1.0 + ps._gdf.get(w, 0)) ** a.pmi_mu)   # noqa: SLF001
                for w, n in c.items()
                if n >= a.k and w != u and w not in co]
        if not kept:
            continue
        kept.sort(key=lambda t: -t[1])
        kept = kept[:a.tran_bac_ra]
        ra[int(u)] = {int(w): float(n) for w, n in kept}
        n_canh += len(kept)

    Path(a.ra).parent.mkdir(parents=True, exist_ok=True)
    with open(a.ra, "wb") as f:
        pickle.dump({"canh": ra, "tran": tran, "k": a.k,
                     "tran_q": a.tran_q, "tran_bac_ra": a.tran_bac_ra}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n### {len(ra):,} nguồn · {n_canh:,} cạnh bậc 2 → {a.ra}")
    print(f"### so cạnh bậc 1 ({sum(ds):,}): {n_canh / max(1, sum(ds)):.2f}×")
    print(f"[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
