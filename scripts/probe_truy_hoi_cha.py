"""§85 ĐO — `rank_matches` truy ra cha nào cho prompt định nghĩa, và vì sao?

§84 dồn nút thắt về đây: kênh trật tự tốt đúng bằng cha mà nó được cho. `lúa là`
truy đúng ⇒ sinh gần nguyên văn nguồn; `hổ là` truy ra `xương hổ dùng làm…` ở
MỌI `top_n`, MỌI trọng số ⇒ hỏng bất kể chỉnh gì phía sau.

Nghi vấn: `_content_ids` bỏ hư từ, nên `«hổ là»` rút còn ĐÚNG MỘT token nội dung
(`hổ`). Với truy vấn một token thì không còn gì để phân biệt — mọi cha chứa `hổ`
hoà nhau ở phần phủ, và thứ quyết định là các yếu tố phụ (độ dài, power, df).

Probe in ra, cho mỗi prompt vàng:
    · token nội dung còn lại sau `_content_ids`
    · 8 cha đầu bảng: điểm, power, độ dài, và VĂN BẢN
để thấy tận mắt cha đúng đứng thứ mấy, hay không có trong bảng.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ (a) Nếu prompt định nghĩa rút về ≤1 token nội dung ⇒ xác nhận truy vấn    ║
║     KHÔNG có tín hiệu phân biệt; hướng vá là giữ lại tín hiệu khuôn       ║
║     «X là …» chứ không phải chỉnh trọng số xếp hạng.                      ║
║ (b) Nếu cha ĐÚNG có trong bảng nhưng xếp sau ⇒ lỗi XẾP HẠNG.             ║
║ (c) Nếu cha ĐÚNG không có trong bảng ⇒ lỗi THU HỒI, chỉnh hạng vô ích.   ║
║ Ba nhánh dẫn tới ba bản vá khác nhau — phải biết nhánh nào TRƯỚC khi vá.  ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/probe_truy_hoi_cha.py \
        checkpoints/network_v12_sent_dn.pkl
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.parent_store import _content_ids  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
PROMPT = ["hổ là", "lúa là", "sắt là", "chó là", "sóc là"]
# mẩu chữ phải có trong cha ĐÚNG (lấy từ dinh_nghia.jsonl)
DUNG = {"hổ là": "loài động vật có lãnh thổ",
        "lúa là": "cây lương thực",
        "sắt là": "kim loại màu xám",
        "chó là": "gia súc thuộc nhóm",
        "sóc là": "thú gặm nhấm"}


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_dn.pkl"
    state, _ = load_checkpoint(ck)
    v, ps = state.vocab, state.parents
    # ⚠️ PHẢI cấu hình `ps` y hệt `generate()` (generate_hebbian.py ~4104).
    # Bản đầu của probe gọi `rank_matches` trên store TRẦN ⇒ `directed_bigram_
    # bonus` = 0 (mặc định của `getattr`), tức đo một đường KHÔNG TỒN TẠI trong
    # production, và khối hoà điểm quan sát được có thể là hiện vật của chính
    # probe. Cùng loại lỗi với §81/§83: quên truyền thứ đường thật vẫn truyền.
    cfg = HCAConfig()
    ps._vocab_ref = v
    ps.focus_pos_alpha = cfg.focus_pos_alpha
    ps.directed_bigram_bonus = cfg.directed_bigram_bonus
    ps.directed_bigram_min_idf = cfg.directed_bigram_min_idf
    ps.directed_bigram_dmax = int(cfg.directed_bigram_dmax)
    ps.directed_bigram_lam = float(cfg.directed_bigram_lam)
    ps.directed_bigram_accum = str(cfg.directed_bigram_accum)
    print(f"\n### {ck} · {len(ps.parents)} cha")
    print(f"### dir_bonus={ps.directed_bigram_bonus} dmax={ps.directed_bigram_dmax}"
          f" accum={ps.directed_bigram_accum} lam={ps.directed_bigram_lam}\n")

    for p in PROMPT:
        qs = [int(t) for t in v.encode_prompt(p)]
        noi_dung = _content_ids(qs, v, skip=set())
        print(f"── «{p}»  ·  {len(qs)} token → NỘI DUNG {len(noi_dung)}: "
              f"{[v.detokenize([t]) for t in noi_dung]}")
        ranked = ps.rank_matches(qs, v, None, fuzzy_topk=0, top_n=8)
        moc = DUNG[p]
        thay = None
        for i, (pi, sc) in enumerate(ranked, 1):
            rec = ps.parents[int(pi)]
            txt = v.detokenize(list(rec.emit_ids))
            if moc in txt and thay is None:
                thay = i
            print(f"     {i}. điểm={sc:7.3f} power={getattr(rec,'power',0):>4}"
                  f" dài={len(rec.emit_ids):>3}  {txt[:88]}")
        # cha ĐÚNG có tồn tại trong kho không, và đứng thứ mấy trong bảng rộng?
        rong = ps.rank_matches(qs, v, None, fuzzy_topk=0, top_n=200)
        hang = None
        for i, (pi, _s) in enumerate(rong, 1):
            if moc in v.detokenize(list(ps.parents[int(pi)].emit_ids)):
                hang = i
                break
        co_trong_kho = any(
            moc in v.detokenize(list(r.emit_ids)) for r in ps.parents)
        print(f"     ⇒ cha ĐÚNG («{moc}»): trong top-8 {thay or '—'}"
              f" · trong top-200 hạng {hang or '—'}"
              f" · CÓ trong kho: {co_trong_kho}\n")

    print(f"[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
