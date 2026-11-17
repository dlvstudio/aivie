"""§88 — dựng corpus LOẠI SUY từ tri thức THẬT của từ điển (công thức §80).

Vì sao không dùng thứ đã có:
  · `so_sanh_that.jsonl` (TK_052) KHÔNG phải corpus so sánh — 11 553 câu chỉ
    CHỨA chữ `như`, mà `như` ở đó nghĩa «chẳng hạn» (liệt kê). Train nó bơm đúng
    khuôn liệt kê mà TK_052 vòng 6 phải vá để khỏi phát cạnh NGƯỢC.
  · `loai_suy.jsonl` là TỔNG HỢP và vô nghĩa (`ngặt như loài biển`), lặp 4 lần.

Nguyên liệu (§88.1, đo trước khi dựng): rút tính từ từ chính CÂU ĐỊNH NGHĨA
(`X là <lớp>, …<tính từ>…`) cho **118 thuộc tính có ≥2 vật mang CÙNG LỚP** —
đúng điều kiện ② của cơ chế, và là tri thức THẬT chứ không bịa:

    lớn  → chim ×5: cú, hạc, sếu, trả, ưng
    cứng → phần ×6: lõi, móng, mỏ, răng, sừng, yếm

LỌC ĐỘ ĐẶC TRƯNG: thuộc tính mang bởi 129 thứ (`dài`) không phân biệt được gì.
Giữ khoảng 2…`--max-mang` vật mang. Đây là lọc theo TẦN SUẤT ĐO ĐƯỢC, không phải
danh sách từ viết tay (Luật 0).

Mỗi bộ phát ra:

    <A> như <m1> .      ← cạnh thuộc tính, khuôn `như` mà §53 `via_attr` đọc
    <A> như <m2> .
    (câu `<mi> là <C>` KHÔNG phát lại — đã có sẵn trong `dinh_nghia.jsonl`)

và ghi khoá (A, C, m3) trong đó m3 là thành viên CÙNG LỚP chưa từng đứng cạnh A
— đó là «con khỉ» của bộ đó, dùng để chấm thu hồi.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/dung_corpus_loai_suy_that.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_aivie.hca.hub_spread import load_pos_map  # noqa: E402

T0 = time.time()
RXD = re.compile(r"^(.+?)\s+là\s+(.+)$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nguon", default="data/text_corpus/dinh_nghia.jsonl")
    ap.add_argument("--pos", default="data/dict/vi_dict_raw.jsonl")
    ap.add_argument("--out", default="data/text_corpus/loai_suy_that.jsonl")
    ap.add_argument("--khoa", default="out/loai_suy_that_khoa.json")
    ap.add_argument("--max-mang", type=int, default=20,
                    help="bỏ thuộc tính quá phổ thông (mang bởi > N vật)")
    ap.add_argument("--tu-dien-tho", default="data/dict/vi_dict_raw.jsonl")
    ap.add_argument("--wiki", default="data/text_corpus/viwiki.jsonl")
    ap.add_argument("--dung-wiki", action="store_true")
    ap.add_argument("--min-mang", type=int, default=2,
                    help="§88.13 — số vật mang CÙNG LỚP tối thiểu để thành bộ. "
                         "2 = ràng buộc cũ (đo được: chỉ 2/50 thuộc tính thành "
                         "ngữ qua nổi, vì thành ngữ cố tình chọn vật mẫu KHÁC "
                         "lớp). 1 = chuỗi vẫn đủ mắt xích: `nhanh như cắt` + "
                         "`cắt là chim` + `chim ⊃ {cú, cò…}` ⇒ `nhanh như cú`.")
    ap.add_argument("--bac-toi-da", type=int, default=0,
                    help="0 = tự chọn từ PHÂN BỐ (trung vị×4); >0 = ép ngưỡng")
    ap.add_argument("--nhac-lai", type=int, default=2)
    a = ap.parse_args()

    pos = {k.lower(): str(v).lower()
           for k, v in load_pos_map(a.pos).items()}
    la_tt = lambda w: pos.get(w, "").startswith("tính")   # noqa: E731
    la_dt = lambda w: pos.get(w, "").startswith("danh")   # noqa: E731

    def don_vi(ws: list[str]) -> list[str]:
        """Tách mệnh đề thành ĐƠN VỊ TỪ, khớp mục từ điển dài nhất (3→2→1)."""
        ra, i = [], 0
        while i < len(ws):
            for n in (3, 2, 1):
                if i + n <= len(ws) and " ".join(ws[i:i + n]) in pos:
                    ra.append(" ".join(ws[i:i + n]))
                    i += n
                    break
            else:
                ra.append(ws[i])
                i += 1
        return ra

    def quet(ws: list[str], nhan) -> list[str]:
        """Quét mệnh đề, KHỚP MỤC TỪ ĐIỂN DÀI NHẤT trước (3→2→1 âm tiết).

        ⚠️ Bản đầu tách theo DẤU CÁCH rồi tra từng âm tiết. Tiếng Việt đa âm
        tiết nên `phức tạp`→`phức`, `trung bình`→`trung`, `mĩ miều`→`mĩ` đều lọt
        qua bộ lọc tính từ và sinh ra rác (`phức như robot`, `trung như não`).
        Đọc 14 mẫu bằng mắt mới lộ — không chỉ số nào bắt được.
        """
        ra, i = [], 0
        while i < len(ws):
            for n in (3, 2, 1):
                if i + n <= len(ws):
                    cum = " ".join(ws[i:i + n])
                    if cum in pos:
                        if nhan(cum):
                            ra.append(cum)
                        i += n
                        break
            else:
                i += 1
                continue
            if not (i and cum in pos):
                i += 1
        return ra

    mang: dict[str, set[str]] = defaultdict(set)
    lop: dict[str, str] = {}
    n_cau = 0

    def nguon_cap():
        """(chủ ngữ, vị ngữ) từ BA nguồn — §88.7 đường 1: cùng ba luật, nhiều
        nguyên liệu hơn. `dinh_nghia.jsonl` chỉ có 10 712 câu `X là …`; từ điển
        thô có 36 764 mục `word`+`meaning`, viwiki có 285 935 dòng văn xuôi."""
        for ln in Path(a.nguon).open(encoding="utf-8"):          # 1. dinh_nghia
            try:
                t = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(t, str) and (m := RXD.match(t.strip())):
                yield m.group(1), m.group(2)
        q = Path(a.tu_dien_tho)                                   # 2. từ điển thô
        if q.exists():
            for ln in q.open(encoding="utf-8"):
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict) and d.get("word") and d.get("meaning"):
                    yield str(d["word"]), str(d["meaning"])
        w = Path(a.wiki)                                          # 3. viwiki
        if w.exists() and a.dung_wiki:
            for ln in w.open(encoding="utf-8"):
                try:
                    t = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if not isinstance(t, str):
                    continue
                cau1 = t.split(" . ")[0].split(". ")[0]
                if (m := RXD.match(cau1.strip())):
                    yield m.group(1), m.group(2)

    for chu, ve in nguon_cap():
        n_cau += 1
        chu = chu.strip().lower()
        if " " in chu or not la_dt(chu):
            continue
        # LỚP = cụm danh từ ĐẦU vị ngữ, khớp dài nhất (`dụng cụ`, không `dụng`)
        dws = ve.strip().lower().split()
        c0 = next((" ".join(dws[:n]) for n in (3, 2, 1)
                   if " ".join(dws[:n]) in pos and la_dt(" ".join(dws[:n]))), None)
        if c0:
            lop.setdefault(chu, c0)
        # §88.4 THẮT — chỉ nhận mệnh đề GẮN TRỰC TIẾP vào chủ ngữ.
        #
        # Bản trước nhận MỌI tính từ trong vị ngữ ⇒ ~50% rác (`trung như tỵ`,
        # `rỗng như dê`), vì vị ngữ chứa cả tính từ mô tả thứ khác và mệnh đề
        # định nghĩa vị trí/thứ tự (`tỵ là chi thứ sáu trong mười hai chi`).
        #
        # Ba luật CẤU TRÚC, không danh sách từ viết tay (Luật 0):
        #   1. BỎ mệnh đề ĐẦU — đó là mệnh đề giống/loài (`thú gặm nhấm lớn hơn
        #      chuột`), tính từ ở đó tả GIỐNG chứ không tả cá thể.
        #   2. Nhận `<danh từ> <tính từ>` đúng 2 đơn vị — khuôn thuộc tính kinh
        #      điển của tiếng Việt: `mõm tròn`, `đuôi xù`, `da trơn`.
        #   3. Nhận mệnh đề chỉ gồm ĐÚNG MỘT tính từ — `tròn`, `xù`.
        # Mọi mệnh đề khác BỎ.
        ves = [x.strip().lower() for x in re.split(r"[,;]", ve)]
        for j, c in enumerate(ves):
            dv = don_vi(c.split())
            if len(dv) == 2 and la_dt(dv[0]) and la_tt(dv[1]):
                w = dv[1]                      # luật 2 — khuôn `mõm tròn`
            elif j >= 1 and len(dv) == 1 and la_tt(dv[0]):
                w = dv[0]                      # luật 3 — mệnh đề chỉ 1 tính từ
            else:
                continue
            if w != chu:
                mang[w].add(chu)

    # ── ĐƯỜNG B (§88.10): thành ngữ so sánh, CHỈ quét TỪ ĐIỂN.
    # Quét mọi nguồn thì `như` ở viwiki nghĩa «chẳng hạn» và cho rác (§88.2);
    # trong từ điển `như` thật sự là so sánh ⇒ `nhanh như cắt`, `đen như mực`.
    RXN = re.compile(r"(\S+(?:\s+\S+){0,2})\s+như\s+((?:\S+\s+){0,2}\S+)")
    q = Path(a.tu_dien_tho)
    n_b = 0
    if q.exists():
        for ln in q.open(encoding="utf-8"):
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict):
                continue
            tho = f"{d.get('word','')} {d.get('meaning','')} {d.get('example','')}"
            for seg in re.split(r"[;:.]", tho):
                if " như " not in seg:
                    continue
                for m in RXN.finditer(seg.lower()):
                    A_ = don_vi(m.group(1).split())
                    B_ = don_vi(m.group(2).split())
                    if A_ and B_ and la_tt(A_[-1]) and la_dt(B_[0]):
                        mang[A_[-1]].add(B_[0])
                        n_b += 1

    # ── LỌC BẬC — vật mang phục vụ QUÁ NHIỀU thuộc tính là hư từ/chung chung
    # (`cái`, `thế`, `không`, `màu`: từ điển gán nhãn danh từ nhưng chúng không
    # phải mục tiêu so sánh). Ngưỡng lấy TỪ PHÂN BỐ, không tự đặt.
    bac: Counter = Counter()
    for A_, ms_ in mang.items():
        for x in ms_:
            bac[x] += 1
    if bac:
        tv = sorted(bac.values())[len(bac) // 2]
        nguong = a.bac_toi_da or max(2, tv * 4)
        bo = {x for x, k in bac.items() if k > nguong}
        print(f"### LỌC BẬC · trung vị bậc={tv} · ngưỡng={nguong}"
              f" · bỏ {len(bo)}/{len(bac)} vật mang")
        print(f"###   bỏ nhiều nhất: "
              f"{[x for x, _ in bac.most_common(10)]}")
        for A_ in list(mang):
            mang[A_] -= bo
            if not mang[A_]:
                del mang[A_]

    dong: list[str] = []
    khoa: list[dict] = []
    for A, ms in sorted(mang.items()):
        if not (2 <= len(ms) <= a.max_mang):
            continue
        # gom theo LỚP; ngưỡng số vật mang cùng lớp = `--min-mang` (§88.13)
        theo: dict[str, list[str]] = defaultdict(list)
        for x in ms:
            theo[lop.get(x, "?")].append(x)
        for C, xs in theo.items():
            if C == "?" or len(xs) < a.min_mang:
                continue
            xs = sorted(xs)
            # thành viên CÙNG LỚP nhưng KHÔNG mang A ⇒ «con khỉ»
            khi = sorted(x for x, c in lop.items() if c == C and x not in ms)
            if not khi:
                continue
            for m1 in xs:
                for _ in range(a.nhac_lai):
                    dong.append(f"{A} như {m1} .")
            khoa.append({"thuoc_tinh": A, "lop": C,
                         "gieo": xs, "khi": khi[:20]})

    Path(a.out).write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in dong) + "\n",
        encoding="utf-8")
    Path(a.khoa).write_text(
        json.dumps(khoa, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n### đường A {n_cau} mục · đường B {n_b} cặp thành ngữ"
          f" · {len(mang)} thuộc tính sau lọc")
    print(f"### GIỮ {len(khoa)} bộ (thuộc tính × lớp) · {len(dong)} dòng")
    print(f"### → {a.out}\n### → {a.khoa}")
    for k in khoa[:6]:
        print(f"   «{k['thuoc_tinh']}» lớp={k['lop']!r}"
              f" gieo={k['gieo'][:5]} khỉ={k['khi'][:5]}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
