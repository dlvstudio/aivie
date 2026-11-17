#!/usr/bin/env python
"""P89 (TK_045 §13.65) — **LAN TRUYỀN KÍCH THÍCH + VIẾT CẠNH** (ý tưởng tác giả 2026-08-04).

Khác hẳn §13.61/§13.63: ở đó tôi tính **điểm số lúc truy vấn** từ các kết nối **đã có**. Ở đây
cạnh mới được **VIẾT VÀO đồ thị**, qua lan truyền nhiều chặng:

    khỉ ──đồng kích thích──> nghịch ngợm ──lan──> chó, mèo ──lan──> động vật
    ⇒ VIẾT cạnh mới:  khỉ ──> động vật

Ba chỗ khác thật sự: **lặp** (cạnh vòng 1 nuôi vòng 2), **cạnh là thật** (mọi cơ chế hạ nguồn
dùng được, không chỉ một bộ xếp), **thưa hoá lúc viết** (chỉ đủ mạnh mới thành cạnh).

## Vì sao bắt đầu bằng CORPUS ĐỒ CHƠI

Ví dụ 5 câu của tác giả là **test đơn vị hoàn hảo**: chân lý hiển nhiên, chạy trong một giây.
Không qua nổi đây thì không cần đo trên mini. Và §13 đã tốn nhiều vòng vì xây trước, đo sau.

## HAI TEST — dương tính VÀ âm tính

**T1 (phải suy ra):** `khỉ → động vật`, dù `con khỉ là động vật` **chưa hề có** trong corpus.

**T2 (KHÔNG được suy ra):** thêm `cái bàn màu nâu` + `con chó màu nâu`. `bàn` chung thuộc tính
`nâu` với `chó`, nhưng **không** được suy ra `bàn → động vật`. Đây là chỗ bậc hai sai loại dễ
nhất — thuộc tính chung tình cờ ≠ cùng lớp. Chặn được ở đồ chơi mới đáng đi tiếp.

## 🔴 Bản đầu TRƯỢT CẢ HAI, và chẩn đoán mới là thứ đáng giá

Bản đầu chặn trạm trung chuyển theo **tần suất** (`df/n ≥ 0,6`). Kết quả:

| | df/n | |
|---|---:|---|
| `nghịch`, `ngợm` — cầu **TỐT** trong T1 | **0,60** | **bị chặn** ⇒ T1 không suy ra được gì |
| `nâu` — cầu **XẤU** trong T2 | 0,29 | **không chặn** ⇒ T2 suy ra `bàn → động vật` |

⇒ **Tần suất KHÔNG tách được cầu tốt khỏi cầu xấu — và nó tách NGƯỢC.** Một cây cầu tốt là
thuộc tính **chung cho nhiều thành viên của lớp**, nên nó **tần suất cao** theo định nghĩa.
Chặn theo tần suất là chặn đúng cái mang thông tin.

## Khác biệt THẬT — đếm được, và không phải tần suất

    khỉ →nghịch ngợm→ {chó, mèo} → động vật    2 thực thể trung gian
    bàn →nâu→        {chó}       → động vật    1 thực thể trung gian

⇒ Điều kiện đúng là **SỐ THỰC THỂ TRUNG GIAN ĐỘC LẬP** ủng hộ suy luận (`min_support`), không
phải tần suất của cây cầu. Một thực thể chung thuộc tính = **trùng hợp**; hai trở lên = **lớp**.

Hub vẫn phải chặn, nhưng chỉ ở mức **gần phổ quát** (`con` = 5/5) — nơi token không còn phân
biệt được gì.

Chạy: `python scripts/probe_hebbian_spread.py`
"""
from __future__ import annotations

import sys
from collections import defaultdict

T1_CORPUS = [
    "con mèo là động vật",
    "con chó là động vật",
    "con mèo nghịch ngợm",
    "con chó nghịch ngợm",
    "con khỉ nghịch ngợm",
]

T2_CORPUS = T1_CORPUS + [
    "cái bàn màu nâu",
    "con chó màu nâu",
]


def build(sents: list[str]) -> tuple[dict, dict, int]:
    """A[a][b] = số câu chứa cả a và b · df[t] = số câu chứa t."""
    A: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    df: dict[str, int] = defaultdict(int)
    for s in sents:
        ws = sorted(set(s.split()))
        for w in ws:
            df[w] += 1
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                A[ws[i]][ws[j]] += 1.0
                A[ws[j]][ws[i]] += 1.0
    return A, df, len(sents)


def spread(A, df, n_sent, seed: str, *, hops: int = 3, decay: float = 0.5,
           hub_max: float = 0.95, min_support: int = 2):
    """Lan kích thích + ĐẾM SỐ THỰC THỂ TRUNG GIAN ủng hộ mỗi đích.

    `hub_max` chỉ chặn token **gần phổ quát** (`con` = 5/5) — nơi không còn phân biệt được gì.
    Thứ gánh việc phân loại là `min_support`: đích chỉ được viết thành cạnh khi có **≥
    `min_support` thực thể trung gian ĐỘC LẬP** dẫn tới nó. Một thực thể = trùng hợp; hai
    trở lên = lớp. Xem chẩn đoán ở đầu file: chặn theo tần suất tách NGƯỢC.
    """
    acc: dict[str, float] = defaultdict(float)
    # nguồn gốc: đích → tập thực thể ở chặng ÁP CHÓT dẫn tới nó
    prov: dict[str, set[str]] = defaultdict(set)
    front: dict[str, tuple[float, frozenset[str]]] = {seed: (1.0, frozenset())}
    for hop in range(hops):
        nxt: dict[str, tuple[float, set[str]]] = {}
        for a, (va, via) in front.items():
            if a != seed and df[a] / n_sent >= hub_max:
                continue                      # hub gần phổ quát: nhận nhưng không truyền
            deg = sum(A[a].values()) or 1.0
            for b, w in A[a].items():
                if b == seed:
                    continue
                add = va * (w / deg) * decay
                cur = nxt.get(b)
                # thực thể trung gian = node ở chặng ≥2 (bỏ chặng 1 vì đó là thuộc tính chung)
                nv = set(via) | ({a} if hop >= 1 else set())
                if cur is None:
                    nxt[b] = (add, nv)
                else:
                    nxt[b] = (cur[0] + add, cur[1] | nv)
        for b, (val, nv) in nxt.items():
            acc[b] += val
            prov[b] |= nv
        front = {b: (val, frozenset(nv)) for b, (val, nv) in nxt.items()}
        if not front:
            break
    return dict(acc), {k: v for k, v in prov.items()}


def report(name: str, sents: list[str], seed: str, want: list[str], forbid: list[str]) -> bool:
    A, df, n = build(sents)
    act, prov = spread(A, df, n, seed)
    direct = set(A[seed])
    MIN_SUP = 2
    new = {k: v for k, v in act.items()
           if k not in direct and k != seed and len(prov.get(k, set())) >= MIN_SUP}
    top = sorted(new.items(), key=lambda x: -x[1])
    print(f"\n=== {name} · mầm = {seed!r} ===")
    print(f"  cạnh TRỰC TIẾP đã có: {sorted(direct)}")
    print("  cạnh MỚI suy ra (giảm dần):")
    for k, val in top[:6]:
        print(f"     {k:<12} {val:.4f}   qua {sorted(prov.get(k, set()))}")
    if not top:
        print("     (không có)")
    ok = True
    for w in want:
        got = w in new
        print(f"  {'✅' if got else '❌'} PHẢI suy ra {w!r}: {'có' if got else 'KHÔNG'}")
        ok &= got
    for w in forbid:
        got = w in new
        print(f"  {'❌' if got else '✅'} KHÔNG được suy ra {w!r}: {'CÓ (sai)' if got else 'không'}")
        ok &= not got
    return ok


def main() -> int:
    print("P89 — lan truyền kích thích + viết cạnh (corpus đồ chơi)")
    for s in T1_CORPUS:
        print(f"   train: {s}")

    t1 = report("T1 dương tính", T1_CORPUS, "khỉ", want=["động", "vật"], forbid=[])

    print("\n  thêm dữ liệu bẫy:")
    for s in T2_CORPUS[len(T1_CORPUS):]:
        print(f"   train: {s}")
    t2 = report("T2 âm tính", T2_CORPUS, "bàn", want=[], forbid=["động", "vật"])

    print(f"\n⇒ T1 {'ĐẠT' if t1 else 'TRƯỢT'} · T2 {'ĐẠT' if t2 else 'TRƯỢT'}"
          f"  ⇒ {'ĐI TIẾP được' if (t1 and t2) else 'DỪNG — chưa đáng đo trên mini'}")
    return 0 if (t1 and t2) else 1


if __name__ == "__main__":
    sys.exit(main())
