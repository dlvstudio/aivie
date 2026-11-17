"""Chương-trình "TOÁN TIỂU HỌC" tổng-hợp bằng code (TK_104) — đề + lời giải từng bước, đáp-số kiểm-chứng tuyệt-đối.

Mỗi LỚP = một mức độ khó (số bước tính, cỡ số, phép toán). Sinh vô-hạn, tức-thì, không cần LLM.
Lớp 6 KHÔNG bao giờ dùng để train — chỉ để đo "nghĩ nhiều vòng hơn có giải được bài khó hơn đã học không".

  lớp 1 · 1 bước  · cộng/trừ trong phạm-vi 10 (không nhớ)
  lớp 2 · 1-2 bước · cộng/trừ trong phạm-vi 100 (có nhớ)
  lớp 3 · 2 bước  · thêm nhân/chia (bảng cửu-chương, chia hết)
  lớp 4 · 3 bước  · số đến 1000, trộn 4 phép
  lớp 5 · 4 bước  · số đến 1000, trộn 4 phép
  lớp 6 · 5-6 bước (NGOÀI CHƯƠNG-TRÌNH — chỉ kiểm-tra)

Định-dạng (một mẫu = một dòng):
  Đề: <bài toán>\\nGiải: <bước 1>. <bước 2>. … Đáp số: <N>.

    python curriculum/gen_toan.py --per-level 40000 --test 500 --out data/curriculum/toan
"""
from __future__ import annotations
import argparse, json, os, random

TEN = ["Lan", "Minh", "An", "Hoa", "Nam", "Mai", "Tùng", "Linh", "Bình", "Hà", "Quân", "Thảo", "Phúc", "Vy", "Khoa"]
VAT = [("quả táo", "quả"), ("cái kẹo", "cái"), ("quyển vở", "quyển"), ("bông hoa", "bông"), ("viên bi", "viên"),
       ("con gà", "con"), ("chiếc bút", "chiếc"), ("quả cam", "quả"), ("cái bánh", "cái"), ("con cá", "con"),
       ("tờ giấy màu", "tờ"), ("hộp sữa", "hộp")]
NHAN = ["mỗi hộp có", "mỗi túi có", "mỗi rổ có", "mỗi đĩa có", "mỗi hàng có"]
TUI = ["hộp", "túi", "rổ", "đĩa", "hàng"]


def op_text(op, rnd, who, n, unit):
    """Câu kể cho một bước biến-đổi số lượng hiện có."""
    if op == "+":
        return rnd.choice([f"{who} được cho thêm {n} {unit}", f"{who} mua thêm {n} {unit}",
                           f"{who} nhặt thêm được {n} {unit}"])
    if op == "-":
        return rnd.choice([f"{who} cho bạn {n} {unit}", f"{who} dùng hết {n} {unit}",
                           f"{who} làm rơi mất {n} {unit}"])
    if op == "*":                                         # kết-quả = cur × n
        return rnd.choice([f"số {unit} của {who} tăng lên gấp {n} lần", f"số {unit} của {who} được nhân lên {n} lần"])
    t = rnd.choice(TUI)                                    # kết-quả = cur : n (giữ đúng MỘT trong n phần bằng nhau)
    return rnd.choice([f"{who} chia đều số {unit} vào {n} {t} rồi chỉ giữ lại một {t}",
                       f"{who} chia số {unit} thành {n} phần bằng nhau và giữ lại một phần"])


def make(level, rnd):
    """Sinh MỘT bài ở lớp `level` → (đề, lời giải, đáp-số)."""
    who = rnd.choice(TEN)
    item, unit = rnd.choice(VAT)
    if level == 1:
        a = rnd.randint(1, 9)
        if rnd.random() < 0.5:
            b = rnd.randint(1, 10 - a); ops = [("+", b)]
        else:
            b = rnd.randint(1, a); ops = [("-", b)]
    else:
        steps = {2: rnd.choice([1, 2]), 3: 2, 4: 3, 5: 4, 6: rnd.choice([5, 6]),
                 7: rnd.choice([7, 8]), 8: rnd.choice([10, 12])}[level]   # 7-8: chỉ kiểm-tra ngoại-suy SÂU
        hi = 100 if level == 2 else (60 if level == 3 else 1000)
        allowed = ["+", "-"] if level == 2 else ["+", "-", "*", "/"]
        a = rnd.randint(2, hi // 2 if level >= 4 else hi - 1)
        ops, cur = [], a
        for _ in range(steps):
            for _try in range(30):                         # chọn phép sao cho kết-quả hợp-lệ (nguyên, dương, ≤ trần)
                op = rnd.choice(allowed)
                if op == "+":
                    b = rnd.randint(1, max(1, hi - cur)) if cur < hi else None
                elif op == "-":
                    b = rnd.randint(1, cur - 1) if cur > 1 else None
                elif op == "*":
                    b = rnd.randint(2, 9) if cur * 2 <= hi else None
                    if b is not None and cur * b > hi:
                        b = None
                else:
                    divs = [d for d in range(2, 10) if cur % d == 0]
                    b = rnd.choice(divs) if divs else None
                if b is not None:
                    break
            else:
                op, b = "+", 1
            ops.append((op, b))
            cur = {"+": cur + b, "-": cur - b, "*": cur * b, "/": cur // b}[op]
    # đề
    q = f"{who} có {a} {item}."
    for i, (op, b) in enumerate(ops):
        q += (" Rồi " if i == 0 else " Sau đó ") + op_text(op, rnd, who, b, unit) + "."
    q += f" Hỏi cuối cùng {who} có bao nhiêu {unit}?"
    # lời giải từng bước
    cur, sol = a, []
    for op, b in ops:
        nxt = {"+": cur + b, "-": cur - b, "*": cur * b, "/": cur // b}[op]
        sym = {"+": "+", "-": "-", "*": "×", "/": ":"}[op]
        sol.append(f"{cur} {sym} {b} = {nxt}")
        cur = nxt
    return q, ". ".join(sol) + f". Đáp số: {cur}.", cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=40000)
    ap.add_argument("--test", type=int, default=500, help="số bài kiểm-tra giữ riêng mỗi lớp")
    ap.add_argument("--out", default="data/curriculum/toan")
    ap.add_argument("--seed", type=int, default=104)
    ap.add_argument("--deep-only", action="store_true", help="chỉ sinh đề kiểm-tra lớp 7-8 (không đụng lớp 1-6)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(a.seed)
    seen = set()
    if a.deep_only:                                         # không trùng mọi đề đã có
        for fn in os.listdir(a.out):
            seen |= {json.loads(l)["q"] for l in open(f"{a.out}/{fn}", encoding="utf-8")} if fn.endswith(".jsonl") else set()
        rnd = random.Random(a.seed + 7)
    for level in (range(7, 9) if a.deep_only else range(1, 7)):
        n_train = 0 if level >= 6 else a.per_level
        rows = []
        while len(rows) < n_train + a.test:
            q, s, ans = make(level, rnd)
            if q in seen:                                   # bài kiểm-tra KHÔNG trùng bài train
                continue
            seen.add(q)
            rows.append({"level": level, "q": q, "sol": s, "ans": ans})
        for split, part in (("test", rows[:a.test]), ("train", rows[a.test:])):
            if part:
                with open(f"{a.out}/lop{level}_{split}.jsonl", "w", encoding="utf-8") as f:
                    for r in part:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"lớp {level}: train {n_train} · test {a.test}", flush=True)


if __name__ == "__main__":
    main()
