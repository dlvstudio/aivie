#!/usr/bin/env python3
"""TK_108 sau C6 (user chọn phương án (a), 01/11): dữ liệu SFT bản B = TOÀN BỘ chat P3 (sft_p3.jsonl, ~107M token)
+ ~75M token code / tool-use đã kiểm chạy thật (≈ 41% token) → so với bản A (chỉ chat P3) xem trộn code có làm hỏng chat/tiếng Việt không.

01/11 (user duyệt, lịch rút gọn): CẢ A lẫn B thêm office_fix (vòng thầy văn phòng 19–21/10, sau sft_p3) → thay bước expert S3.
    A = chat P3 + office_fix · B = A + code/tool.

Lấy mẫu cố định (seed) — chạy lại ra đúng file. Xáo trộn cả file: trainer lấy 2% ĐẦU làm val → val có cả chat lẫn code.

    python scripts/build_sft_c6.py --only office --out data/sft_c6/sft_a.jsonl
    python scripts/build_sft_c6.py --out data/sft_c6/sft_b.jsonl
"""
import argparse, json, os, random

OFFICE = [
    ("data/office_fix/fix.jsonl", None),      # bản thầy sửa (Excel/Word/email…); last_only → chỉ học lượt cuối
    ("data/office_fix/off.jsonl", None),
]
CODE = [   # (file, số mẫu lấy; None = lấy hết)
    ("data/c3/train.jsonl", None),            # quỹ đạo tool-use (thầy)
    ("data/c3/train_r.jsonl", 6000),          # tool-use có lượt SAI chèn sẵn (nolearn) → phục hồi
    ("data/c3/train_r2.jsonl", None),         # lặp y hệt edit sai → đổi cách
    ("data/c7/train.jsonl", None),            # đề thực tế tiếng Việt (2 lớp kiểm chạy thật)
    ("data/c2/oci_verified.jsonl", 30000),    # OpenCodeInstruct qua test sandbox
    ("data/c2/debug_v2.jsonl", 6000),         # viết → chạy → sửa
    ("data/c2b/mix_verified.jsonl", None),    # 7 ngôn ngữ + T-SQL + HTML đã chạy
    ("data/c2b/mix_fix.jsonl", None),         # sai → lỗi → bản sửa
    ("data/c2b/mix_A.jsonl", 8000),
    ("data/code_tb/train.jsonl", 4000),       # thầy giảng thuật toán bằng tiếng Việt
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", default="data/sft_p3/sft_p3.jsonl")
    ap.add_argument("--out", default="data/sft_c6/sft_b.jsonl")
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--only", choices=["office", "all"], default="all")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [l for l in open(a.chat, encoding="utf-8") if l.strip()]
    stat = {"chat": (len(rows), sum(len(l) for l in rows))}
    for f, k in OFFICE + (CODE if a.only == "all" else []):
        ls = [l for l in open(f, encoding="utf-8") if l.strip()]
        if k is not None and k < len(ls):
            ls = rng.sample(ls, k)
        out = []
        for l in ls:   # trainer chỉ đọc "messages" → bỏ trường thừa (tests, ref…) cho file gọn
            r = json.loads(l)
            if r.get("last_only"):   # lượt trợ lý trước đó chỉ là ngữ cảnh
                last = max(i for i, m in enumerate(r["messages"]) if m["role"] == "assistant")
                r["messages"] = [dict(m, nolearn=True) if m["role"] == "assistant" and i != last else m
                                 for i, m in enumerate(r["messages"])]
            out.append(json.dumps({"source": os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f),
                                   "messages": r["messages"]}, ensure_ascii=False) + "\n")
        rows += out
        stat[f] = (len(out), sum(len(l) for l in out))
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        fo.writelines(rows)
    tot = sum(c for _, c in stat.values())
    for k, (n, c) in stat.items():
        print(f"{k:34s} {n:7d} mẫu · {c / tot * 100:5.1f}% ký tự")
    off = sum(c for k, (_, c) in stat.items() if "office_fix" in k)
    print(f"### {os.path.basename(a.out)}: {len(rows)} mẫu · văn phòng {off / tot * 100:.1f}% · code/tool "
          f"{100 - (stat['chat'][1] + off) / tot * 100:.1f}% ký tự → {a.out}")


if __name__ == "__main__":
    main()
