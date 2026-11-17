"""Chuẩn hoá dữ liệu văn phòng: công thức Excel/Sheets LUÔN nằm riêng trong khối ```excel (sau vòng thầy P3.3' model hay viết
công thức lẫn trong câu → Excel 17/27 → 9/27 chấm chặt; 16 → 13 chấm lỏng).

  python scripts/build_office_fix.py data/office_v2/train.jsonl:data/office_fix/off.jsonl data/v5/sft.jsonl:data/office_fix/fix.jsonl
Mỗi cặp vào:ra; giữ nguyên mọi dòng, chỉ sửa các lượt assistant có công thức chưa nằm trong khối.
"""
import json, os, re, sys

HEAD = re.compile(r"=\s*[A-Z][A-Z0-9\.]*\(")


def formula_at(t, m):
    """công thức bắt đầu ở m.start(), kết thúc ở dấu ) cân bằng."""
    d = 0
    for i in range(m.end() - 1, len(t)):
        if t[i] == "(":
            d += 1
        elif t[i] == ")":
            d -= 1
            if d == 0:
                return t[m.start():i + 1]
    return None


def in_block(t):
    return any(ln.strip().startswith("=") for b in re.findall(r"```[a-zA-Z]*\n(.*?)```", t, re.S) for ln in b.splitlines())


def norm(t):
    # 1) ```=SUM(..)``` một dòng / ```excel =SUM(..)``` → khối nhiều dòng
    t2 = re.sub(r"```(?:excel|Excel)?[ \t]*(=[^\n`]+?)[ \t]*```", lambda m: f"\n```excel\n{m.group(1).strip()}\n```\n", t)
    if in_block(t2):
        return t2
    # 2) công thức đầu tiên viết lẫn trong câu (có thể kèm `...`) → tách thành khối
    m = HEAD.search(t2)
    if not m:
        return t2
    f = formula_at(t2, m)
    if not f:
        return t2
    s, e = m.start(), m.start() + len(f)
    if s > 0 and t2[s - 1] == "`" and e < len(t2) and t2[e] == "`":
        s, e = s - 1, e + 1
    before, after = t2[:s].rstrip(" :"), t2[e:].lstrip(" .`")
    out = f"{before}:\n\n```excel\n{f}\n```\n\n{after}" if before else f"```excel\n{f}\n```\n\n{after}"
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def main():
    for pair in sys.argv[1:]:
        src, dst = pair.split(":")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        n = ch = 0
        with open(dst, "w", encoding="utf-8") as fo:
            for l in open(src, encoding="utf-8"):
                r = json.loads(l); n += 1
                for msg in r.get("messages", []):
                    if msg["role"] == "assistant" and HEAD.search(msg["content"]):
                        new = norm(msg["content"])
                        if new != msg["content"]:
                            msg["content"] = new; ch += 1
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[office_fix] {src} → {dst} · {n} dòng · sửa {ch} lượt", flush=True)


if __name__ == "__main__":
    main()
