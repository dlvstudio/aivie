"""VLM V1 · TỰ DỰNG ẢNH CHỤP MÀN HÌNH tiếng Việt có ĐÁP ÁN CHÍNH XÁC (không cần người gán nhãn, không cần thầy nhìn ảnh).

Nội dung lấy từ dữ liệu sẵn có của ta (email/văn phòng, bài SGK, code) + bảng/biểu đồ sinh ngẫu nhiên → dựng HTML giống
Excel / hộp thư / Word / trình soạn code / ứng dụng chat / biểu đồ → trình duyệt không giao diện chụp lại (nhiều phông,
cỡ chữ, sáng/tối, độ rộng màn hình). Vì ta biết chính xác chữ và số trong ảnh → câu hỏi–đáp kiểm được bằng máy:
  chép lại chữ · đọc ô Excel / công thức ô đang chọn / tổng cột · người gửi, tiêu đề email · dòng N của code ·
  giá trị cột biểu đồ, cột cao nhất · (câu cần hiểu: tóm tắt, soạn trả lời… → "teach": thầy Gemma viết đáp án TỪ chữ gốc)

    python scripts/gen_screens.py render --n 2000 --out data/vlm/screens          # Mac: Chrome; box: playwright
    python scripts/gen_screens.py teach  --out data/vlm/screens --llm http://127.0.0.1:8000/v1   # thầy điền câu cần hiểu
    → data/vlm/screens/img/*.png + train.jsonl ({"images": [...], "messages": [...], "src": ...})
"""
from __future__ import annotations
import argparse, html, json, os, random, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FONTS = ["Arial", "Helvetica", "Times New Roman", "Georgia", "Verdana", "Tahoma", "Roboto", "Noto Sans", "Segoe UI",
         "Calibri", "Inter"]
MONO = ["Menlo", "Consolas", "Courier New", "DejaVu Sans Mono", "Monaco"]
SIZES = [(1280, 800), (1366, 768), (1440, 900), (1536, 864), (1920, 1080), (1280, 1024), (1024, 768)]
HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương"]
DEM = ["Văn", "Thị", "Minh", "Thu", "Đức", "Ngọc", "Hữu", "Thanh", "Quang", "Thùy", "Gia", "Bảo"]
TEN = ["An", "Bình", "Châu", "Dũng", "Giang", "Hà", "Hải", "Hùng", "Lan", "Linh", "Long", "Mai", "Nam", "Nga", "Phúc",
       "Quân", "Sơn", "Tâm", "Thảo", "Trang", "Tuấn", "Vy", "Yến"]
SP = ["Áo sơ mi", "Quần jean", "Giày thể thao", "Túi xách", "Đồng hồ", "Bàn phím", "Chuột không dây", "Tai nghe",
      "Màn hình 24 inch", "Ổ cứng SSD", "Cà phê hạt", "Trà xanh", "Nước mắm", "Gạo ST25", "Sữa tươi", "Bánh quy"]
TINH = ["Hà Nội", "TP.HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Huế", "Nha Trang", "Vinh", "Quy Nhơn", "Đà Lạt"]
PHONG = ["Kế toán", "Kinh doanh", "Nhân sự", "Kỹ thuật", "Marketing", "Hành chính"]
THANG = [f"Tháng {i}" for i in range(1, 13)]


def cut(s, n):
    """cắt ≤ n ký tự ở cuối câu (hoặc cuối từ + "…") — không cắt giữa chữ."""
    if len(s) <= n:
        return s
    k = max(s.rfind(x, 0, n) for x in (". ", "! ", "? ", "\n"))
    return s[:k + 1].rstrip() if k > n // 2 else s[:s.rfind(" ", 0, n)].rstrip() + "…"


def ten(r):
    return f"{r.choice(HO)} {r.choice(DEM)} {r.choice(TEN)}"


def col(i):
    s = ""
    i += 1
    while i:
        i, k = divmod(i - 1, 26)
        s = chr(65 + k) + s
    return s


def vn(x):
    return f"{x:,}".replace(",", ".")


def theme(r):
    dark = r.random() < 0.3
    return dict(dark=dark, bg="#1e1e1e" if dark else "#ffffff", fg="#e6e6e6" if dark else "#1f1f1f",
                dim="#9a9a9a" if dark else "#666666", line="#3a3a3a" if dark else "#d9d9d9",
                bar="#2b2b2b" if dark else "#f3f3f3", acc=r.choice(["#217346", "#2b579a", "#c43e1c", "#0b6bcb", "#7a3ec4"]),
                font=r.choice(FONTS), fs=r.choice([12, 13, 14, 14, 15, 16, 18]))


def page(t, body, w):
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>*{{box-sizing:border-box}}body{{margin:0;background:{t['bg']};"
            f"color:{t['fg']};font:{t['fs']}px/1.45 '{t['font']}',sans-serif;width:{w}px}}"
            f"table{{border-collapse:collapse}}</style></head><body>{body}</body></html>")


def topbar(t, title, menus):
    m = "".join(f"<span style='margin-right:18px'>{x}</span>" for x in menus)
    return (f"<div style='background:{t['acc']};color:#fff;padding:6px 14px;font-size:{t['fs']}px'>{html.escape(title)}</div>"
            f"<div style='background:{t['bar']};padding:6px 14px;border-bottom:1px solid {t['line']};color:{t['dim']}'>{m}</div>")


# ── mẫu 1: Excel ──────────────────────────────────────────────────────────────────────
def m_excel(r, src):
    t = theme(r); w, _ = r.choice(SIZES)
    kind = r.choice(["ban", "luong", "kho"])
    if kind == "ban":
        hdr = ["STT", "Sản phẩm", "Khu vực", "Số lượng", "Đơn giá", "Thành tiền"]
        rows = []
        for i in range(r.randint(5, 14)):
            q, p = r.randint(1, 120), r.choice([15, 25, 49, 85, 120, 250, 399, 1200]) * 1000
            rows.append([i + 1, r.choice(SP), r.choice(TINH), q, p, q * p])
    elif kind == "luong":
        hdr = ["STT", "Họ tên", "Phòng ban", "Ngày công", "Lương cơ bản", "Thực lĩnh"]
        rows = []
        for i in range(r.randint(5, 14)):
            d, l = r.randint(18, 26), r.choice([6, 7, 8, 9, 10, 12, 15, 18, 22]) * 1_000_000
            rows.append([i + 1, ten(r), r.choice(PHONG), d, l, round(l * d / 26 / 1000) * 1000])
    else:
        hdr = ["STT", "Mặt hàng", "Tồn đầu", "Nhập", "Xuất", "Tồn cuối"]
        rows = []
        for i in range(r.randint(5, 14)):
            a, b = r.randint(0, 300), r.randint(0, 200)
            c = r.randint(0, a + b)
            rows.append([i + 1, r.choice(SP), a, b, c, a + b - c])
    nc = len(hdr)
    sel_r = r.randrange(len(rows)); sel_c = nc - 1
    L = col(sel_c)
    formula = {"ban": f"=D{sel_r+2}*E{sel_r+2}", "luong": f"=ROUND(E{sel_r+2}*D{sel_r+2}/26,-3)",
               "kho": f"=C{sel_r+2}+D{sel_r+2}-E{sel_r+2}"}[kind]
    fmt = lambda v: vn(v) if isinstance(v, int) and v >= 1000 else str(v)
    cell = f"padding:3px 8px;border:1px solid {t['line']};white-space:nowrap"
    th = "".join(f"<td style='{cell};background:{t['bar']};color:{t['dim']};text-align:center'>{col(j)}</td>" for j in range(nc))
    body = [f"<tr><td style='{cell};background:{t['bar']}'></td>{th}</tr>",
            f"<tr><td style='{cell};background:{t['bar']};color:{t['dim']}'>1</td>" +
            "".join(f"<td style='{cell};font-weight:bold'>{h}</td>" for h in hdr) + "</tr>"]
    for i, row in enumerate(rows):
        tds = ""
        for j, v in enumerate(row):
            sel = (i == sel_r and j == sel_c)
            st = cell + (";text-align:right" if isinstance(v, int) else "") + (f";outline:2px solid {t['acc']}" if sel else "")
            tds += f"<td style='{st}'>{fmt(v)}</td>"
        body.append(f"<tr><td style='{cell};background:{t['bar']};color:{t['dim']}'>{i+2}</td>{tds}</tr>")
    fbar = (f"<div style='padding:5px 14px;border-bottom:1px solid {t['line']};font-family:monospace'>"
            f"<b>{L}{sel_r+2}</b> &nbsp; <i>fx</i> &nbsp; {formula}</div>")
    title = r.choice({"ban": ["Bao_cao_ban_hang.xlsx", "Doanh thu Q" + str(r.randint(1, 4)) + ".xlsx"],
                      "luong": ["Bang_luong_T" + str(r.randint(1, 12)) + ".xlsx", "Lương nhân viên.xlsx"],
                      "kho": ["Ton_kho.xlsx", "Sổ theo dõi kho.xlsx"]}[kind])
    h = page(t, topbar(t, title + " - Excel", ["Tệp", "Trang đầu", "Chèn", "Bố trí", "Công thức", "Dữ liệu"]) + fbar +
             f"<div style='padding:10px'><table>{''.join(body)}</table></div>", w)
    # hỏi–đáp có đáp án chính xác
    qa = []
    i = r.randrange(len(rows)); j = r.randrange(1, nc)
    v = fmt(rows[i][j])
    qa.append((f"Ô {col(j)}{i+2} trong bảng có giá trị gì?",
               f"Ô {col(j)}{i+2} thuộc cột \"{hdr[j]}\", dòng STT {rows[i][0]}: ô ghi {v}. Vậy ô {col(j)}{i+2} có giá trị {v}."
               if READ[0] else f"Ô {col(j)}{i+2} có giá trị {v}."))
    qa.append(("Ô đang được chọn là ô nào và chứa công thức gì?", f"Ô đang chọn là {L}{sel_r+2}, chứa công thức {formula}."))
    k = r.choice([3, 5]) if kind != "luong" else 5
    s = sum(x[k] for x in rows)
    qa.append((f"Tính tổng cột \"{hdr[k]}\".",
               f"Các giá trị cột \"{hdr[k]}\" lần lượt là: {', '.join(fmt(x[k]) for x in rows)}. Cộng lại được {fmt(s)}."
               if READ[0] else f"Tổng cột \"{hdr[k]}\" là {fmt(s)}."))
    qa.append(("Bảng có bao nhiêu dòng dữ liệu (không tính dòng tiêu đề)?", f"Bảng có {len(rows)} dòng dữ liệu."))
    mdt = "| " + " | ".join(hdr) + " |\n|" + "---|" * nc + "\n" + "\n".join("| " + " | ".join(fmt(v) for v in x) + " |" for x in rows)
    qa.append(("Chép lại bảng trong ảnh dưới dạng bảng Markdown.", mdt))
    best = max(rows, key=lambda x: x[-1])
    qa.append((f"Dòng nào có \"{hdr[-1]}\" lớn nhất?",
               (f"Cột \"{hdr[-1]}\" theo từng dòng: " + "; ".join(f"STT {x[0]}: {fmt(x[-1])}" for x in rows) + ". ")
               * READ[0] + f"Dòng STT {best[0]} ({best[1]}) có \"{hdr[-1]}\" lớn nhất: {fmt(best[-1])}."))
    return h, w, qa, f"excel_{kind}"


# ── mẫu 2: hộp thư ────────────────────────────────────────────────────────────────────
def m_email(r, src):
    t = theme(r); w, _ = r.choice(SIZES)
    body = r.choice(src["email"])
    subj = next((l.split(":", 1)[1].strip() for l in body.splitlines() if l.lower().startswith(("tiêu đề:", "chủ đề:", "subject:"))), None)
    text = "\n".join(l for l in body.splitlines() if not l.lower().startswith(("tiêu đề:", "chủ đề:", "subject:"))).strip()
    subj = subj or r.choice(["Về việc sắp xếp lịch họp", "Thông báo nội bộ", "Xác nhận đơn hàng", "Phản hồi đề xuất"])
    frm = ten(r); dom = r.choice(["congty.vn", "abc.com.vn", "gmail.com", "truong.edu.vn"])
    addr = "".join(c for c in frm.split()[-1].lower() if c.isascii()) + "." + "".join(c for c in frm.split()[0].lower() if c.isascii())
    date = f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/2026 {r.randint(7, 18):02d}:{r.choice(['05','15','30','45'])}"
    inbox = "".join(f"<div style='padding:8px 12px;border-bottom:1px solid {t['line']};color:{t['dim']}'><b style='color:{t['fg']}'>"
                    f"{ten(r)}</b><br>{r.choice(['Báo giá tháng này', 'Lịch họp tuần sau', 'Hoá đơn VAT', 'Tài liệu đính kèm', 'Nhắc việc'])}</div>"
                    for _ in range(r.randint(4, 9)))
    para = "".join(f"<p style='margin:0 0 10px'>{html.escape(p)}</p>" for p in text.split("\n") if p.strip())
    main = (f"<div style='padding:18px 24px'><div style='font-size:1.4em;font-weight:600;margin-bottom:8px'>{html.escape(subj)}</div>"
            f"<div style='color:{t['dim']};margin-bottom:16px'><b style='color:{t['fg']}'>{frm}</b> &lt;{addr}@{dom}&gt; · {date}</div>{para}</div>")
    h = page(t, topbar(t, "Hộp thư đến", ["Thư mới", "Trả lời", "Chuyển tiếp", "Lưu trữ", "Xoá"]) +
             f"<div style='display:flex'><div style='width:260px;border-right:1px solid {t['line']}'>{inbox}</div><div style='flex:1'>{main}</div></div>", w)
    qa = [("Ai là người gửi email đang mở?", f"Người gửi là {frm} (địa chỉ {addr}@{dom})."),
          ("Tiêu đề của email là gì?", f"Tiêu đề email: \"{subj}\"."),
          ("Email được gửi lúc nào?", f"Email được gửi lúc {date}."),
          ("Chép lại nội dung email đang mở.", text),
          ("TEACH:Tóm tắt email này trong 2–3 câu.", text),
          ("TEACH:Giúp tôi soạn email trả lời ngắn gọn, lịch sự cho email này.", text)]
    return h, w, qa, "email"


# ── mẫu 3: văn bản Word ───────────────────────────────────────────────────────────────
def m_doc(r, src):
    t = theme(r); t["dark"] = False; t["bg"], t["fg"] = "#e9e9e9", "#1f1f1f"
    w, _ = r.choice(SIZES)
    title, text = r.choice(src["doc"])
    paper = min(860, w - 120)
    para = "".join(f"<p style='margin:0 0 10px;text-align:justify'>{html.escape(p)}</p>" for p in text.split("\n") if p.strip())
    h = page(t, topbar(t, title[:60] + ".docx - Word", ["Tệp", "Trang đầu", "Chèn", "Thiết kế", "Bố trí", "Tham chiếu", "Xem lại"]) +
             f"<div style='padding:24px'><div style='background:#fff;width:{paper}px;margin:0 auto;padding:48px 56px;box-shadow:0 1px 6px #0003'>"
             f"<h2 style='margin:0 0 16px;font-size:1.5em'>{html.escape(title)}</h2>{para}</div></div>", w)
    qa = [("Tiêu đề văn bản trong ảnh là gì?", title),
          ("Chép lại toàn bộ nội dung văn bản trong ảnh.", title + "\n\n" + text),
          ("TEACH:Nêu ý chính của văn bản này.", title + "\n\n" + text),
          ("TEACH:Đặt 2 câu hỏi ôn tập dựa trên văn bản này và trả lời.", title + "\n\n" + text)]
    return h, w, qa, "doc"


# ── mẫu 4: trình soạn code (+ code cài lỗi · terminal báo lỗi) ────────────────────────────
MUTS = [(r" \+ ", " - ", "dấu + thành -"), (r" < ", " <= ", "< thành <="), (r" <= ", " < ", "<= thành <"),
        (r" > ", " >= ", "> thành >="), (r" == ", " != ", "== thành !="), (r"range\((\w+)\)", r"range(\1 - 1)", "range(n) thành range(n - 1)"),
        (r"return (?!None\b|True\b|False\b)(\w+)$", r"return \1 + 1", "giá trị trả về bị cộng thêm 1"), (r"\[0\]", "[1]", "chỉ số [0] thành [1]"),
        (r" \* ", " + ", "dấu * thành +"), (r"\+= 1", "-= 1", "+= 1 thành -= 1")]


def mutate(code, r):
    """cài MỘT lỗi logic vào 1 dòng (chạy được nhưng sai kết quả) → (code lỗi, số dòng, dòng gốc, dòng lỗi, mô tả) | None."""
    lines = code.split("\n")
    cands = [(i, pat, rep, d) for i, l in enumerate(lines) if not l.strip().startswith(("#", "def ", "assert", "print"))
             for pat, rep, d in MUTS if re.search(pat, l)]
    if not cands:
        return None
    i, pat, rep, d = r.choice(cands)
    bad = re.sub(pat, rep, lines[i], count=1)
    return "\n".join(lines[:i] + [bad] + lines[i + 1:]), i + 1, lines[i].strip(), bad.strip(), d


def editor_html(code, r, dark, w, fname, term=None):
    bg, fg, dim = ("#1e1e1e", "#d4d4d4", "#858585") if dark else ("#ffffff", "#1f1f1f", "#999999")
    mono = r.choice(MONO); fs = r.choice([12, 13, 14, 15])
    rows = "".join(f"<tr><td style='color:{dim};text-align:right;padding-right:16px;user-select:none'>{i+1}</td>"
                   f"<td style='white-space:pre'>{html.escape(l) or ' '}</td></tr>" for i, l in enumerate(code.split("\n")))
    hb = (f"<div style='background:{'#323233' if dark else '#dddddd'};color:{dim};padding:6px 14px;font:13px sans-serif'>{fname} — Visual Studio Code</div>"
          f"<div style='background:{bg};color:{fg};padding:12px 8px;font:{fs}px/1.5 \"{mono}\",monospace'><table>{rows}</table></div>")
    if term:                                           # bảng TERMINAL dưới trình soạn (kiểu VS Code)
        tb = "#181818" if dark else "#f3f3f3"
        hb += (f"<div style='background:{tb};border-top:1px solid {dim};color:{dim};padding:4px 14px;font:12px sans-serif'>"
               f"TERMINAL</div><div style='background:{tb};color:{fg};padding:8px 14px;font:{fs}px/1.45 \"{mono}\",monospace;"
               f"white-space:pre'>{html.escape(term)}</div>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>body{{margin:0;background:{bg};width:{w}px}}</style></head>"
            f"<body>{hb}</body></html>")


def m_code(r, src):
    dark = r.random() < 0.7
    w, _ = r.choice(SIZES)
    code = r.choice(src["code"])
    fname = r.choice(["main.py", "xu_ly.py", "bai_tap.py", "utils.py", "app.py"])
    mut = mutate(code, r) if r.random() < 0.35 else None
    shown = mut[0] if mut else code
    lines = shown.split("\n")
    h = editor_html(shown, r, dark, w, fname)
    n = r.randrange(len(lines))
    while not lines[n].strip():
        n = r.randrange(len(lines))
    fns = re.findall(r"^\s*def (\w+)", shown, re.M)
    n_code = sum(1 for l in lines if l.strip())
    qa = [(f"Dòng {n+1} của đoạn code viết gì?", lines[n].strip()),
          ("Chép lại đoạn code trong ảnh.", "```python\n" + shown + "\n```"),
          ("TEACH:Giải thích đoạn code này làm gì.", shown),
          (f"Đoạn code trong ảnh có bao nhiêu dòng (không tính dòng trống)?", f"Đoạn code có {n_code} dòng (không tính dòng trống).")]
    if fns:
        qa.append(("Hàm đầu tiên trong đoạn code tên là gì?", f"Hàm đầu tiên tên là `{fns[0]}`."))
        if len(fns) > 1:
            qa.append(("Trong ảnh có những hàm nào?", "Có " + str(len(fns)) + " hàm: " + ", ".join(f"`{x}`" for x in fns) + "."))
    if mut:                                            # thầy viết đáp án TỪ mô tả lỗi chính xác (không đoán)
        ref = (f"(Ghi chú cho người chấm: code có MỘT lỗi logic ở dòng {mut[1]}: `{mut[3]}` — đúng phải là `{mut[2]}` "
               f"({mut[4]}).)\n{shown}")
        qa.append(("TEACH:Đoạn code này có lỗi không? Nếu có, chỉ ra dòng nào và sửa giúp.", ref))
    else:
        qa.append(("TEACH:Đoạn code này có lỗi không? Nếu có, chỉ ra dòng nào và sửa giúp.",
                   "(Ghi chú cho người chấm: code ĐÚNG, không có lỗi logic.)\n" + shown))
    return h, w, qa, "code_bug" if mut else "code", {"ref": shown, "bug": list(mut[1:]) if mut else None}


def pick_err(line, r, before=""):
    """lỗi HỢP với nội dung dòng (chia → ZeroDivisionError, lấy phần tử → IndexError/KeyError, cộng → TypeError…)."""
    import builtins, keyword
    raw = re.sub(r"#.*", "", line)                                           # bỏ chú thích
    code_ = re.sub(r"[fbr]?(\"[^\"]*\"|'[^']*')", "", raw)               # bỏ chuỗi ký tự
    if re.match(r"\s*for .+ in ", code_):
        rhs = code_.split(" in ", 1)[1]                                      # biến vòng lặp không phải "chưa định nghĩa"
    elif re.match(r"\s*[\w\[\]., \"']+\s*[-+*/]?=[^=]", code_):
        rhs = code_.split("=", 1)[1]                                         # tên ở VẾ PHẢI
    else:
        rhs = code_
    names = [x for x in re.findall(r"(?<![.\w])([A-Za-z_]\w*)\b(?!\s*=[^=])", rhs)
             if not keyword.iskeyword(x) and x not in dir(builtins)]
    loopv = set(re.findall(r"for\s+([\w, ]+?)\s+in\b", code_))
    loopv = {v.strip() for g in loopv for v in g.split(",")}
    names = [x for x in names if x not in loopv and not re.search(rf"\b{x}\b", before)]   # chưa từng xuất hiện → NameError hợp lý
    opts = [("NameError", f"name '{r.choice(names)}' is not defined")] if names else []
    if re.search(r"(/|//|%)\s*(?![0-9.]+\b)[A-Za-z_(]", rhs):                # chia cho BIẾN (chia hằng số thì không thể 0)
        opts += [("ZeroDivisionError", "division by zero")] * 3
    raw_rhs = raw.split(" in ", 1)[1] if re.match(r"\s*for .+ in ", raw) else (
        raw.split("=", 1)[1] if re.match(r"\s*[\w\[\]., \"']+\s*[-+*/]?=[^=]", raw) else raw)
    k = re.search(r"\w\[([^\]]+)\]", raw_rhs)                               # đọc phần tử (không tính gán / list rỗng)
    if k:
        key = k.group(1).strip()
        if key[0] in "'\"":
            opts += [("KeyError", key)] * 2                                  # khoá chuỗi → chỉ có thể KeyError
        else:
            opts += [("IndexError", "list index out of range")] * 2
    if re.search(r"\+|\bsum\(", rhs):
        opts += [("TypeError", "unsupported operand type(s) for +: 'int' and 'str'")] * 2
    if "int(" in rhs or "float(" in rhs:
        opts += [("ValueError", "invalid literal for int() with base 10: 'abc'")] * 3
    m_ = re.search(r"(\w+)\.(\w+)\(", line)
    if m_ and not re.search(rf"import\s+{m_.group(1)}\b|from\s+\S+\s+import\s+.*\b{m_.group(1)}\b", before):
        opts += [("AttributeError", f"'NoneType' object has no attribute '{m_.group(2)}'")] * 2
    return r.choice(opts) if opts else None


ERRS = [("NameError", "name '{v}' is not defined"), ("TypeError", "unsupported operand type(s) for +: 'int' and 'str'"),
        ("IndexError", "list index out of range"), ("ZeroDivisionError", "division by zero"),
        ("KeyError", "'{v}'"), ("AttributeError", "'NoneType' object has no attribute '{v}'"),
        ("ValueError", "invalid literal for int() with base 10: '{v}'")]


def m_term(r, src):
    """trình soạn + TERMINAL: chạy file → traceback trỏ ĐÚNG dòng trong code đang hiện (đáp án đọc được từ ảnh)."""
    dark = r.random() < 0.75
    w, _ = r.choice(SIZES)
    code = r.choice(src["code"])
    lines = code.split("\n")
    fname = r.choice(["main.py", "xu_ly.py", "bai_tap.py", "tinh_toan.py", "app.py"])
    skip = ("def ", "#", "class ", "else", "elif", "try", "except", "finally", "import ", "from ", "@", "print(")
    doc, ind = set(), False
    for k_, l in enumerate(lines):                                       # dòng nằm trong docstring """…""" → không chọn
        q = l.count('"""') + l.count("'''")
        if ind or q:
            doc.add(k_)
        if q % 2:
            ind = not ind
    body = [(i, e) for i, l in enumerate(lines) if l.strip() and i not in doc and not l.strip().startswith(skip)
            for e in [pick_err(l, r, "\n".join(lines[:i]))] if e]
    if not body:
        return m_code(r, src)
    i, (et, em) = r.choice(body)
    fn = None
    for k in range(i, -1, -1):
        m_ = re.match(r"\s*def (\w+)", lines[k])
        if m_:
            fn = m_.group(1); break
    user = r.choice(["user", "hoc_sinh", "dev", "lan", "minh"])
    path = f"/home/{user}/du_an/{fname}"
    tb = [f"$ python3 {fname}", "Traceback (most recent call last):"]
    if fn:
        call = next((k for k, l in enumerate(lines) if fn + "(" in l and not l.strip().startswith("def ")), None)
        if call is not None:
            tb += [f'  File "{path}", line {call+1}, in <module>', f"    {lines[call].strip()}"]
        tb += [f'  File "{path}", line {i+1}, in {fn}', f"    {lines[i].strip()}"]
    else:
        tb += [f'  File "{path}", line {i+1}, in <module>', f"    {lines[i].strip()}"]
    tb.append(f"{et}: {em}")
    term = "\n".join(tb)
    h = editor_html(code, r, dark, w, fname, term)
    where = f"dòng {i+1}" + (f", trong hàm `{fn}`" if fn else "")
    qa = [("Chương trình báo lỗi gì?", f"Chương trình báo lỗi `{et}: {em}`."),
          ("Lỗi xảy ra ở dòng nào?", f"Lỗi xảy ra ở {where}: `{lines[i].strip()}`."),
          ("Chép lại nội dung trong terminal.", "```\n" + term + "\n```"),
          ("TEACH:Giải thích lỗi này và cách sửa.", f"(Ghi chú cho người chấm: traceback như trong ảnh, lỗi {et} ở {where}.)\n"
                                                    f"Code:\n{code}\n\nTerminal:\n{term}")]
    return h, w, qa, "term", {"ref": code + "\n\n[TERMINAL]\n" + term}


# ── mẫu 5: biểu đồ cột (SVG) ──────────────────────────────────────────────────────────
def m_chart(r, src):
    t = theme(r); w, _ = r.choice(SIZES)
    labs = r.choice([THANG[:r.randint(4, 12)], r.sample(TINH, r.randint(4, 8)), r.sample(PHONG, r.randint(3, 6))])
    name, unit = r.choice([("Doanh thu", "triệu đồng"), ("Số đơn hàng", "đơn hàng"), ("Lượt khách", "lượt khách"),
                           ("Sản lượng", "sản phẩm"), ("Chi phí", "triệu đồng")])
    vals = [r.randint(5, 95) * r.choice([1, 10]) for _ in labs]
    title = name + " " + r.choice(["năm 2026", "quý III", "theo khu vực", "6 tháng đầu năm"])
    W, H, pad = min(w - 80, 1100), 460, 60
    mx = max(vals) * 1.15; bw = (W - 2 * pad) / len(vals)
    bars = ""
    for i, (l, v) in enumerate(zip(labs, vals)):
        x = pad + i * bw + bw * 0.15; bh = (H - 2 * pad) * v / mx; y = H - pad - bh
        bars += (f"<rect x='{x:.0f}' y='{y:.0f}' width='{bw*0.7:.0f}' height='{bh:.0f}' fill='{t['acc']}'/>"
                 f"<text x='{x+bw*0.35:.0f}' y='{y-6:.0f}' text-anchor='middle' fill='{t['fg']}'>{v}</text>"
                 f"<text x='{x+bw*0.35:.0f}' y='{H-pad+20:.0f}' text-anchor='middle' fill='{t['dim']}'>{html.escape(l)}</text>")
    svg = (f"<svg width='{W}' height='{H}' style='font:{t['fs']}px \"{t['font']}\",sans-serif'>"
           f"<line x1='{pad}' y1='{H-pad}' x2='{W-pad}' y2='{H-pad}' stroke='{t['line']}'/>{bars}</svg>")
    h = page(t, f"<div style='padding:28px 40px'><h2 style='margin:0 0 4px'>{html.escape(title)}</h2>"
                f"<div style='color:{t['dim']};margin-bottom:12px'>Đơn vị: {unit}</div>{svg}</div>", w)
    i = r.randrange(len(labs)); k = max(range(len(vals)), key=lambda j: vals[j])
    rd = ", ".join(f"{l} {v}" for l, v in zip(labs, vals))
    qa = [(f"Giá trị của \"{labs[i]}\" trên biểu đồ là bao nhiêu?",
           (f"Trên cột \"{labs[i]}\" có ghi số {vals[i]}. Vậy " if READ[0] else "") + f"\"{labs[i]}\" có giá trị {vals[i]} {unit}."),
          ("Cột nào cao nhất và bằng bao nhiêu?",
           (f"Các cột: {rd}. " if READ[0] else "") + f"Cột cao nhất là \"{labs[k]}\" với {vals[k]} {unit}."),
          ("Tổng các giá trị trên biểu đồ là bao nhiêu?",
           (" + ".join(str(v) for v in vals) + f" = {sum(vals)}. " if READ[0] else "") + f"Tổng các giá trị là {sum(vals)} {unit}."),
          ("Biểu đồ này nói về điều gì?", f"Biểu đồ cột thể hiện {title.lower()} (đơn vị: {unit}) cho {len(labs)} mục: "
                                          + ", ".join(f"{l} {v}" for l, v in zip(labs, vals)) + ".")]
    return h, w, qa, "chart"


# ── mẫu 6: ứng dụng chat ──────────────────────────────────────────────────────────────
def m_chat(r, src):
    t = theme(r); w, _ = r.choice(SIZES)
    conv = r.choice(src["chat"])
    me, other = "Tôi", ten(r)
    bub = ""
    lines = []
    for k, m in enumerate(conv):
        mine = m["role"] == "user"
        who = me if mine else other
        txt = cut(m["content"], 400)
        lines.append(f"{who}: {txt}")
        bub += (f"<div style='display:flex;justify-content:{'flex-end' if mine else 'flex-start'};margin:8px 0'>"
                f"<div style='max-width:62%;padding:8px 12px;border-radius:14px;background:{t['acc'] if mine else t['bar']};"
                f"color:{'#fff' if mine else t['fg']};white-space:pre-wrap'>{html.escape(txt)}</div></div>")
    h = page(t, topbar(t, other, ["Gọi", "Video", "Tìm kiếm", "Thông tin"]) + f"<div style='padding:12px 24px'>{bub}</div>", w)
    qa = [("Người đang trò chuyện với tôi tên là gì?", f"Người đó tên là {other}."),
          ("Chép lại đoạn hội thoại trong ảnh.", "\n".join(lines)),
          ("TEACH:Tin nhắn cuối cùng trong ảnh nói gì? Gợi ý cho tôi một câu trả lời phù hợp.", "\n".join(lines))]
    return h, w, qa, "chat"


READ = [False]          # V5 --read-first: đáp án ĐỌC trước (giá trị từng ô/cột) rồi mới kết luận
MAKERS = {"excel": (m_excel, 2), "email": (m_email, 2), "doc": (m_doc, 2), "code": (m_code, 2), "chart": (m_chart, 1.5),
          "chat": (m_chat, 1), "term": (m_term, 1)}


def load_src(rng):
    def rows(p):
        return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []
    off = rows("data/office/train.jsonl") + rows("data/office_v2/train.jsonl")
    email = [x["messages"][-1]["content"] for x in off
             if "email" in x["messages"][0]["content"].lower() and 200 < len(x["messages"][-1]["content"]) < 1800
             and "```" not in x["messages"][-1]["content"]]
    doc = []
    for x in rows("data/sgk/train.jsonl"):
        q, a_ = x["messages"][0]["content"], x["messages"][-1]["content"]
        if 400 < len(a_) < 2200:
            m = re.search(r"bài (.+?)(?: trong| môn| \(|\?|$)", q)
            tt = (m.group(1) if m else q[:60]).strip().strip('"“”\'').rstrip("?").strip()
            doc.append((tt[:1].upper() + tt[1:], a_.replace("**", "")))
    code = []
    for x in rows("data/code_tb/train.jsonl"):
        for blk in re.findall(r"```python\n(.*?)```", x["messages"][-1]["content"], re.S):
            n = blk.count("\n")
            if 4 <= n <= 40 and max(len(l) for l in blk.split("\n")) < 100:
                code.append(blk.rstrip())
    chat = [x["messages"][:r_] for x in rows("data/multiturn/train.jsonl") for r_ in [min(len(x["messages"]), 6)]
            if len(x["messages"]) >= 3]
    print(f"[src] email {len(email)} · văn bản {len(doc)} · code {len(code)} · chat {len(chat)}", flush=True)
    return {"email": email, "doc": doc, "code": code, "chat": chat}


class Shooter:
    """playwright (box, nhanh: giữ 1 trình duyệt) → nếu không có thì Chrome không giao diện qua dòng lệnh (Mac)."""
    def __init__(self):
        try:
            from playwright.sync_api import sync_playwright
            self.pw = sync_playwright().start()
            self.b = self.pw.chromium.launch()
            self.page = self.b.new_page()
            self.mode = "playwright"
        except Exception:
            self.mode = "chrome"
            self.chrome = next(p for p in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                                           "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]
                               if os.path.exists(p))

    def shot(self, h, w, out, max_h=1400):
        self._shot(h, w, out, max_h)
        from PIL import Image, ImageChops          # cắt khoảng trống thừa phía dưới (giữ lề 24px)
        im = Image.open(out).convert("RGB")
        bg = Image.new("RGB", im.size, im.getpixel((im.width - 1, im.height - 1)))
        box = ImageChops.difference(im, bg).getbbox()
        if box and box[3] + 24 < im.height:
            im.crop((0, 0, im.width, box[3] + 24)).save(out)

    def _shot(self, h, w, out, max_h):
        if self.mode == "playwright":
            self.page.set_viewport_size({"width": w, "height": 600})
            self.page.set_content(h)
            hh = min(max_h, self.page.evaluate("document.documentElement.scrollHeight"))
            self.page.set_viewport_size({"width": w, "height": max(300, hh)})
            self.page.screenshot(path=out)
            return
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(h)
        subprocess.run([self.chrome, "--headless=new", "--hide-scrollbars", f"--window-size={w},{900}",
                        f"--screenshot={out}", "file://" + f.name], capture_output=True, timeout=60)
        os.unlink(f.name)


def cmd_render(a):
    rng = random.Random(a.seed)
    src = load_src(rng)
    os.makedirs(f"{a.out}/img", exist_ok=True)
    names = [k for k in MAKERS if (k not in ("email", "doc", "code", "chat", "term") or src[k if k != "term" else "code"])]
    w_ = dict((k, float(v)) for k, v in (x.split(":") for x in a.weights.split(","))) if a.weights else {}
    wts = [w_.get(k, MAKERS[k][1]) for k in names]
    READ[0] = a.read_first
    sh = Shooter()
    print(f"[render] {sh.mode} · mẫu {names}", flush=True)
    n_ok = 0
    with open(f"{a.out}/{a.split}.jsonl", "a", encoding="utf-8") as f:
        for i in range(a.n):
            kind = rng.choices(names, wts)[0]
            h, w, qa, tag, *meta = MAKERS[kind][0](rng, src)
            meta = meta[0] if meta else {}
            name = f"{a.split}_{a.seed}_{i:06d}.png"
            try:
                sh.shot(h, w, f"{a.out}/img/{name}")
            except Exception as e:
                print(f"[render] lỗi {name}: {e}", flush=True); continue
            if kind in ("code", "term") and rng.random() < a.photo_frac:     # chụp màn hình bằng điện thoại
                from gen_news import photo_like
                from PIL import Image
                im = Image.open(f"{a.out}/img/{name}").convert("RGB")      # thêm viền trước → nghiêng/phối cảnh không cắt chữ
                pw, ph = int(im.width * 0.07), int(im.height * 0.07)
                pad = Image.new("RGB", (im.width + 2 * pw, im.height + 2 * ph), tuple(rng.randint(15, 60) for _ in range(3)))
                pad.paste(im, (pw, ph)); pad.save(f"{a.out}/img/{name}")
                photo_like(f"{a.out}/img/{name}", rng)
                os.replace(f"{a.out}/img/{name}", f"{a.out}/img/{name[:-4]}.jpg"); name = name[:-4] + ".jpg"
                tag += "_chup"
            # 1 ảnh → 1–3 câu hỏi nối thành hội thoại nhiều lượt (ảnh ở lượt đầu)
            pick = rng.sample(qa, min(len(qa), rng.choice([1, 1, 2, 3])))
            msgs, teach = [], []
            for k, (q, ans) in enumerate(pick):
                if q.startswith("TEACH:"):
                    q = q[6:]; teach.append({"turn": k, "ref": ans}); ans = ""
                msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
            f.write(json.dumps({"images": [f"img/{name}"], "messages": msgs, "src": "screen_" + tag,
                                **({"teach": teach} if teach else {}), **meta}, ensure_ascii=False) + "\n")
            n_ok += 1
            if i % 200 == 0:
                print(f"[render] {i}/{a.n}", flush=True)
    print(f"### {n_ok} ảnh → {a.out}/{a.split}.jsonl", flush=True)


P_TEACH = """Người dùng gửi cho trợ lý AI một ẢNH CHỤP MÀN HÌNH ({kind}). Nội dung chữ trong ảnh (chính xác):
<<<
{ref}
>>>
Câu hỏi của người dùng về ảnh: {q}

Viết câu trả lời của trợ lý bằng tiếng Việt tự nhiên, như thể trợ lý đang NHÌN ảnh (đừng nói "dựa vào văn bản được cung cấp").
Chỉ dùng thông tin có trong ảnh, không bịa. Gọn, đúng trọng tâm; không dùng LaTeX. Chỉ viết câu trả lời."""
KIND_VI = {"email": "hộp thư email", "doc": "văn bản Word", "code": "trình soạn code", "chat": "ứng dụng nhắn tin",
           "news": "trang báo điện tử hoặc báo in"}


def cmd_teach(a):
    """điền đáp án cho các lượt 'teach' bằng thầy (Gemma qua vLLM/Ollama) — thầy đọc CHỮ GỐC, không cần nhìn ảnh."""
    import concurrent.futures as cf
    from gen_office import ollama, CJK
    p = f"{a.out}/{a.split}.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]

    def work(r):
        kind = KIND_VI.get(r["src"].replace("screen_", "").split("_")[0], "màn hình")
        try:
            for t in r.get("teach", []):
                q = r["messages"][2 * t["turn"]]["content"]
                ans = ollama(P_TEACH.format(kind=kind, ref=t["ref"][:3500], q=q), a.llm, temp=0.5, num_predict=900).strip()
                if not ans or CJK.search(ans):
                    return None
                r["messages"][2 * t["turn"] + 1]["content"] = ans
            r.pop("teach", None)
            return r
        except Exception:
            return None
    todo = [r for r in rows if r.get("teach")]
    done = [r for r in rows if not r.get("teach")]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        out = [x for x in ex.map(work, todo) if x]
    with open(p, "w", encoding="utf-8") as f:
        for r in done + out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"### thầy viết {len(out)}/{len(todo)} mẫu cần hiểu · tổng {len(done) + len(out)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "teach"])
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/vlm/screens")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="", help="vd excel:4,chart:3,email:1 (mặc định trong MAKERS)")
    ap.add_argument("--read-first", action="store_true")
    ap.add_argument("--photo-frac", type=float, default=0.0, help="tỉ lệ ảnh code/terminal làm giống ảnh chụp màn hình")
    a = ap.parse_args()
    {"render": cmd_render, "teach": cmd_teach}[a.cmd](a)


if __name__ == "__main__":
    main()
