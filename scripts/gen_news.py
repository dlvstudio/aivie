"""VLM V5 · ẢNH TRANG BÁO (web tin tức + báo in) có ĐÁP ÁN CHÍNH XÁC — dạy model đọc báo tiếng Việt.

Chữ bài báo: FineWeb-2 tiếng Việt (văn bản web thật, nhiều tin tức; lọc kiểu tin). Tên báo là tên GIẢ (không mạo danh báo thật).
  article  trang bài báo điện tử: tên báo, chuyên mục, tiêu đề, tác giả/giờ đăng, sapo, ảnh + chú thích, thân bài, "Đọc nhiều"
  home     trang chủ báo điện tử: lưới tin (ảnh nhỏ + tiêu đề + chuyên mục/giờ)
  paper    trang báo in: măng-sét, số báo/ngày/giá, tiêu đề lớn, chữ chia 3–5 cột, ảnh, "Tin vắn"; ~50% làm giống ẢNH CHỤP
           (nghiêng, phối cảnh, giấy ngả màu, mờ, bóng, nén JPEG)
Trình duyệt ĐO vị trí từng đoạn → chỉ hỏi những đoạn HIỆN TRỌN trong ảnh (đáp án đúng tuyệt đối).
Câu cần hiểu (tóm tắt, ý chính) → "teach": thầy viết từ chữ gốc (gen_screens.py teach).

    python scripts/gen_news.py --n 3000 --out data/vlm/news --seed 1 [--fineweb data/fineweb2hq/vie_Latn/000_00000.parquet]
"""
from __future__ import annotations
import argparse, html, json, os, random, re, sys

SITES = ["Tin Nhanh 24h", "Sao Việt Online", "Thời Báo Phương Nam", "Nhịp Sống Số", "Tin Tức Việt Nay", "Báo Mới Mỗi Ngày",
         "Đời Sống Xanh", "Người Việt Hôm Nay", "Tiếng Nói Thủ Đô", "Bản Tin Miền Trung", "Kinh Tế & Đời Sống", "Phố Thị News"]
PAPERS = ["THỜI BÁO PHƯƠNG NAM", "TIẾNG NÓI THỦ ĐÔ", "NHÂN DÂN MIỀN TRUNG", "ĐỜI SỐNG VIỆT", "BÁO SÁNG NAY", "TIN TỨC HẰNG NGÀY",
          "LAO ĐỘNG & XÃ HỘI", "GIA ĐÌNH VIỆT"]
CATS = ["Thời sự", "Thế giới", "Kinh doanh", "Thể thao", "Giải trí", "Sức khỏe", "Giáo dục", "Công nghệ", "Du lịch", "Pháp luật",
        "Đời sống", "Bất động sản"]
HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi", "Đỗ", "Ngô", "Dương", "Lý"]
TEN = ["Minh Anh", "Thu Hà", "Quang Huy", "Ngọc Lan", "Đức Thắng", "Thanh Tâm", "Bảo Ngọc", "Hữu Phước", "Mai Linh", "Tuấn Kiệt"]
THU = ["Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy", "Chủ nhật"]
SERIF = ["Noto Serif", "Liberation Serif", "Times New Roman", "Georgia", "DejaVu Serif"]
SANS = ["Noto Sans", "Liberation Sans", "Arial", "Roboto", "Helvetica", "DejaVu Sans"]


# ── nguồn chữ ─────────────────────────────────────────────────────────────────────────
def load_docs(path, n_max=60000):
    """văn bản kiểu tin tức (≥3 đoạn) từ FineWeb-2 tiếng Việt: file parquet cục bộ, không có thì đọc luồng từ HF."""
    pat = re.compile(r"\b(cho biết|theo|phóng viên|ngày \d{1,2}/\d{1,2}|chiều|sáng|tối) ")
    out = []

    def take(x):
        paras = [re.sub(r"\s+", " ", p).strip() for p in x.split("\n")]
        paras = [p for p in paras if len(p) >= 60 and not p.startswith(("-", "*", "+", "|", "#"))]
        if len(paras) >= 3 and pat.search(x) and not re.search(r"https?://|www\.|@|\{|\}", x):
            out.append(paras[:12])
    if path and path.endswith(".jsonl"):                        # đã lọc sẵn (dump) → mỗi dòng = danh sách đoạn
        return [json.loads(l) for l in open(path, encoding="utf-8")][:n_max]
    if path and os.path.exists(path):
        import pyarrow.parquet as pq
        for x in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            take(x)
            if len(out) >= n_max:
                break
    else:
        from datasets import load_dataset
        for r in load_dataset("HuggingFaceFW/fineweb-2", "vie_Latn", split="train", streaming=True):
            take(r["text"])
            if len(out) >= n_max:
                break
    return out


def headline(paras, r):
    """tiêu đề = câu đầu (≤ 16 từ), sapo = phần còn lại của đoạn đầu (≤ 320 ký tự), thân = các đoạn sau."""
    p0 = paras[0]
    m = re.match(r"(.+?[.!?])\s+(.*)", p0)
    first, rest = (m.group(1), m.group(2)) if m else (p0, "")
    w = first.rstrip(".!? ").split()
    if len(w) <= 18:                                             # câu đầu ngắn → cả câu làm tiêu đề
        title = " ".join(w)
    else:                                                        # dài → cắt ở dấu phẩy gần nhất trong 16 từ đầu
        cut = [i for i, x in enumerate(w[:16]) if x.endswith((",", ";", ":")) and i >= 5]
        title = " ".join(w[:cut[-1] + 1] if cut else w[:r.randint(11, 15)])
    title = title.rstrip(",;:")
    if 40 < len(rest) <= 320:
        sapo, body = rest, paras[1:]
    else:
        p1 = paras[1]
        sapo, body = (p1 if len(p1) <= 320 else p1[:320].rsplit(" ", 1)[0] + "…"), paras[2:]
    return title[:1].upper() + title[1:], sapo, body


def photos():
    """ảnh minh hoạ: ảnh COCO/KTVIC đã có mô tả tiếng Việt (TK_106 V1) → chú thích = mô tả ngắn."""
    p = "data/vlm/nat/vision.jsonl"
    if not os.path.exists(p):
        return []
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return [(os.path.abspath("data/vlm/" + r["image"]), r["mo_ta_ngan"]) for r in rows if r["src"] != "openvivqa"]


# ── dựng trang ────────────────────────────────────────────────────────────────────────
def e(s):
    return html.escape(s)


def when(r):
    d, mo, h, mi = r.randint(1, 28), r.randint(1, 12), r.randint(6, 22), r.choice(["05", "15", "20", "30", "45", "50"])
    return f"{r.choice(THU)}, {d}/{mo}/2026 {h:02d}:{mi} (GMT+7)", (d, mo)


def page_article(r, docs, pics):
    paras = r.choice(docs)
    title, sapo, body = headline(paras, r)
    site, cat, author = r.choice(SITES), r.choice(CATS), f"{r.choice(HO)} {r.choice(TEN)}"
    t, _ = when(r)
    acc = r.choice(["#9f224e", "#0b5394", "#c0392b", "#16794c", "#b45309", "#1f2937"])
    font, fs = r.choice(SANS + SERIF), r.choice([15, 16, 17, 18, 19])
    dark = r.random() < 0.12
    bg, fg, dim = ("#16181c", "#e7e7e7", "#9aa0a6") if dark else ("#ffffff", "#222222", "#757575")
    W = r.choice([1280, 1366, 1440, 1536, 1920]); H = r.choice([900, 1000, 1080]) if r.random() < 0.7 else 2200
    pic = r.choice(pics) if pics and r.random() < 0.6 else None
    others = [headline(r.choice(docs), r)[0] for _ in range(5)]
    nav = "".join(f"<span style='margin-right:20px'>{c}</span>" for c in CATS[:r.randint(7, 11)])
    fig = (f"<figure style='margin:14px 0'><img src='file://{pic[0]}' style='width:100%;max-height:420px;object-fit:cover'>"
           f"<figcaption data-k='cap' style='color:{dim};font-size:.85em;margin-top:6px;font-style:italic'>{e(pic[1])}</figcaption></figure>") if pic else ""
    ps = "".join(f"<p data-k='p{i}' style='margin:0 0 14px;text-align:justify'>{e(p)}</p>" for i, p in enumerate(body))
    side = "".join(f"<div data-k='r{i}' style='padding:9px 0;border-bottom:1px solid #ddd'><b style='color:{acc};margin-right:8px'>{i+1}</b>{e(h)}</div>"
                   for i, h in enumerate(others))
    mw = r.choice([0.62, 0.66, 0.7])
    h_ = (f"<div style='background:{acc};color:#fff;padding:10px 32px;font:bold 26px {font},sans-serif' data-k='site'>{e(site)}</div>"
          f"<div style='padding:8px 32px;border-bottom:1px solid #ddd;color:{dim};font-size:14px'>{nav}</div>"
          f"<div style='display:flex;gap:36px;padding:18px 32px'><div style='width:{mw*100:.0f}%'>"
          f"<div style='color:{dim};font-size:13px;margin-bottom:6px'>Trang chủ › <span data-k='cat'>{cat}</span></div>"
          f"<h1 data-k='title' style='font-size:1.9em;line-height:1.25;margin:4px 0 10px'>{e(title)}</h1>"
          f"<div data-k='meta' style='color:{dim};font-size:13px;margin-bottom:12px'>{e(author)} · {e(t)}</div>"
          f"<p data-k='sapo' style='font-weight:600;margin:0 0 12px;text-align:justify'>{e(sapo)}</p>{fig}{ps}</div>"
          f"<div style='flex:1'><div style='font-weight:700;border-bottom:3px solid {acc};padding-bottom:6px'>Đọc nhiều</div>{side}"
          f"<div style='margin-top:18px;height:220px;background:#eee;color:#999;display:flex;align-items:center;justify-content:center'>Quảng cáo</div></div></div>")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><style>body{{margin:0;background:{bg};color:{fg};"
           f"font:{fs}px/1.6 '{font}',sans-serif}}</style></head><body>{h_}</body></html>")
    meta = {"kind": "article", "site": site, "cat": cat, "author": author, "time": t, "title": title, "sapo": sapo,
            "body": body, "cap": pic[1] if pic else None, "others": others}
    return doc, W, H, meta


def page_home(r, docs, pics):
    site, acc = r.choice(SITES), r.choice(["#9f224e", "#0b5394", "#c0392b", "#16794c", "#1f2937"])
    font, fs = r.choice(SANS), r.choice([14, 15, 16])
    W, H = r.choice([1280, 1366, 1440, 1920]), r.choice([900, 1000, 1080])
    ncol = r.choice([3, 4])
    items = []
    for _ in range(ncol * r.choice([2, 3])):
        items.append((headline(r.choice(docs), r)[0], r.choice(CATS), f"{r.randint(1, 23)} giờ trước", r.choice(pics) if pics else None))
    cards = "".join(
        f"<div style='width:{100/ncol - 2:.1f}%'>"
        + (f"<img src='file://{p[0]}' style='width:100%;height:150px;object-fit:cover'>" if p else "<div style='height:150px;background:#ccd'></div>")
        + f"<div data-k='h{i}' style='font-weight:700;margin:8px 0 4px;line-height:1.35'>{e(t)}</div>"
        f"<div data-k='m{i}' style='color:#777;font-size:.85em'>{c} · {g}</div></div>"
        for i, (t, c, g, p) in enumerate(items))
    nav = "".join(f"<span style='margin-right:20px'>{c}</span>" for c in CATS)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><style>body{{margin:0;background:#fff;color:#222;font:{fs}px/1.5 '{font}',sans-serif}}</style></head><body>"
           f"<div data-k='site' style='background:{acc};color:#fff;padding:10px 32px;font:bold 26px {font},sans-serif'>{e(site)}</div>"
           f"<div style='padding:8px 32px;border-bottom:1px solid #ddd;color:#666;font-size:14px'>{nav}</div>"
           f"<div style='display:flex;flex-wrap:wrap;gap:2%;row-gap:22px;padding:18px 32px'>{cards}</div></body></html>")
    return doc, W, H, {"kind": "home", "site": site, "items": items}


def page_paper(r, docs, pics):
    paras = r.choice(docs)
    title, sapo, body = headline(paras, r)
    name = r.choice(PAPERS)
    d, mo = r.randint(1, 28), r.randint(1, 12)
    issue = f"Số {r.randint(100, 9999)} · {r.choice(THU)}, ngày {d} tháng {mo} năm 2026 · Giá: {r.choice(['5.000', '6.000', '7.000', '8.000'])} đồng"
    font, fs = r.choice(SERIF), r.choice([16, 17, 18, 19, 20])
    W, H = r.choice([(1240, 1754), (1400, 1900), (1300, 1800)])
    ncol = r.choice([3, 3, 4, 5]) if W >= 1300 else r.choice([3, 4])
    pic = r.choice(pics) if pics and r.random() < 0.6 else None
    brief = [headline(r.choice(docs), r)[0] for _ in range(3)]
    fig = (f"<figure style='margin:0 0 10px;break-inside:avoid'><img src='file://{pic[0]}' style='width:100%;filter:grayscale({r.choice([0, 1])})'>"
           f"<figcaption data-k='cap' style='font-size:.8em;font-style:italic'>{e(pic[1])}</figcaption></figure>") if pic else ""
    ps = "".join(f"<p data-k='p{i}' style='margin:0 0 8px;text-indent:1.4em'>{e(p)}</p>" for i, p in enumerate(body))
    br = "".join(f"<li data-k='b{i}' style='margin-bottom:6px'>{e(b)}</li>" for i, b in enumerate(brief))
    bgc = r.choice(["#f6f1e4", "#f3eee2", "#faf8f2", "#ffffff"])
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><style>body{{margin:0;background:{bgc};color:#1b1b1b;"
           f"font:{fs}px/1.42 '{font}',serif}}</style></head><body>"
           f"<div id='pg' style='width:{W}px;height:{H}px;padding:36px 48px;box-sizing:border-box;overflow:hidden'>"
           f"<div data-k='mast' style='text-align:center;font:900 64px \"{font}\",serif;letter-spacing:2px'>{name}</div>"
           f"<div data-k='issue' style='text-align:center;border-top:3px double #333;border-bottom:1px solid #333;padding:6px 0;margin:6px 0 18px;font-size:15px'>{issue}</div>"
           f"<h1 data-k='title' style='font-size:2.5em;line-height:1.15;margin:0 0 12px;text-align:center'>{e(title.upper() if r.random() < 0.4 else title)}</h1>"
           f"<p data-k='sapo' style='font-weight:700;font-style:italic;margin:0 0 14px;text-align:justify'>{e(sapo)}</p>"
           f"<div id='cols' style='column-count:{ncol};column-gap:26px;column-rule:1px solid #bbb;text-align:justify;height:{H - 640}px;column-fill:auto'>{fig}{ps}</div>"
           f"<div style='border-top:2px solid #333;margin-top:14px;padding-top:8px'><b>TIN VẮN</b><ul style='margin:6px 0 0 18px;padding:0'>{br}</ul></div>"
           f"</div></body></html>")
    meta = {"kind": "paper", "name": name, "issue": issue, "title": title, "sapo": sapo, "body": body,
            "cap": pic[1] if pic else None, "brief": brief}
    return doc, W, H, meta


# ── hỏi–đáp (chỉ phần HIỆN TRỌN trong ảnh) ────────────────────────────────────────────
def qa_of(meta, vis, r):
    q = []
    k = meta["kind"]
    ps = [i for i in range(len(meta.get("body", []))) if f"p{i}" in vis]
    if k == "article":
        if "site" in vis:
            q.append(("Đây là trang báo nào?", f"Đây là trang {meta['site']}."))
        if "title" in vis:
            q.append(("Tiêu đề bài báo trong ảnh là gì?", meta["title"]))
        if "meta" in vis and "cat" in vis:
            q.append(("Bài báo thuộc chuyên mục nào, ai viết và đăng lúc nào?",
                      f"Bài thuộc chuyên mục {meta['cat']}, tác giả {meta['author']}, đăng {meta['time']}."))
        if "sapo" in vis:
            q.append(("Chép lại đoạn sapo (đoạn mở đầu in đậm) của bài.", meta["sapo"]))
        if ps:
            j = r.choice(ps)
            q.append((f"Chép lại đoạn thứ {j + 1} của thân bài.", meta["body"][j]))
        if meta["cap"] and "cap" in vis:
            q.append(("Chú thích của ảnh trong bài viết là gì?", meta["cap"]))
        rs = [i for i in range(5) if f"r{i}" in vis]
        if len(rs) >= 3:
            q.append(("Mục \"Đọc nhiều\" có những tin nào?", "\n".join(f"{i + 1}. {meta['others'][i]}" for i in rs)))
        full = "\n\n".join([meta["title"], meta["sapo"]] + [meta["body"][i] for i in ps])
        if "title" in vis and "sapo" in vis and len(full) <= 1800:
            q.append(("Chép lại toàn bộ chữ của bài báo trong ảnh (tiêu đề, sapo và thân bài).", full))
        if "title" in vis and ps:
            q.append(("TEACH:Tóm tắt nội dung bài báo trong ảnh.", full[:3500]))
            q.append(("TEACH:Bài báo này nói về vấn đề gì? Nêu 3 ý chính.", full[:3500]))
    elif k == "home":
        its = [i for i in range(len(meta["items"])) if f"h{i}" in vis and f"m{i}" in vis]
        if "site" in vis:
            q.append(("Đây là trang chủ của báo nào?", f"Đây là trang chủ của {meta['site']}."))
        if len(its) >= 3:
            q.append(("Liệt kê các tiêu đề tin trên trang.", "\n".join(f"- {meta['items'][i][0]}" for i in its)))
            j = r.choice(its)
            t, c, g, _ = meta["items"][j]
            q.append((f"Tin thứ {j + 1} (đếm từ trái sang phải, trên xuống dưới) có tiêu đề gì, thuộc chuyên mục nào?",
                      f"Tin thứ {j + 1} có tiêu đề \"{t}\", thuộc chuyên mục {c}, đăng {g}."))
            c = meta["items"][r.choice(its)][1]
            hit = [meta["items"][i][0] for i in its if meta["items"][i][1] == c]
            q.append((f"Trên trang có tin nào thuộc chuyên mục {c} không?",
                      f"Có {len(hit)} tin thuộc chuyên mục {c}: " + "; ".join(f"\"{h}\"" for h in hit) + "."))
            q.append(("TEACH:Trang này có những tin nổi bật gì? Tóm tắt ngắn gọn.",
                      "\n".join(f"- {meta['items'][i][0]} ({meta['items'][i][1]})" for i in its)))
    else:
        if "mast" in vis and "issue" in vis:
            q.append(("Đây là tờ báo nào, số mấy, phát hành ngày nào?", f"Đây là báo {meta['name']}. {meta['issue']}."))
            q.append(("Giá bán của tờ báo là bao nhiêu?", f"Giá bán là {meta['issue'].split('Giá: ')[1]}."))
        if "title" in vis:
            q.append(("Tiêu đề bài chính trên trang báo là gì?", meta["title"]))
        if "sapo" in vis:
            q.append(("Chép lại đoạn in nghiêng đậm dưới tiêu đề.", meta["sapo"]))
        if ps:
            q.append(("Chép lại đoạn đầu tiên của bài chính.", meta["body"][ps[0]]))
            j = r.choice(ps)
            q.append((f"Chép lại đoạn thứ {j + 1} của bài chính.", meta["body"][j]))
        bs = [i for i in range(3) if f"b{i}" in vis]
        if len(bs) == 3:
            q.append(("Mục \"Tin vắn\" có những tin gì?", "\n".join(f"- {meta['brief'][i]}" for i in bs)))
        if meta["cap"] and "cap" in vis:
            q.append(("Chú thích dưới ảnh là gì?", meta["cap"]))
        if "title" in vis and ps:
            full = "\n\n".join([meta["title"], meta["sapo"]] + [meta["body"][i] for i in ps])
            q.append(("TEACH:Tóm tắt bài chính trên trang báo.", full[:3500]))
    return q


JS_VIS = """(box) => {
  const W = innerWidth, H = innerHeight, out = [];
  const c = document.getElementById('cols'); const cr = c ? c.getBoundingClientRect() : null;
  for (const el of document.querySelectorAll('[data-k]')) {
    const rs = [...el.getClientRects()]; if (!rs.length) continue;
    let ok = rs.every(q => q.top >= 0 && q.left >= 0 && q.bottom <= H && q.right <= W && q.width > 0);
    if (ok && cr && c.contains(el)) ok = rs.every(q => q.right <= cr.right + 1 && q.bottom <= cr.bottom + 1 && q.left >= cr.left - 1);
    if (ok) out.push(el.dataset.k);
  }
  return out; }"""


def photo_like(path, r):
    """trang báo in → giống ảnh chụp bằng điện thoại: nghiêng, phối cảnh, giấy ngả màu, bóng, mờ, nhiễu, nén JPEG."""
    from PIL import Image, ImageFilter, ImageEnhance
    import numpy as np
    im = Image.open(path).convert("RGB")
    w, h = im.size
    d = lambda: r.uniform(-0.035, 0.035)
    quad = (w * d(), h * d(), w * d(), h * (1 + d()), w * (1 + d()), h * (1 + d()), w * (1 + d()), h * d())
    im = im.transform((w, h), Image.QUAD, quad, Image.BICUBIC, fillcolor=tuple(r.randint(40, 120) for _ in range(3)))
    im = im.rotate(r.uniform(-2.5, 2.5), Image.BICUBIC, expand=False, fillcolor=(70, 70, 70))
    a = np.asarray(im).astype(np.float32)
    gx = np.linspace(r.uniform(0.75, 1.0), r.uniform(0.9, 1.08), w)[None, :, None]
    gy = np.linspace(r.uniform(0.8, 1.05), r.uniform(0.85, 1.05), h)[:, None, None]
    a = a * gx * gy * np.array([1.0, r.uniform(0.96, 1.0), r.uniform(0.86, 0.97)])[None, None]
    a += np.random.default_rng(r.randint(0, 10 ** 9)).normal(0, r.uniform(2, 7), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r.uniform(0.2, 0.9)))
    im = ImageEnhance.Contrast(im).enhance(r.uniform(0.85, 1.1))
    im.save(path, "JPEG", quality=r.randint(70, 88))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/vlm/news")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fineweb", default="data/fineweb2hq/vie_Latn/000_00000.parquet")
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--dump", action="store_true", help="chỉ tải/lọc bài rồi ghi docs.jsonl + docs_test.jsonl")
    a = ap.parse_args()
    r = random.Random(a.seed)
    docs = load_docs(a.fineweb, a.docs)
    r.shuffle(docs)
    os.makedirs(a.out, exist_ok=True)
    if a.dump:                                                  # tải/lọc 1 lần → file train/test RIÊNG (không trùng bài)
        k = max(500, len(docs) // 40)
        for name, part in (("docs_test.jsonl", docs[:k]), ("docs.jsonl", docs[k:])):
            with open(f"{a.out}/{name}", "w", encoding="utf-8") as f:
                for d_ in part:
                    f.write(json.dumps(d_, ensure_ascii=False) + "\n")
        print(f"### dump {len(docs) - k} bài train · {k} bài test → {a.out}/docs*.jsonl", flush=True)
        return
    from playwright.sync_api import sync_playwright
    pics = photos()
    print(f"[news] {len(docs)} bài · {len(pics)} ảnh minh hoạ", flush=True)
    os.makedirs(f"{a.out}/img", exist_ok=True)
    pw = sync_playwright().start(); br = pw.chromium.launch(); pg = br.new_page()
    tmp = os.path.abspath(f"{a.out}/_tmp_{a.split}_{a.seed}.html")
    n_ok = 0
    with open(f"{a.out}/{a.split}.jsonl", "a", encoding="utf-8") as f:
        for i in range(a.n):
            kind = r.choices(["article", "home", "paper"], [5, 2, 3])[0]
            doc, W, H, meta = {"article": page_article, "home": page_home, "paper": page_paper}[kind](r, docs, pics)
            open(tmp, "w", encoding="utf-8").write(doc)
            try:
                pg.set_viewport_size({"width": W, "height": H})
                pg.goto("file://" + tmp, wait_until="load")
                if kind == "article" and H > 1500:                       # trang dài: cắt đúng chiều cao nội dung (≤ H)
                    H = min(H, pg.evaluate("document.documentElement.scrollHeight"))
                    pg.set_viewport_size({"width": W, "height": H})
                vis = set(pg.evaluate(JS_VIS))
                name = f"{a.split}_{a.seed}_{i:06d}.jpg"
                path = f"{a.out}/img/{name}"
                pg.screenshot(path=path, type="jpeg", quality=85)
            except Exception as ex:
                print(f"[news] lỗi {i}: {ex}", flush=True); continue
            photo = kind == "paper" and r.random() < 0.5 or kind == "article" and r.random() < 0.1
            if photo:
                photo_like(path, r)
            qa = qa_of(meta, vis, r)
            if not qa:
                continue
            pick = r.sample(qa, min(len(qa), r.choice([1, 2, 2, 3])))
            msgs, teach = [], []
            for k, (q, ans) in enumerate(pick):
                if q.startswith("TEACH:"):
                    q = q[6:]; teach.append({"turn": k, "ref": ans}); ans = ""
                msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
            f.write(json.dumps({"images": [f"img/{name}"], "messages": msgs, "src": f"news_{kind}" + ("_chup" if photo else ""),
                                **({"teach": teach} if teach else {})}, ensure_ascii=False) + "\n")
            n_ok += 1
            if i % 200 == 0:
                print(f"[news] {i}/{a.n}", flush=True)
    br.close(); pw.stop()
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"### {n_ok} ảnh báo → {a.out}/{a.split}.jsonl", flush=True)


if __name__ == "__main__":
    main()
