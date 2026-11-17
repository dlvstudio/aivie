"""VLM · ẢNH CHỮ TỔNG HỢP để dạy model ĐỌC CHỮ (TK_106 §5 — chẩn đoán: V1 đọc chữ kém, thiếu dữ liệu hay nén quá?).

Vẽ bằng PIL (không cần trình duyệt → rất nhanh): đoạn văn tiếng Việt thật (bài SGK, văn phòng, hội thoại) với phông /
cỡ chữ / màu / độ rộng ngẫu nhiên → đáp án chính xác: chép lại cả đoạn, dòng thứ N, từ đầu/cuối, đếm dòng.

    python scripts/gen_ocr.py --n 40000 --out data/vlm/ocr --workers 24
"""
from __future__ import annotations
import argparse, glob, json, os, random, re, sys
import concurrent.futures as cf


def fonts():
    fs = []
    for p in glob.glob("/usr/share/fonts/truetype/**/*.ttf", recursive=True) + glob.glob("/System/Library/Fonts/Supplemental/*.ttf"):
        b = os.path.basename(p)
        if re.match(r"(NotoSans|NotoSerif|NotoSansMono|Liberation|DejaVu|Roboto|Arial|Times|Georgia|Verdana|Tahoma)[A-Za-z-]*\.ttf$", b) \
                and not re.search(r"(Condensed|Thin|Light|ExtraLight|Black|Display|UI)", b) and "NotoSans" + "Thai" not in b:
            if b.startswith(("NotoSans", "NotoSerif")) and not re.match(r"Noto(Sans|Serif|SansMono)-(Regular|Bold|Italic|BoldItalic|Medium)\.ttf$", b):
                continue
            fs.append(p)
    return sorted(set(fs))


def texts():
    out = []
    for p in ("data/sgk/train.jsonl", "data/office/train.jsonl", "data/multiturn/train.jsonl", "data/calib/train.jsonl"):
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            for m in json.loads(l)["messages"]:
                t = re.sub(r"[*#`]", "", m["content"]).strip()
                if 40 < len(t) < 3000 and "```" not in m["content"]:
                    out.append(t)
    return out


def wrap(draw, text, font, width):
    lines = []
    for para in text.split("\n"):
        words, cur = para.split(), ""
        for w in words:
            t = (cur + " " + w).strip()
            if draw.textlength(t, font=font) <= width:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        lines.append(cur)
    return [l for l in lines if l.strip()]


G = {}


def init(fs, src, num_frac=0.0, photo_frac=0.0):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    G.update(fs=fs, src=src, num_frac=num_frac, photo_frac=photo_frac)


SP = ["Cà phê sữa", "Bánh mì thịt", "Trà đào", "Nước suối", "Phở bò", "Cơm tấm", "Sữa tươi", "Bút bi", "Vở 96 trang",
      "Áo thun", "Dầu gội", "Gạo ST25", "Nước mắm", "Mì gói", "Trứng gà", "Bột giặt", "Pin AA", "Khẩu trang"]


def vn(x):
    return f"{x:,}".replace(",", ".")


def numbers_text(r):
    """chữ NHIỀU SỐ (V5): hoá đơn bán lẻ / dòng mã đơn–ngày–tiền–SĐT → (chữ, hỏi–đáp đọc-trước-rồi-trả-lời)."""
    if r.random() < 0.5:
        items = []
        for _ in range(r.randint(2, 7)):
            q, p = r.randint(1, 12), r.choice([5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 45, 60, 89, 120]) * 1000
            items.append((r.choice(SP), q, p, q * p))
        tot = sum(x[3] for x in items)
        lines = [r.choice(["HOÁ ĐƠN BÁN LẺ", "PHIẾU THANH TOÁN", "HÓA ĐƠN"]),
                 f"Ngày: {r.randint(1, 28):02d}/{r.randint(1, 12):02d}/2026  Số: {r.randint(1000, 99999)}"]
        lines += [f"{n} x{q}  {vn(p)}  = {vn(t)}" for n, q, p, t in items]
        lines.append(f"Tổng cộng: {vn(tot)} đ")
        rd = "; ".join(f"{n} {vn(t)}" for n, _, _, t in items)
        qa = [("Tổng tiền trên hoá đơn là bao nhiêu?", f"Dòng cuối hoá đơn ghi \"Tổng cộng: {vn(tot)} đ\". Vậy tổng tiền là {vn(tot)} đồng."),
              ("Liệt kê các món và thành tiền trên hoá đơn.", f"Các món: {rd}. Tổng cộng {vn(tot)} đồng.")]
        j = r.randrange(len(items))
        n, q, p, t = items[j]
        qa.append((f"Món \"{n}\" mua bao nhiêu, đơn giá và thành tiền là bao nhiêu?",
                   f"Dòng \"{n}\" ghi: số lượng {q}, đơn giá {vn(p)}, thành tiền {vn(t)} đồng."))
        return "\n".join(lines), qa
    rows = []
    for _ in range(r.randint(3, 8)):
        rows.append((f"DH{r.randint(10000, 99999)}", f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/2026",
                     vn(r.randint(5, 9999) * 1000), f"09{r.randint(0, 9)}{r.randint(1000000, 9999999)}"))
    lines = [f"Mã {a}  Ngày {b}  Số tiền {c} đ  SĐT {d}" for a, b, c, d in rows]
    j = r.randrange(len(rows))
    a, b, c, d = rows[j]
    qa = [(f"Đơn hàng {a} có số tiền bao nhiêu và đặt ngày nào?", f"Dòng mã {a} ghi: ngày {b}, số tiền {c} đ, SĐT {d}. Vậy đơn {a} đặt ngày {b}, số tiền {c} đồng."),
          (f"Số điện thoại của đơn {a} là gì?", f"Dòng mã {a} ghi SĐT {d}.")]
    return "\n".join(lines), qa


def make(args):
    i, seed, out = args
    fs, src = G["fs"], G["src"]
    from PIL import Image, ImageDraw, ImageFont
    r = random.Random(seed)
    num_qa = None
    if r.random() < G.get("num_frac", 0.0):
        text, num_qa = numbers_text(r)
    else:
        text = r.choice(src)
    if len(text) > 900:                                      # đoạn vừa phải: ≤ ~14 dòng
        k = r.randrange(0, len(text) - 600)
        text = text[k:k + r.randint(200, 800)]
        text = text[text.find(" ") + 1:text.rfind(" ")]
    size = r.choice([12, 13, 14, 15, 16, 18, 20, 22, 24, 28])
    font = ImageFont.truetype(r.choice(fs), size)
    W = r.choice([360, 480, 640, 800, 960, 1200, 1400])
    dark = r.random() < 0.25
    bg = (r.randint(15, 45),) * 3 if dark else tuple(r.randint(225, 255) for _ in range(3))
    fg = tuple(r.randint(200, 255) for _ in range(3)) if dark else tuple(r.randint(0, 70) for _ in range(3))
    img = Image.new("RGB", (W, 10), bg)
    d = ImageDraw.Draw(img)
    pad = r.randint(8, 40)
    lines = wrap(d, text, font, W - 2 * pad)[:r.randint(1, 14) if num_qa is None else 40]
    if not lines:
        return None
    lh = int(size * r.uniform(1.25, 1.7))
    H = 2 * pad + lh * len(lines)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    for k, ln in enumerate(lines):
        d.text((pad, pad + k * lh), ln, font=font, fill=fg)
    name = f"img/{seed}.png"
    img.save(f"{out}/{name}")
    if r.random() < G.get("photo_frac", 0.0):                   # giống ảnh chụp (nghiêng, mờ, nhiễu, JPEG)
        from gen_news import photo_like
        os.rename(f"{out}/{name}", f"{out}/img/{seed}.jpg"); name = f"img/{seed}.jpg"
        photo_like(f"{out}/{name}", r)
    full = "\n".join(lines)
    if num_qa is not None and len(lines) == len(text.split("\n")):  # hoá đơn hiện trọn (không bị gói dòng) → hỏi số
        q, a = r.choice(num_qa) if r.random() < 0.7 else ("Chép lại chữ trong ảnh.", full)
        return {"images": [name], "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                "src": "ocr_so", "max_tiles": 12}
    qs = [("Chép lại chữ trong ảnh.", full), ("Đọc giúp mình nội dung trong ảnh này.", full)]
    if len(lines) >= 2:
        n = r.randrange(len(lines))
        qs.append((f"Dòng thứ {n + 1} trong ảnh viết gì?", lines[n]))
        qs.append(("Ảnh có bao nhiêu dòng chữ?", f"Ảnh có {len(lines)} dòng chữ."))
    qs.append(("Câu đầu tiên trong ảnh là gì?", lines[0].split(". ")[0].strip()))
    q, a = r.choice(qs[:2]) if r.random() < 0.6 else r.choice(qs)
    return {"images": [name], "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            "src": "ocr_synth", "max_tiles": 12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default="data/vlm/ocr")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-frac", type=float, default=0.0, help="V5: tỉ lệ ảnh nhiều số (hoá đơn, mã đơn/ngày/tiền/SĐT)")
    ap.add_argument("--photo-frac", type=float, default=0.0, help="V5: tỉ lệ làm giống ảnh chụp")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/img", exist_ok=True)
    fs, src = fonts(), texts()
    print(f"[ocr] {len(fs)} phông · {len(src)} đoạn văn", flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(src)
    jobs = [(i, a.seed * 10 ** 7 + i, a.out) for i in range(a.n)]
    n = 0
    with cf.ProcessPoolExecutor(a.workers, initializer=init, initargs=(fs, src[:50000], a.num_frac, a.photo_frac)) as ex, open(f"{a.out}/train.jsonl", "w", encoding="utf-8") as f:
        for row in ex.map(make, jobs, chunksize=64):
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    print(f"### {n} ảnh chữ → {a.out}/train.jsonl", flush=True)


if __name__ == "__main__":
    main()
