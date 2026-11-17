"""§TK_077 — SINH DATA SFT ĐA DẠNG (khắc phục task-narrowing của QLoRA cũ).

QLoRA cũ hỏng vì train TOÀN 1 kiểu "viết đoạn văn về X" → quên dịch/email/toán.
Script này dựng corpus ĐA DẠNG 12 kiểu tác vụ, thầy = gemma4:26b (think=false).
Toán: sinh đề + TỰ tính đáp án bằng Python (chắc đúng), thầy chỉ viết lời giải.
Grounded (tóm tắt/QA): lấy đoạn THẬT từ viwiki làm ngữ cảnh.

Xuất data/lora_vi_div/{train,valid}.jsonl theo format messages (mlx_lm chat).

    .venv/bin/python scripts/gen_sft_diverse.py --scale 1.0
"""
import sys, os, time, json, random, re, argparse, urllib.request

VIWIKI = "data/text_corpus/viwiki.jsonl"
OLLAMA = "http://127.0.0.1:11434/api/chat"
TEACHER = "gemma4:26b-a4b-it-qat"
T0 = time.time()
RNG = random.Random(7)


def teach(prompt, num_predict=320, temperature=0.6, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    body = {"model": TEACHER, "messages": msgs, "stream": False, "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict}}
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            OLLAMA, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}), timeout=240)
        return json.load(r).get("message", {}).get("content", "").strip()
    except Exception as e:
        return ""


def load_leads(n=6000, minlen=160, maxlen=900):
    """Lấy n đoạn viwiki đủ dài; bỏ trang định-hướng (dễ gây bịa)."""
    out = []
    for line in open(VIWIKI):
        try:
            t = json.loads(line)
        except Exception:
            continue
        if not (isinstance(t, str) and minlen <= len(t) <= maxlen):
            continue
        head = t[:60].lower()
        if "có thể là" in head or "có thể chỉ" in head or "thường chỉ" in head:
            continue  # trang định-hướng, không phải fact
        out.append(t.strip())
        if len(out) >= n:
            break
    RNG.shuffle(out)
    return out


def topic_of(lead):
    """Chủ đề = cụm trước ' là '."""
    head = lead[:120]
    m = re.split(r"\s+là\s+|\s+hay\s+|\s+\(|,", head, maxsplit=1)[0].strip()
    return m if 2 < len(m) < 60 else lead[:40]


# ---- các bộ sinh theo kiểu tác vụ; trả về (user, assistant) hoặc None ----

def t_paragraph(leads):
    # NEO vào đoạn thật để tránh bịa: thầy diễn giải lại từ nguồn.
    p = RNG.choice(leads); tp = topic_of(p)
    a = teach(f"Dựa vào thông tin sau, viết một đoạn văn ngắn 3-5 câu mạch lạc bằng tiếng Việt "
              f"giới thiệu về '{tp}'. CHỈ dùng dữ kiện trong nguồn, không bịa thêm. "
              f"Chỉ viết đoạn văn.\n\nNguồn:\n{p}", 300)
    return (f"Viết một đoạn văn ngắn bằng tiếng Việt về: {tp}", a) if len(a) > 60 else None


def t_summarize(leads):
    p = RNG.choice(leads)
    a = teach(f"Tóm tắt đoạn sau thành 1-2 câu tiếng Việt, giữ ý chính:\n\n{p}", 160)
    return (f"Tóm tắt đoạn văn sau bằng 1-2 câu:\n\n{p}", a) if len(a) > 30 else None


def t_qa(leads):
    p = RNG.choice(leads)
    # thầy tự đặt câu hỏi + trả lời dựa trên đoạn
    raw = teach(f"Dựa DUY NHẤT vào đoạn sau, hãy tạo 1 câu hỏi tiếng Việt và câu trả lời "
                f"ngắn gọn cho nó. Trả về đúng dạng:\nHỏi: ...\nĐáp: ...\n\nĐoạn:\n{p}", 200)
    m = re.search(r"H[ỏo]i:\s*(.+?)\s*Đ[áa]p:\s*(.+)", raw, re.S)
    if not m:
        return None
    q, a = m.group(1).strip(), m.group(2).strip()
    return (q, a) if len(q) > 8 and len(a) > 3 else None


def t_translate_vi_en(leads):
    p = RNG.choice(leads)
    sent = re.split(r"(?<=[.!?])\s", p)[0][:300]
    a = teach(f"Dịch câu sau sang tiếng Anh, chỉ trả về bản dịch:\n\n{sent}", 200,
              temperature=0.3)
    return (f"Dịch câu sau sang tiếng Anh:\n\n{sent}", a) if len(a) > 10 else None


def t_translate_en_vi(_leads):
    seeds = [
        "The mitochondria is the powerhouse of the cell.",
        "Climate change is one of the most pressing issues of our time.",
        "She has been learning to play the piano for three years.",
        "The new policy will take effect at the beginning of next month.",
        "Artificial intelligence is transforming how we work and live.",
        "Please let me know if you have any questions about the report.",
        "The ancient city was discovered by archaeologists in 1922.",
        "A balanced diet and regular exercise are key to good health.",
        "The company announced a significant increase in quarterly profits.",
        "Reading books broadens the mind and enriches the soul.",
    ]
    s = RNG.choice(seeds)
    a = teach(f"Dịch câu tiếng Anh sau sang tiếng Việt tự nhiên, chỉ trả về bản dịch:\n\n{s}",
              160, temperature=0.3)
    return (f"Dịch câu sau sang tiếng Việt:\n\n{s}", a) if len(a) > 8 else None


def t_math(_leads):
    kind = RNG.choice(["mul", "pct", "rate", "div", "add_seq"])
    if kind == "mul":
        x, y = RNG.randint(11, 99), RNG.randint(100, 9999)
        ans = x * y
        q = f"Một cửa hàng bán {x} sản phẩm, mỗi sản phẩm giá {y} đồng. Tổng tiền là bao nhiêu?"
    elif kind == "pct":
        base, pct = RNG.choice([200, 500, 1200, 3000, 80000]), RNG.choice([5, 10, 15, 20, 25])
        ans = base * pct // 100
        q = f"{pct}% của {base} là bao nhiêu?"
    elif kind == "rate":
        t = RNG.choice([2, 3, 4]); per = RNG.choice([20, 30, 60])
        ans = (t * 60) // per if per else 0
        # số lần nhân đôi... đơn giản: quãng đường
        speed = RNG.choice([40, 50, 60]); hours = RNG.choice([2, 3, 4])
        ans = speed * hours
        q = f"Một xe chạy {speed} km/giờ trong {hours} giờ. Xe đi được bao nhiêu km?"
    elif kind == "div":
        per = RNG.randint(3, 12); each = RNG.randint(4, 20); tot = per * each
        ans = each
        q = f"Có {tot} cái kẹo chia đều cho {per} bạn. Mỗi bạn được mấy cái?"
    else:
        n = RNG.randint(5, 12); ans = n * (n + 1) // 2
        q = f"Tính tổng các số nguyên từ 1 đến {n}."
    a = teach(f"Giải bài toán sau bằng tiếng Việt, trình bày ngắn gọn các bước rồi kết luận. "
              f"ĐÁP ÁN ĐÚNG là {ans}. Hãy viết lời giải dẫn đến đúng đáp án đó.\n\nĐề: {q}",
              240, temperature=0.3)
    return (q, a) if (a and str(ans) in a) else None


def t_list(_leads):
    topics = ["giữ sức khỏe", "học tiếng Anh hiệu quả", "quản lý thời gian",
              "tiết kiệm tiền", "bảo vệ môi trường", "đọc sách hiệu quả",
              "chuẩn bị phỏng vấn xin việc", "ngủ ngon hơn", "nấu ăn ngon",
              "tập trung khi làm việc", "chăm sóc cây cảnh", "du lịch tiết kiệm"]
    tp = RNG.choice(topics)
    a = teach(f"Liệt kê 4-5 lời khuyên hữu ích để {tp}, mỗi ý một dòng bắt đầu bằng '- '.", 260)
    return (f"Cho tôi vài lời khuyên để {tp}.", a) if a.count("\n") >= 2 else None


def t_email(_leads):
    scen = ["xin nghỉ phép 2 ngày vì việc gia đình", "cảm ơn khách hàng đã hợp tác",
            "mời đồng nghiệp họp dự án lúc 3 giờ chiều thứ Năm",
            "xin lỗi vì giao hàng trễ và đề nghị bồi thường",
            "hỏi thông tin về khóa học tiếng Anh", "đề xuất tăng lương với sếp",
            "từ chối lời mời hợp tác một cách lịch sự"]
    s = RNG.choice(scen)
    a = teach(f"Viết một email tiếng Việt ngắn gọn, lịch sự để {s}. Có tiêu đề và lời chào.", 340)
    return (f"Viết giúp tôi một email để {s}.", a) if len(a) > 80 else None


def t_explain(leads):
    p = RNG.choice(leads); tp = topic_of(p)
    a = teach(f"Dựa vào thông tin sau, giải thích ngắn gọn dễ hiểu bằng tiếng Việt cho người mới "
              f"'{tp} là gì?'. CHỈ dùng dữ kiện trong nguồn, không bịa.\n\nNguồn:\n{p}", 260)
    return (f"Giải thích cho tôi: {tp} là gì?", a) if len(a) > 50 else None


def t_rewrite(leads):
    p = re.split(r"(?<=[.!?])\s", RNG.choice(leads))[0][:220]
    a = teach(f"Viết lại câu sau cho trang trọng và mạch lạc hơn, giữ nguyên ý, "
              f"chỉ trả về câu đã viết lại:\n\n{p}", 200)
    return (f"Viết lại câu sau cho hay và trang trọng hơn:\n\n{p}", a) if len(a) > 20 else None


def t_sentiment(_leads):
    revs = [("Sản phẩm giao nhanh, đóng gói cẩn thận, tôi rất hài lòng!", "tích cực"),
            ("Chất lượng quá tệ, dùng một hôm đã hỏng.", "tiêu cực"),
            ("Cũng bình thường, không có gì đặc biệt.", "trung tính"),
            ("Nhân viên phục vụ nhiệt tình, món ăn ngon tuyệt vời.", "tích cực"),
            ("Chờ đợi quá lâu mà không ai hỗ trợ, thất vọng.", "tiêu cực"),
            ("Giá hơi cao nhưng dùng tạm được.", "trung tính"),
            ("Tuyệt vời, chắc chắn sẽ quay lại lần sau.", "tích cực"),
            ("Không đúng như mô tả, tôi muốn trả hàng.", "tiêu cực")]
    txt, lab = RNG.choice(revs)
    a = f"Cảm xúc: {lab}."
    return (f"Phân loại cảm xúc (tích cực / tiêu cực / trung tính) của đánh giá sau:\n\n\"{txt}\"", a)


def t_creative(_leads):
    prompts = ["một câu chuyện ngắn 3 câu về một chú mèo lạc đường",
               "một bài thơ 4 dòng về mùa thu Hà Nội",
               "câu mở đầu hấp dẫn cho một truyện trinh thám",
               "một lời chúc sinh nhật ý nghĩa cho người bạn thân",
               "một slogan quảng cáo cho quán cà phê nhỏ",
               "một câu chuyện ngắn 3 câu về chuyến phiêu lưu của giọt nước"]
    p = RNG.choice(prompts)
    a = teach(f"Sáng tác {p} bằng tiếng Việt.", 240, temperature=0.9)
    return (f"Sáng tác {p}.", a) if len(a) > 30 else None


# (bộ sinh, tỷ trọng)
TASKS = [
    (t_paragraph, 0.14), (t_summarize, 0.13), (t_qa, 0.13),
    (t_translate_vi_en, 0.09), (t_translate_en_vi, 0.08),
    (t_math, 0.11), (t_list, 0.08), (t_email, 0.06),
    (t_explain, 0.09), (t_rewrite, 0.04), (t_sentiment, 0.03), (t_creative, 0.02),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 ≈ 1500 mẫu")
    ap.add_argument("--out", default="data/lora_vi_div")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    total = int(1500 * a.scale)
    leads = load_leads()
    print(f"[{time.time()-T0:.0f}s] {len(leads)} lead; mục tiêu {total} mẫu", flush=True)

    # kế hoạch số lượng mỗi kiểu
    plan = []
    for fn, w in TASKS:
        plan += [fn] * max(1, round(total * w))
    RNG.shuffle(plan)

    rows, counts = [], {}
    for i, fn in enumerate(plan):
        r = fn(leads)
        if r:
            u, asst = r
            rows.append({"messages": [{"role": "user", "content": u},
                                      {"role": "assistant", "content": asst}]})
            counts[fn.__name__] = counts.get(fn.__name__, 0) + 1
        if i % 25 == 0:
            print(f"[{time.time()-T0:.0f}s] {i}/{len(plan)} · giữ {len(rows)} · "
                  f"{fn.__name__}", flush=True)

    RNG.shuffle(rows)
    n_val = max(40, len(rows) // 20)
    val, tr = rows[:n_val], rows[n_val:]
    with open(f"{a.out}/train.jsonl", "w") as f:
        for r in tr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{a.out}/valid.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}: train {len(tr)} · valid {len(val)}", flush=True)
    print("phân bố:", json.dumps(counts, ensure_ascii=False), flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
