"""§TK_103 · BỘ ĐÁNH-GIÁ SẠCH cho trợ-lý tiếng Việt (thay bộ 67 câu bị nhiễm — §20).

Bài học §20: 60 câu held-out cũ lấy từ `sft_mix` = dữ-liệu SFT của model 1.3B → 21/60 câu trả-lời chép
nguyên đáp-án mẫu. Bộ này tạo câu MỚI và loại mọi câu trùng/gần-trùng với 85K câu hỏi trong `sft_mix`.

4 phần (thầy soạn = Gemma 4 26B qua Ollama, `think: false`):
  • open           80 câu — 8 nhóm nhu-cầu × 10, chấm bằng giám-khảo (tự-nhiên / đúng-hữu-ích)
  • fact           40 câu — rút từ bài viwiki dài; đáp-án ngắn PHẢI xuất-hiện nguyên-văn trong bài
  • unanswerable   20 câu — hỏi về thực-thể BỊA (người/sách/luật/địa-danh); trả lời đúng = nói không biết
  • math           20 câu — toán lời-văn đáp-số nguyên; Gemma giải lại ĐỘC-LẬP, chỉ giữ câu hai lần khớp

    .venv/bin/python scripts/build_eval_clean.py            # → data/eval_clean/eval_v1.jsonl
"""
from __future__ import annotations
import json, os, random, re, sys, time, unicodedata, urllib.request
from collections import defaultdict

OLLAMA = "http://127.0.0.1:11434/api/chat"
GEMMA = "gemma4:26b-a4b-it-qat"
OUT = "data/eval_clean/eval_v1.jsonl"
RNG = random.Random(20260930)

OPEN = {
    "van_phong": "việc văn phòng: viết email công việc, thông báo, biên bản, lịch họp, báo cáo ngắn, trả lời khách hàng",
    "tom_tat": "tóm tắt hoặc viết lại một đoạn văn bản CÓ SẴN NGAY TRONG CÂU HỎI (tự viết kèm đoạn văn 60-120 chữ: tin tức, thông báo, nội quy, email dài)",
    "lap_trinh": "lập trình: viết hàm nhỏ, giải thích đoạn code, sửa lỗi, câu lệnh SQL, Excel, git (Python, JavaScript, SQL)",
    "dich": "dịch thuật Anh-Việt hoặc Việt-Anh câu/đoạn ngắn, ghi rõ chiều dịch và kèm câu cần dịch",
    "kien_thuc": "kiến thức phổ thông mức sách giáo khoa: khoa học, địa lý, lịch sử thế giới, sinh học, vật lý, cần giải thích",
    "doi_song": "đời sống và sức khoẻ thường ngày: nấu ăn, chăm con, ngủ nghỉ, tập thể dục, tài chính cá nhân, mẹo vặt",
    "phap_luat": "pháp luật và thủ tục hành chính Việt Nam: giấy tờ, hợp đồng lao động, thuế thu nhập, giao thông, cư trú",
    "suy_luan": "suy luận và lập luận: câu đố logic, so sánh ưu nhược điểm, lập kế hoạch, ra quyết định có lý do",
}


def gemma(prompt, temp=0.8, fmt_json=True, timeout=900, think=False, tries=3):
    """think=True: Gemma suy-nghĩ nội-bộ (chậm hơn, chính-xác hơn) — dùng khi KIỂM đáp-án.
    JSON hỏng/cắt cụt → thử lại; hết lượt → {} (không làm sập cả script)."""
    body = {"model": GEMMA, "stream": False, "think": think, "options": {"temperature": temp},
            "messages": [{"role": "user", "content": prompt}]}
    if fmt_json:
        body["format"] = "json"
    for _ in range(tries):
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        txt = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
        if not fmt_json:
            return txt
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            continue
    return {}


def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"[^\w\s]", " ", s)


def shingles(s, n=3):
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


class Contam:
    """Chỉ-mục 3-gram từ của mọi câu hỏi trong sft_mix → phát-hiện trùng/gần-trùng."""

    def __init__(self, path="data/sft_vi/sft_mix.jsonl"):
        self.inv, self.size = defaultdict(set), []
        for line in open(path, encoding="utf-8"):
            for m in json.loads(line)["messages"]:
                if m["role"] == "user":
                    sh = shingles(m["content"])
                    k = len(self.size)
                    self.size.append(len(sh))
                    for g in sh:
                        self.inv[g].add(k)
        print(f"[contam] {len(self.size)} câu hỏi SFT đã lập chỉ-mục", flush=True)

    def worst(self, q):
        sh = shingles(q)
        cnt = defaultdict(int)
        for g in sh:
            for k in self.inv.get(g, ()):
                cnt[k] += 1
        if not cnt:
            return 0.0
        return max(c / (len(sh) + self.size[k] - c) for k, c in cnt.items())   # Jaccard lớn nhất


def gen_open(con):
    items = []
    for cat, desc in OPEN.items():
        got = []
        for attempt in range(3):
            r = gemma(f"""Hãy viết 14 câu hỏi/yêu cầu KHÁC NHAU mà một người Việt thực sự sẽ gõ cho trợ lý AI,
về chủ đề: {desc}.
Đa dạng độ dài (từ 1 câu ngắn đến 4-5 câu), văn phong (trang trọng, bình dân, có lỗi gõ nhẹ), người hỏi
(học sinh, nhân viên văn phòng, người lớn tuổi, chủ quán). Mỗi câu phải tự đủ nghĩa, không cần ngữ cảnh ngoài.
Trả về JSON: {{"items": ["...", "..."]}}""", temp=0.9)
            for q in r.get("items", []):
                q = str(q).strip()
                if len(q) >= 12 and con.worst(q) < 0.35 and all(norm(q) != norm(x) for x in got):
                    got.append(q)
            if len(got) >= 10:
                break
        RNG.shuffle(got)
        items += [{"type": "open", "category": cat, "q": q} for q in got[:10]]
        print(f"[open] {cat}: {min(10, len(got))} câu", flush=True)
    return items


def gen_fact(con, n=40):
    pool = []
    for line in open("data/text_corpus/viwiki.jsonl", encoding="utf-8"):
        t = json.loads(line)
        if isinstance(t, str) and len(t) > 6000 and "có thể là" not in t[:80].lower():
            pool.append(t)
    RNG.shuffle(pool)
    print(f"[fact] {len(pool)} bài viwiki dài (>6000 ký-tự) làm nguồn", flush=True)
    items = []
    for art in pool:
        if len(items) >= n:
            break
        passage = art[:2500]
        try:
            r = gemma(f"""Đoạn văn (trích Wikipedia tiếng Việt):
\"\"\"{passage}\"\"\"

Viết MỘT câu hỏi tiếng Việt mà người hiểu biết có thể trả lời, với đáp án NGẮN (tên người, con số, năm,
địa danh) được nêu NGUYÊN VĂN trong đoạn văn. Câu hỏi phải tự đủ nghĩa: nêu rõ đối tượng, không viết
"theo đoạn văn". Tránh hỏi chi tiết quá vụn vặt.
Trả về JSON: {{"question": "...", "answer": "..."}}""", temp=0.3)
        except Exception:
            continue
        q, a = str(r.get("question", "")).strip(), str(r.get("answer", "")).strip()
        if not q or not a or len(a) > 60 or norm(a).strip() not in norm(passage) or con.worst(q) >= 0.35:
            continue
        items.append({"type": "fact", "category": "su_kien", "q": q, "gold": a,
                      "wiki_title": re.split(r" là |\(|,", art, maxsplit=1)[0][:80]})
    print(f"[fact] {len(items)} câu (đáp-án kiểm nguyên-văn trong bài)", flush=True)
    return items


def gen_unanswerable(con, n=20):
    titles = set()
    for line in open("data/text_corpus/viwiki.jsonl", encoding="utf-8"):
        t = json.loads(line)
        if isinstance(t, str):
            titles.add(norm(re.split(r" là |\(|,", t, maxsplit=1)[0]).strip())
    r = gemma("""Viết 30 câu hỏi tiếng Việt về những thứ KHÔNG TỒN TẠI nhưng nghe như có thật: người nổi tiếng
bịa (có họ tên Việt), cuốn sách/tiểu thuyết bịa, điều luật/nghị định bịa (số hiệu bịa), địa danh bịa, phát
minh bịa, sự kiện lịch sử bịa. Hỏi tự nhiên như thể chúng có thật (ví dụ hỏi tác giả, năm, nội dung).
Trợ lý trung thực phải trả lời là không biết/không tìm thấy thông tin.
Trả về JSON: {"items": [{"question": "...", "fake_entity": "tên thực thể bịa"}]}""", temp=0.9)
    items = []
    for it in r.get("items", []):
        q, ent = str(it.get("question", "")).strip(), norm(str(it.get("fake_entity", ""))).strip()
        if q and ent and ent not in titles and con.worst(q) < 0.35:
            items.append({"type": "unanswerable", "category": "khong_co_that", "q": q,
                          "fake_entity": it.get("fake_entity")})
    RNG.shuffle(items)
    print(f"[unanswerable] {min(n, len(items))} câu (thực-thể không là tiêu-đề viwiki)", flush=True)
    return items[:n]


def gen_math(con, n=20):
    items = []
    for _ in range(3):
        r = gemma("""Viết 12 bài toán lời văn tiếng Việt mức tiểu học/THCS (mua bán, chuyển động, tỉ lệ phần trăm,
chia đều, tuổi, hình học đơn giản), mỗi bài cần 2-4 bước tính, ĐÁP SỐ LÀ MỘT SỐ NGUYÊN.
Trả về JSON: {"items": [{"question": "...", "answer": 123}]}""", temp=0.9)
        for it in r.get("items", []):
            q = str(it.get("question", "")).strip()
            try:
                a = int(str(it.get("answer")).replace(".", "").replace(",", "").strip())
            except ValueError:
                continue
            if not q or con.worst(q) >= 0.35:
                continue
            chk = gemma(f"Giải bài toán sau thật cẩn thận.\n{q}\n"
                        'Chỉ trả về JSON: {"answer": <số nguyên>}', temp=0, think=True)
            try:
                a2 = int(str(chk.get("answer")).replace(".", "").replace(",", "").strip())
            except ValueError:
                continue
            if a2 == a:                                         # hai lần giải độc-lập khớp → giữ
                items.append({"type": "math", "category": "toan", "q": q, "gold": str(a)})
        if len(items) >= n:
            break
    print(f"[math] {min(n, len(items))} câu (hai lần giải khớp đáp-số)", flush=True)
    return items[:n]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    T0 = time.time()
    con = Contam()
    items = []
    for name, fn in [("open", gen_open), ("fact", gen_fact), ("unanswerable", gen_unanswerable), ("math", gen_math)]:
        stage = os.path.join(os.path.dirname(OUT), f"_stage_{name}.jsonl")      # lưu TỪNG PHẦN
        if os.path.exists(stage):
            part = [json.loads(l) for l in open(stage, encoding="utf-8")]
            print(f"[{name}] dùng lại {len(part)} câu đã lưu", flush=True)
        else:
            part = fn(con)
            with open(stage, "w", encoding="utf-8") as f:
                for it in part:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        items += part
    for i, it in enumerate(items):
        it["id"] = f"ev1_{it['type']}_{i:03d}"
        it["contam_jaccard"] = round(con.worst(it["q"]), 3)
    with open(OUT, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    by = defaultdict(int)
    for it in items:
        by[it["type"]] += 1
    print(f"### LƯU {OUT} · {len(items)} câu · {dict(by)} · trùng-SFT lớn nhất "
          f"{max(it['contam_jaccard'] for it in items):.2f} · {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
