"""TK_104 §16 · KIẾN THỨC SÁCH GIÁO KHOA (GDPT 2018) cho 1.3B — thầy Gemma-4-31B, có kiểm lại, LẶP NHIỀU KIỂU để NHỚ được.

"Physics of Language Models": model chỉ LẤY RA được sự kiện khi gặp nó nhiều lần dưới nhiều cách diễn đạt → mỗi bài có:
  bài học tóm tắt (250–400 từ) + 8 cặp hỏi–đáp (hỏi cùng ý theo nhiều kiểu) + hội thoại nhiều lượt (giảng → hỏi tiếp)
Thân 1.3B KHÔNG có FFN (nơi model biến áp lưu sự kiện) → kiến thức này nạp vào EXPERT HỌC THUẬT mới (GatedFFN).

  topics : (môn, lớp) theo chương trình GDPT 2018 → thầy liệt kê ~18 bài/chủ đề chính → data/sgk/topics.jsonl
  gen    : mỗi bài → bài học + hỏi–đáp; thầy KIỂM LẠI (nhiệt độ 0) loại câu sai → data/sgk/raw.jsonl
  build  : → data/sgk/train.jsonl (bài học / hỏi–đáp / hội thoại nhiều lượt) + data/sgk/test.jsonl (1 hỏi–đáp GIỮ RIÊNG mỗi bài,
           kiến thức có trong bài học đã train → đo NHỚ, không đo đoán)

    python scripts/gen_sgk.py topics --ollama http://127.0.0.1:8000/v1
    python scripts/gen_sgk.py gen --workers 48 --ollama http://127.0.0.1:8000/v1
    python scripts/gen_sgk.py build
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_office import ollama, loads_loose, CJK

D = "data/sgk"
T0 = time.time()

CURRIC = {}
for g in range(1, 6):
    CURRIC[g] = ["Toán", "Tiếng Việt", "Đạo đức"] + (["Tự nhiên và Xã hội"] if g <= 3 else ["Khoa học", "Lịch sử và Địa lý"])
for g in range(6, 10):
    CURRIC[g] = ["Toán", "Ngữ văn", "Khoa học tự nhiên", "Lịch sử và Địa lý", "Giáo dục công dân", "Tin học", "Công nghệ"]
for g in range(10, 13):
    CURRIC[g] = ["Toán", "Ngữ văn", "Vật lý", "Hoá học", "Sinh học", "Lịch sử", "Địa lý", "Giáo dục kinh tế và pháp luật", "Tin học"]

P_TOPICS = """Liệt kê {n} bài học / chủ đề kiến thức CHÍNH của môn {mon} lớp {lop} theo chương trình giáo dục phổ thông Việt Nam
(GDPT 2018). Mỗi mục là tên ngắn gọn, cụ thể (không ghi "Ôn tập", "Kiểm tra"). Trả về DUY NHẤT JSON: {{"topics": ["...", ...]}}"""

P_GEN = """Bạn là giáo viên giỏi, soạn tài liệu cho học sinh Việt Nam.
Môn {mon} lớp {lop} — bài: "{topic}".
Viết đúng định dạng sau (không thêm gì khác):
<<<BAI_HOC>>>
Tóm tắt kiến thức trọng tâm của bài (250–400 từ, tiếng Việt chuẩn): khái niệm/định nghĩa, công thức hoặc sự kiện quan trọng
(kèm số liệu, mốc thời gian, tên riêng chính xác nếu có), 1–2 ví dụ minh hoạ. Chỉ viết điều CHẮC CHẮN ĐÚNG theo SGK.
Viết công thức bằng ký hiệu thường (vd: V = π × r² × h, x² − 1 = 0), KHÔNG dùng LaTeX ($...$, \\frac, \\times).
<<<HOI_DAP>>>
8 cặp hỏi–đáp về kiến thức TRONG bài học trên, mỗi cặp viết trên MỘT dòng theo mẫu: H: <câu hỏi> || Đ: <trả lời>
Câu hỏi tự nhiên như học sinh hỏi, đa dạng cách hỏi (có vài cặp hỏi CÙNG một ý bằng cách khác nhau).
Câu trả lời như một người thầy tận tình: nêu THẲNG ý chính ngay câu đầu, rồi giải thích thêm / ví dụ ngắn (tổng 2–5 câu,
không cộc lốc một câu); khoảng một nửa số câu trả lời kết bằng một câu hỏi mở gợi ý học tiếp (vd: "Em có muốn tìm hiểu
thêm về … không?"). Không vòng vo, không lặp lại câu hỏi."""

P_CHECK = """Kiểm tra tài liệu học tập dưới đây (môn {mon} lớp {lop}). Tìm lỗi SAI KIẾN THỨC (sai sự kiện, số liệu, công thức, tên riêng).
BÀI HỌC:
{lesson}

HỎI–ĐÁP:
{qa}

Trả về DUY NHẤT JSON: {{"bai_hoc_dung": true/false, "qa_sai": [số thứ tự các cặp hỏi–đáp SAI, bắt đầu từ 1]}}"""


def cmd_topics(a):
    os.makedirs(D, exist_ok=True)
    jobs = [(lop, mon) for lop, ms in CURRIC.items() for mon in ms]

    def one(j):
        lop, mon = j
        try:
            o = loads_loose(ollama(P_TOPICS.format(n=a.n, mon=mon, lop=lop), a.ollama, temp=0.3, fmt="json", num_predict=800))
            return [{"lop": lop, "mon": mon, "topic": t.strip()} for t in o["topics"] if isinstance(t, str) and t.strip()]
        except Exception as e:
            print("lỗi", j, e, flush=True); return []
    out = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(one, jobs):
            out += r
    seen, uniq = set(), []
    for r in out:
        k = (r["lop"], r["mon"], r["topic"].lower())
        if k not in seen:
            seen.add(k); uniq.append(r)
    with open(f"{D}/topics.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(uniq):
            f.write(json.dumps({"i": i, **r}, ensure_ascii=False) + "\n")
    print(f"### {len(jobs)} (môn, lớp) → {len(uniq)} bài · {time.time()-T0:.0f}s")


def parse_qa(block):
    out = []
    for line in block.splitlines():
        m = re.match(r"\s*(?:\d+[.)]\s*)?H:\s*(.+?)\s*\|\|\s*Đ:\s*(.+)$", line.strip())
        if m:
            out.append({"q": m.group(1).strip(), "a": m.group(2).strip()})
    return out


def work(args):
    r, url = args
    try:
        t = ollama(P_GEN.format(**r), url, temp=0.7, num_predict=1800)
        if "<<<BAI_HOC>>>" not in t or "<<<HOI_DAP>>>" not in t:
            return {**r, "drop": "sai định dạng"}
        lesson, qa_block = t.split("<<<BAI_HOC>>>", 1)[1].split("<<<HOI_DAP>>>", 1)
        lesson, qa = lesson.strip(), parse_qa(qa_block)
        if len(lesson) < 400 or len(qa) < 5 or CJK.search(lesson + qa_block):
            return {**r, "drop": f"ngắn/lỗi ({len(lesson)} ký tự, {len(qa)} cặp)"}
        qa_txt = "\n".join(f"{k+1}. H: {x['q']} || Đ: {x['a']}" for k, x in enumerate(qa))
        c = loads_loose(ollama(P_CHECK.format(mon=r["mon"], lop=r["lop"], lesson=lesson, qa=qa_txt), url, temp=0, fmt="json",
                               num_predict=200))
        if c.get("bai_hoc_dung") is False:
            return {**r, "drop": "bài học sai kiến thức"}
        bad = {int(x) for x in c.get("qa_sai", []) if str(x).isdigit()}
        qa = [x for k, x in enumerate(qa) if k + 1 not in bad]
        return {**r, "lesson": lesson, "qa": qa, "qa_loai": len(bad)}
    except Exception as e:
        return {**r, "drop": f"{type(e).__name__}: {str(e)[:80]}"}


def cmd_gen(a):
    topics = [json.loads(l) for l in open(f"{D}/topics.jsonl", encoding="utf-8")]
    out = f"{D}/raw.jsonl"
    done = {json.loads(l)["i"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    todo = [(r, a.ollama) for r in topics if r["i"] not in done]
    print(f"[sgk] {len(topics)} bài · còn {len(todo)}", flush=True)
    kept = 0
    with open(out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            kept += "drop" not in r
            if k % 50 == 0 or k == len(todo):
                el = time.time() - T0
                print(f"[{el:.0f}s] {k}/{len(todo)} · đạt {kept} · {60*k/el:.1f} bài/phút", flush=True)
    print("### sgk xong", flush=True)


ASK_LESSON = ["Giảng giúp em bài \"{t}\" ({m} lớp {l}) với.", "Tóm tắt kiến thức bài {t} môn {m} lớp {l}.",
              "Cho mình hỏi bài {t} trong {m} lớp {l} có những ý chính nào?", "Ôn lại giúp em bài {t} ({m} {l}) nhé."]


def cmd_build(a):
    rng = random.Random(16)
    rows = []
    for l in open(f"{D}/raw.jsonl", encoding="utf-8"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if "drop" not in r and len(r.get("qa", [])) >= 4:
            rows.append(r)
    train, test = [], []
    for r in rows:
        qa = list(r["qa"]); rng.shuffle(qa)
        test.append({"q": qa[0]["q"], "ref": qa[0]["a"], "mon": r["mon"], "lop": r["lop"], "topic": r["topic"]})
        rest = qa[1:]
        ask = rng.choice(ASK_LESSON).format(t=r["topic"], m=r["mon"], l=r["lop"])
        train.append({"messages": [{"role": "user", "content": ask}, {"role": "assistant", "content": r["lesson"]}], "src": "sgk_bai"})
        for x in rest:
            train.append({"messages": [{"role": "user", "content": x["q"]}, {"role": "assistant", "content": x["a"]}], "src": "sgk_qa"})
        k = rng.randint(2, min(4, len(rest)))                     # hội thoại nhiều lượt: giảng bài → hỏi tiếp
        conv = [{"role": "user", "content": ask}, {"role": "assistant", "content": r["lesson"]}]
        for x in rest[:k]:
            conv += [{"role": "user", "content": x["q"]}, {"role": "assistant", "content": x["a"]}]
        train.append({"messages": conv, "src": "sgk_hoithoai"})
    rng.shuffle(train)
    with open(f"{D}/train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{D}/test.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"bài đạt {len(rows)} → train {len(train)} mẫu {dict(Counter(r['src'] for r in train))} · test {len(test)} câu")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("topics"); t.add_argument("--n", type=int, default=18); t.add_argument("--workers", type=int, default=16)
    g = sp.add_parser("gen"); g.add_argument("--workers", type=int, default=48)
    for p in (t, g):
        p.add_argument("--ollama", default="http://127.0.0.1:8000/v1")
    sp.add_parser("build")
    a = ap.parse_args()
    {"topics": cmd_topics, "gen": cmd_gen, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
