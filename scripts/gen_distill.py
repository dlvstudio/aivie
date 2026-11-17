"""§TK_075 — SINH DATA CHƯNG CẤT: gemma4:26b (thầy) viết đoạn văn tiếng Việt SẠCH
về nhiều chủ đề → corpus để train mạng nhỏ (trò). Distillation kiểu data-level.

Chủ đề trích từ lead viwiki (cụm trước "là"). Thầy think=false (nhanh). Xuất jsonl
{"topic","text"} → tokenize bằng ingest_text.py → train.

    .venv/bin/python scripts/gen_distill.py --n 2000 --out data/distill/gemma_paras.jsonl
"""
import sys, time, json, argparse, re, urllib.request, random

VIWIKI = "data/text_corpus/viwiki.jsonl"
OLLAMA = "http://127.0.0.1:11434/api/chat"
TEACHER = "gemma4:26b-a4b-it-qat"


def gen(topic, num_predict=256):
    prompt = (f"Viết một đoạn văn ngắn 3-5 câu, chính xác và mạch lạc, bằng tiếng Việt, "
              f"giới thiệu về: {topic}. Chỉ viết nội dung đoạn văn, không tiêu đề, không giải thích thêm.")
    body = {"model": TEACHER, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "think": False, "options": {"temperature": 0.5, "num_predict": num_predict}}
    r = urllib.request.urlopen(urllib.request.Request(
        OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=180)
    return json.load(r).get("message", {}).get("content", "").strip()


def extract_topics(n):
    """Chủ đề = cụm trước ' là ' trong lead (hoặc 2-5 từ đầu)."""
    topics = []
    seen = set()
    for line in open(VIWIKI):
        try:
            t = json.loads(line)
        except Exception:
            continue
        if not isinstance(t, str) or len(t) < 60:
            continue
        head = t[:120]
        m = re.split(r"\s+là\s+|\s+hay\s+|\s+\(|,", head, maxsplit=1)[0].strip()
        w = m.split()
        if 1 <= len(w) <= 6 and m.lower() not in seen and len(m) > 2:
            seen.add(m.lower()); topics.append(m)
        if len(topics) >= n * 3:
            break
    random.Random(0).shuffle(topics)
    return topics[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/distill/gemma_paras.jsonl")
    a = ap.parse_args()
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    T0 = time.time()
    topics = extract_topics(a.n)
    print(f"[{time.time()-T0:.0f}s] {len(topics)} chủ đề; bắt đầu sinh…", flush=True)
    f = open(a.out, "w"); ok = 0
    for i, tp in enumerate(topics):
        try:
            txt = gen(tp)
        except Exception as e:
            txt = ""
        if txt and len(txt) > 40:
            f.write(json.dumps({"topic": tp, "text": txt}, ensure_ascii=False) + "\n"); f.flush(); ok += 1
        if i % 50 == 0:
            print(f"[{time.time()-T0:.0f}s] {i}/{len(topics)} · ok {ok} · vd: {txt[:70]}", flush=True)
    f.close()
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}: {ok} đoạn", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
