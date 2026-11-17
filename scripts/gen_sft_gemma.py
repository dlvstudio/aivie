"""Sequence-KD: câu hỏi có sẵn → Gemma viết Đáp: → JSONL messages (train_hybrid_sft.py).

Smoke:
    .venv/bin/python scripts/gen_sft_gemma.py --n 8 --out data/sft/gemma_rewrites_smoke.jsonl

Lô thật (song song pretrain):
    .venv/bin/python scripts/gen_sft_gemma.py --n 5000 --out data/sft/gemma_rewrites.jsonl
"""
import json, time, argparse, urllib.request, random, os, re

OLLAMA = "http://127.0.0.1:11434/api/chat"
TEACHER = "gemma4:26b-a4b-it-qat"
SYS = (
    "Bạn là trợ lý tiếng Việt. Trả lời chính xác, mạch lạc, đúng trọng tâm. "
    "Không nhắc lại câu hỏi, không tiêu đề. Không dùng tiếng Anh trừ tên riêng hoặc code."
)
SKIP_RE = re.compile(
    r"spotify|hình ảnh|chia sẻ nó|youtube|facebook|in ra màn hình máy in",
    re.I,
)
FACT_SEEDS = [
    "Thủ đô của Việt Nam là thành phố nào?",
    "Ai là tác giả Truyện Kiều?",
    "Sông Hồng hay Mê Kông dài hơn?",
    "Nước sôi ở nhiệt độ bao nhiêu độ C (mực nước biển)?",
    "Một năm nhuận có bao nhiêu ngày?",
    "Tính 2^6. Viết ngắn, có phép tính.",
]


def strip_hoi(p):
    s = (p or "").strip()
    s = re.sub(r"^Hỏi:\s*", "", s)
    return s.strip()


def chat(q, num_predict=400):
    body = {
        "model": TEACHER,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": q},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.4, "num_predict": num_predict},
    }
    r = urllib.request.urlopen(
        urllib.request.Request(
            OLLAMA, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=180,
    )
    msg = json.load(r).get("message", {}) or {}
    txt = (msg.get("content") or "").strip()
    txt = re.sub(r"^Đáp:\s*", "", txt)
    return txt


def load_seeds(path, n, rng, max_q=400):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        q = strip_hoi(r.get("prompt") or "")
        if not q or len(q) < 12 or len(q) > max_q:
            continue
        if SKIP_RE.search(q):
            continue
        rows.append({"q": q, "orig": r.get("answer") or "", "seed_source": r.get("source") or path})
    rng.shuffle(rows)
    return rows[:n]


def ok_ans(a, min_len=6):
    if not a or len(a) < min_len:
        return False
    # quá nhiều ASCII → nghi lẫn Anh
    letters = [c for c in a if c.isalpha()]
    if letters:
        ascii_ratio = sum(c.isascii() for c in letters) / len(letters)
        if ascii_ratio > 0.85 and len(a) > 80:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/sft/sft_train.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="data/sft/gemma_rewrites_smoke.jsonl")
    ap.add_argument("--facts", action="store_true", help="thêm vài câu fact sạch (dùng lúc smoke)")
    ap.add_argument("--num-predict", type=int, default=400)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rng = random.Random(0)
    T0 = time.time()
    if a.facts:
        facts = [{"q": q, "orig": "", "seed_source": "fact"} for q in FACT_SEEDS[: a.n]]
        seeds = facts + load_seeds(a.seeds, max(0, a.n - len(facts)), rng)
    else:
        seeds = load_seeds(a.seeds, a.n, rng)

    print(f"[{time.time()-T0:.0f}s] {len(seeds)} câu · teacher={TEACHER}", flush=True)
    fp = open(a.out, "w", encoding="utf-8")
    ok = fail = 0
    for i, s in enumerate(seeds):
        t1 = time.time()
        try:
            ans = chat(s["q"], a.num_predict)
        except Exception as e:
            ans, err = "", str(e)
            print(f"  FAIL {i}: {err[:120]}", flush=True)
        dt = time.time() - t1
        good = ok_ans(ans)
        rec = {
            "source": "gemma4-26b",
            "seed_source": s["seed_source"],
            "n_turns": 1,
            "sec": round(dt, 2),
            "messages": [
                {"role": "user", "content": s["q"]},
                {"role": "assistant", "content": ans},
            ],
        }
        if s["orig"]:
            rec["orig_answer"] = s["orig"]
        if good:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n"); fp.flush(); ok += 1
        else:
            fail += 1
        preview = (ans or "")[:90].replace("\n", " ")
        print(f"[{time.time()-T0:.0f}s] {i+1}/{len(seeds)} {'ok' if good else 'drop'} {dt:.1f}s  Q: {s['q'][:50]}", flush=True)
        print(f"         A: {preview}", flush=True)
    fp.close()
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}: ok {ok} · drop {fail}", flush=True)


if __name__ == "__main__":
    main()
