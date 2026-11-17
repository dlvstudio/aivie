"""§TK_103 · PILOT PHƯƠNG ÁN B — sinh dữ liệu chưng-cất cho Zamba2 (thầy = Gemma 4 26B, Ollama).

Mục-tiêu B2: hỏi tiếng Việt → Zamba2 viết nháp tiếng Anh <en>…</en> rồi bản tiếng Việt <vi>…</vi>
trong MỘT lượt. Mỗi lời gọi Gemma trả 3 phần: câu hỏi dịch sang tiếng Anh, trả lời tiếng Anh,
trả lời tiếng Việt → dựng được cả mẫu B2 lẫn mẫu ôn-tập tiếng Anh (chống quên).

Vì sao thầy là Gemma chứ không phải Zamba2: đo trên M4, Zamba2 sinh 284 câu/giờ (lô 8 còn chậm
hơn: 94 câu/giờ — Mamba2 fallback không tận-dụng được lô) → 2K mẫu ~9 giờ. Gemma ~69 tok/s.

Câu hỏi: lấy từ data/sft_vi/sft_mix.jsonl (hội-thoại 1 lượt), chia đều theo nguồn; giữ riêng
một tập held-out để đánh-giá (KHÔNG đưa vào train). Ghi từng mẫu ra JSONL → chạy lại tự nối tiếp.

    .venv/bin/python scripts/gen_pilot_b.py --n 2000 --heldout 60
"""
from __future__ import annotations
import argparse, collections, json, os, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

OLLAMA = "http://127.0.0.1:11434"
GEMMA = "gemma4:26b-a4b-it-qat"
SRC = "data/sft_vi/sft_mix.jsonl"
OUT = "data/pilot_b"

PROMPT = """You are a helpful, accurate assistant. A user asked the question below (usually in Vietnamese).

Reply with EXACTLY these three sections and nothing else:
<q_en>the user's question translated into English</q_en>
<en>your answer in English</en>
<vi>the same answer in natural, fluent Vietnamese, as a native Vietnamese speaker would write it</vi>

Rules: keep answers concise (under 200 words); keep code blocks and `inline code` identical in both
answers; if you are not sure about a fact, say so instead of guessing.

Question:
{q}"""

_TAG = {k: re.compile(rf"<{k}>(.*?)</{k}>", re.S) for k in ("q_en", "en", "vi")}


def gemma(prompt):
    body = {"model": GEMMA, "stream": False, "think": False, "options": {"temperature": 0.2},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return r["message"]["content"], r.get("eval_count", 0)


def pick_questions(n, heldout, seed=0):
    """Câu hỏi 1 lượt, dài 15–600 ký-tự, chia đều theo nguồn (round-robin)."""
    by_src = collections.defaultdict(list)
    for line in open(SRC, encoding="utf-8"):
        r = json.loads(line)
        msgs = r["messages"]
        if len(msgs) != 2 or msgs[0]["role"] != "user":
            continue
        q = msgs[0]["content"].strip()
        if 15 <= len(q) <= 600 and r.get("source") != "gold-khuon":
            by_src[r["source"]].append({"id": r["id"], "source": r["source"],
                                        "domain": r.get("domain"), "q": q})
    rng = random.Random(seed)
    for v in by_src.values():
        rng.shuffle(v)
    order, pools = [], list(by_src.values())
    while len(order) < n + heldout and any(pools):
        for p in pools:
            if p:
                order.append(p.pop())
    rng.shuffle(order)
    return order[heldout:heldout + n], order[:heldout]


def work(item):
    t = time.time()
    try:
        txt, ntok = gemma(PROMPT.format(q=item["q"]))
    except Exception as e:
        return {**item, "error": f"{type(e).__name__}: {e}"}
    parts = {k: (m.group(1).strip() if (m := rx.search(txt)) else "") for k, rx in _TAG.items()}
    if not all(parts.values()):
        return {**item, "error": "thiếu thẻ", "raw": txt[:500]}
    return {**item, **parts, "tok": ntok, "sec": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--heldout", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N mẫu (đo tốc-độ)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    train, held = pick_questions(a.n, a.heldout)
    with open(f"{OUT}/heldout.jsonl", "w", encoding="utf-8") as f:
        for it in held:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    path = f"{OUT}/distill.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")}
    todo = [it for it in train if it["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"train {len(train)} · held-out {len(held)} · đã có {len(done)} · cần sinh {len(todo)} "
          f"· {a.workers} luồng", flush=True)
    T0, ok, err, toks = time.time(), 0, 0, 0
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(work, todo), 1):
            if "error" in res:
                err += 1
                print(f"  ⚠️ {res['id']}: {res['error']}", flush=True)
                continue
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            ok += 1; toks += res["tok"]
            if i % 20 == 0 or i == len(todo):
                el = time.time() - T0
                print(f"[{el:.0f}s] {i}/{len(todo)} · ok {ok} · lỗi {err} · {3600*i/el:.0f} mẫu/giờ "
                      f"· {toks/el:.0f} tok/s · còn ~{(len(todo)-i)*el/i/60:.0f} phút", flush=True)
    print(f"### xong: ok {ok} · lỗi {err} · {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
