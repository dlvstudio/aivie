"""Dữ-liệu CẦU NỐI cho expert lập-trình: bài code (đề + lời giải-thích) dịch sang TIẾNG VIỆT, CODE GIỮ NGUYÊN.

Vấn-đề: expert học bài code đề tiếng Anh → hỏi tiếng Việt ("Cho tôi đoạn code Python đệ quy") thì lạc đề / lặp; thân 1.3B
hiểu tiếng Việt nhưng chưa từng thấy "hỏi tiếng Việt → giải-thích tiếng Việt + code". Gemma dịch phần CHỮ, giữ nguyên
từng ký-tự trong khối ```code``` (kiểm lại: khối code sau dịch phải TRÙNG KHÍT bản gốc, sai → bỏ).

    python scripts/translate_code_vi.py --n 3000 --workers 16   → data/code_sft/vi_code.jsonl (khuôn messages)
"""
from __future__ import annotations
import argparse, json, os, random, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT = "data/code_sft/vi_code.jsonl"
CODE = re.compile(r"```[\w+-]*\n.*?```", re.S)
PROMPT = """Dịch sang tiếng Việt tự nhiên (giọng trợ lý lập trình thân thiện) phần CÂU HỎI và phần LỜI GIẢI dưới đây.
QUY TẮC BẮT BUỘC: giữ NGUYÊN các dấu giữ chỗ [[CODE_1]], [[CODE_2]]… (mỗi cái đúng một lần, đúng vị trí — đó là các khối
code sẽ được chèn lại), giữ nguyên tên hàm/biến/thư viện và mọi thứ nằm trong dấu `...`; chỉ dịch phần chữ.

CÂU HỎI:
{q}

LỜI GIẢI:
{a}

Trả về JSON: {{"cau_hoi": "...", "loi_giai": "..."}}"""


def gemma(prompt):
    body = {"model": "gemma4:26b-a4b-it-qat", "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0.2, "num_ctx": 4096}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]["content"]


def work(r):
    """KHÔNG gửi code cho Gemma: thay khối code bằng [[CODE_i]] → Gemma chỉ dịch chữ (ngắn hơn ~3 lần) → gắn code gốc lại."""
    blocks = CODE.findall(r["a"])
    masked, k = r["a"], 0
    for b in blocks:
        k += 1
        masked = masked.replace(b, f"[[CODE_{k}]]", 1)
    qb = CODE.findall(r["q"])
    q_masked = r["q"]
    for i, b in enumerate(qb, 1):
        q_masked = q_masked.replace(b, f"[[QCODE_{i}]]", 1)
    try:
        j = json.loads(gemma(PROMPT.format(q=q_masked, a=masked)))
        q, a = str(j.get("cau_hoi", "")).strip(), str(j.get("loi_giai", "")).strip()
    except Exception:
        return None
    if not q or not a:
        return None
    for i, b in enumerate(blocks, 1):                                   # mỗi dấu giữ chỗ đúng 1 lần → gắn code lại
        if a.count(f"[[CODE_{i}]]") != 1:
            return None
        a = a.replace(f"[[CODE_{i}]]", b)
    for i, b in enumerate(qb, 1):
        if q.count(f"[[QCODE_{i}]]") != 1:
            return None
        q = q.replace(f"[[QCODE_{i}]]", b)
    if CODE.findall(a) != blocks or "[[" in a or "[[" in q:
        return None
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "src": "vi_code"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--skip", type=int, default=0, help="vị trí trong pool đã xử lý tới (đạt + loại) — tránh dịch lại bài đã loại khi tiếp tục")
    a = ap.parse_args()
    from datasets import load_dataset
    pool = []
    for r in load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train"):
        q, ans = r["query"].strip(), r["answer"].strip()
        blocks = CODE.findall(ans)
        prose = CODE.sub("", ans)
        if r["lang"] == "python" and blocks and len(ans) < 2500 and len(q) < 800 and len(prose.split()) >= 25:
            pool.append({"q": q, "a": ans})
    random.Random(7).shuffle(pool)
    pool = pool[: int(a.n * 1.25)]                                      # dư 25% cho bài bị loại
    print(f"chọn {len(pool)} bài Python có lời giải-thích → dịch", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    T0, bad = time.time(), 0
    done = sum(1 for _ in open(OUT, encoding="utf-8")) if os.path.exists(OUT) else 0   # tiếp-tục: bỏ qua phần đã dịch
    pool = pool[a.skip or done:]
    ok = done
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(work, pool), 1):
            if res is None:
                bad += 1
            else:
                ok += 1
                f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if k % 100 == 0:
                el = time.time() - T0
                print(f"[{el:.0f}s] {k}/{len(pool)} · đạt {ok} · loại {bad} · {k/el*60:.0f}/phút", flush=True)
            if ok >= a.n:
                break
    print(f"### xong · đạt {ok} · loại {bad}", flush=True)


if __name__ == "__main__":
    main()
