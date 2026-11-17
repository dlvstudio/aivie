"""TK_104 §16 · VIẾT LẠI câu trả lời quá ngắn cho đầy đủ hơn (ý người dùng: 1.3B trả lời kiến thức chung thường chỉ 1 câu).

Nguyên nhân: dữ liệu hiệu chỉnh/hỏi–đáp cũ yêu cầu "1–2 câu". Thầy (Gemma-4-31B) viết lại các câu trả lời ngắn: GIỮ NGUYÊN ý đúng,
câu đầu trả lời thẳng, thêm giải thích / ví dụ (2–5 câu), khoảng một nửa kết bằng câu hỏi mở gợi chuyện tiếp. Không thêm điều
không chắc chắn. Câu từ chối (thực thể bịa) giữ nguyên.

    python scripts/expand_answers.py --inp data/calib/train.jsonl --only-src calib_dap --workers 48 --ollama http://127.0.0.1:8000/v1
    (ghi đè --inp, bản cũ lưu <inp>.short.jsonl)
"""
from __future__ import annotations
import argparse, json, os, random, shutil, sys
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_office import ollama, CJK

P = """Câu hỏi của người dùng: {q}
Câu trả lời hiện tại (ĐÚNG nhưng quá ngắn): {a}

Viết lại câu trả lời cho đầy đủ, tự nhiên như một trợ lý tận tình người Việt:
- Câu đầu tiên trả lời THẲNG vào câu hỏi (giữ nguyên ý đúng ở trên).
- Sau đó giải thích thêm, bối cảnh hoặc ví dụ ngắn — tổng 2–5 câu; CHỈ thêm điều chắc chắn đúng, không bịa số liệu.
- {end}
- Không lặp lại câu hỏi, không mở đầu kiểu "Câu hỏi hay!", không dùng LaTeX.
Chỉ viết câu trả lời mới, không thêm gì khác."""
ENDS = ["Kết bằng MỘT câu hỏi mở tự nhiên để gợi chuyện tiếp (vd: \"Bạn có muốn biết thêm về … không?\").",
        "Không cần câu hỏi ở cuối."]


def work(args):
    r, url, seed = args
    m = r["messages"]
    q, a = m[-2]["content"], m[-1]["content"]
    end = ENDS[0] if random.Random(seed).random() < 0.5 else ENDS[1]
    try:
        new = ollama(P.format(q=q, a=a, end=end), url, temp=0.6, num_predict=600).strip()
        if len(new) < len(a) or len(new) > 2000 or CJK.search(new):
            return r
        return {**r, "messages": m[:-1] + [{"role": "assistant", "content": new}], "mo_rong": True}
    except Exception:
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--only-src", default="")
    ap.add_argument("--max-len", type=int, default=220, help="chỉ viết lại câu trả lời ngắn hơn số ký tự này")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--ollama", default="http://127.0.0.1:8000/v1")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    if not os.path.exists(a.inp + ".short.jsonl"):
        shutil.copy(a.inp, a.inp + ".short.jsonl")
    idx = [i for i, r in enumerate(rows) if (not a.only_src or r.get("src") == a.only_src)
           and len(r["messages"][-1]["content"]) < a.max_len and not r.get("mo_rong")]
    print(f"viết lại {len(idx)}/{len(rows)} câu trả lời ngắn", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, new in zip(idx, ex.map(work, [(rows[i], a.ollama, i) for i in idx])):
            rows[i] = new
    with open(a.inp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = sum(1 for i in idx if rows[i].get("mo_rong"))
    print(f"### xong · đã mở rộng {done}/{len(idx)}")
    for i in idx[:2]:
        print("H:", rows[i]["messages"][-2]["content"], "\nĐ:", rows[i]["messages"][-1]["content"][:500], "\n---")


if __name__ == "__main__":
    main()
