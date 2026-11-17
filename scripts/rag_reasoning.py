"""§TK_076 — DEMO: reasoning-model + RAG trên M4 (không cần GPU khủng).

Kết hợp 3 đột phá chạy-local:
  (1) test-time compute: gemma4:26b THINK (suy-luận dài) — model nhỏ nghĩ lâu.
  (2) RAG: BM25 truy đoạn Wikipedia THẬT (viwiki) — grounding của ta.
  (3) verify: neo câu trả lời vào nguồn.

So 2 cột: [A] single-shot (think=false, KHÔNG RAG) vs [C] reasoning+RAG.

    PYTHONPATH=src python scripts/rag_reasoning.py
"""
import sys, time, json, urllib.request
sys.path.insert(0, "scripts")
from rag_seallm import load_corpus  # tái dùng BM25 + corpus

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:26b-a4b-it-qat"


def ask(messages, think, num_predict=1200, temperature=0.3):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "think": think, "options": {"temperature": temperature, "num_predict": num_predict}}
    r = urllib.request.urlopen(urllib.request.Request(
        OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=300)
    m = json.load(r).get("message", {})
    return m.get("content", "").strip(), str(m.get("thinking", "")).strip()


def main():
    bm, snippets = load_corpus(285000)
    QS = [
        "Sông nào dài hơn: sông Hồng hay sông Mê Kông? Giải thích ngắn.",
        "Tại sao lá cây có màu xanh lục?",
        "Nếu một tế bào vi khuẩn phân chia mỗi 20 phút, sau 2 giờ từ 1 tế bào sẽ có bao nhiêu tế bào?",
        "Hà Nội và Thành phố Hồ Chí Minh, nơi nào đông dân hơn và vì sao?",
    ]
    for q in QS:
        print(f"\n{'='*70}\n❓ {q}", flush=True)
        # [A] single-shot: không RAG, không suy-luận
        t = time.time()
        a, _ = ask([{"role": "user", "content": q + " Trả lời ngắn gọn."}], think=False, num_predict=200)
        print(f"\n[A single-shot]  ({time.time()-t:.0f}s)\n  {a}", flush=True)
        # [C] reasoning + RAG: truy nguồn thật + để model SUY-LUẬN
        ids = bm.top(q, k=3)
        ctx = "\n\n".join(f"[Nguồn {j+1}] {snippets[i]}" for j, i in enumerate(ids))
        sys_msg = ("Bạn là trợ lý tiếng Việt. Suy luận cẩn thận dựa trên các NGUỒN được cung cấp "
                   "và kiến thức của bạn, rồi trả lời ngắn gọn, chính xác. Nếu nguồn liên quan, ưu tiên nó.")
        t = time.time()
        c, think = ask([{"role": "system", "content": sys_msg},
                        {"role": "user", "content": f"{ctx}\n\nCâu hỏi: {q}"}], think=True, num_predict=1400)
        print(f"\n[C reasoning+RAG]  ({time.time()-t:.0f}s, nghĩ {len(think.split())} từ)\n  {c}", flush=True)
        print(f"  ↳ nguồn: {snippets[ids[0]][:80]}…", flush=True)


if __name__ == "__main__":
    main()
