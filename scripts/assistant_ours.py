"""§TK_078 — TRỢ LÝ HOÀN CHỈNH DO TA LÀM CHỦ (không mượn LLM ngoài).

Đường đi:
  [RETRIEVE] BM25 của ta trên viwiki  →  [GROUND] dán đoạn thật vào prompt
  →  [GENERATE] C = QLoRA Qwen-3B (ta fine-tune)  →  [VERIFY] grounded_ratio

Cả kho tri thức + truy-xuất (của ta) lẫn model sinh chữ (ta fine-tune) đều là của ta.
Khắc phục đúng điểm yếu đã đo: C hỏi-chay thì bịa; có ngữ-cảnh thì trích trung thành.

    .venv/bin/python scripts/assistant_ours.py --demo
    .venv/bin/python scripts/assistant_ours.py --q "Sao Hỏa có mấy vệ tinh?"
"""
import sys, time, argparse
sys.path.insert(0, "scripts")
from retrieval_v2 import load_corpus_v2, tok, STOP, extract_title  # BM25F (title+body) của ta — v2
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

BASE = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER = "adapters/vi_qlora_div400"
SAMPLER = make_sampler(temp=0.2, top_p=0.9)   # QA cần ổn định (bớt dao động fact)
LOGITS = make_logits_processors(repetition_penalty=1.2, repetition_context_size=40)


class Assistant:
    def __init__(self, max_docs=285000):
        self.bm, self.snips = load_corpus_v2(max_docs)
        t = time.time()
        self.model, self.tok = load(BASE, adapter_path=ADAPTER)
        print(f"[+{time.time()-t:.0f}s] nạp C (QLoRA-3B) xong\n", flush=True)

    def _focus(self, text, query, win=900, step=150):
        """§TK_090 — CỬA-SỔ-TẬP-TRUNG: chọn đoạn ~win ký tự có mật-độ-từ-query cao
        nhất (idf-weighted) thay vì cả 3500c → LLM thấy fact SÂU (vd 'thủ đô Tokyo'
        ký tự 3302) mà ngữ cảnh vẫn gọn/nhanh. Hòa → ưu tiên cửa sổ đầu (lead)."""
        qset = {t for t in tok(query) if t not in STOP and len(t) >= 2}
        if not qset or len(text) <= win:
            return text[:win]
        low = text.lower()
        # cân theo ĐỘ-HIẾM-TRONG-BÀI: từ thực thể (xuất hiện khắp bài) kém định vị;
        # từ thuộc-tính hiếm (vd 'thủ đô' 1 lần) mới trỏ tới câu trả lời.
        wt = {t: self.bm.idf.get(t, 0.0) / (1 + low.count(t)) for t in qset}
        best_s, best_sc = 0, -1.0
        for s in range(0, len(low) - win + 1, step):
            w = low[s:s + win]
            sc = sum(wt[t] for t in qset if t in w)
            if sc > best_sc:
                best_sc, best_s = sc, s
        snip = text[best_s:best_s + win]
        return snip if best_s == 0 else "…" + snip

    def answer(self, query, k=3, max_tokens=220):
        ids = self.bm.top(query, k=k)
        # gắn TIÊU ĐỀ thực thể vào mỗi nguồn → LLM biết cửa-sổ nói về AI (khung ngữ cảnh)
        ctx = "\n\n".join(f"[Nguồn {j+1} — {extract_title(self.snips[i])}] {self._focus(self.snips[i], query)}"
                          for j, i in enumerate(ids))
        user = (f"Dựa DUY NHẤT vào các nguồn dưới đây, hãy trả lời câu hỏi bằng tiếng Việt: "
                f"chính xác, súc tích, và trả lời ĐẦY ĐỦ tất cả các phần được hỏi. "
                f"Nếu nguồn không có thông tin, nói 'Không tìm thấy trong nguồn.'\n\n"
                f"{ctx}\n\n[Câu hỏi] {query}")
        prompt = self.tok.apply_chat_template([{"role": "user", "content": user}],
                                              add_generation_prompt=True)
        ans = generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens,
                       verbose=False, sampler=SAMPLER, logits_processors=LOGITS).strip()
        # verify: từ nội dung (>3 ký tự) trong câu trả lời có trong nguồn không
        src = set(tok(" ".join(self.snips[i] for i in ids)))
        content = [w for w in tok(ans) if len(w) > 3]
        unsup = [w for w in content if w not in src]
        gr = round(1 - len(unsup) / max(len(content), 1), 2)
        return {"answer": ans, "grounded": gr,
                "src": [self.snips[i][:100] for i in ids]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--max-docs", type=int, default=285000)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    ast = Assistant(a.max_docs)
    qs = ([a.q] if a.q else
          ["Sao Hỏa có mấy vệ tinh và vì sao có màu đỏ?",
           "Chó là loài động vật như thế nào?",
           "Biến đổi khí hậu là gì?",
           "Trí tuệ nhân tạo là gì?",
           "Sông Mê Kông chảy qua những nước nào?",
           "Hà Nội là thành phố như thế nào?"] if a.demo else [])
    for q in qs:
        t = time.time()
        r = ast.answer(q, k=a.k)
        print(f"{'='*72}\n❓ {q}   ({time.time()-t:.1f}s · grounded={r['grounded']})\n{'-'*72}", flush=True)
        print(f"💬 {r['answer']}", flush=True)
        print(f"📚 nguồn: {r['src'][0]}…\n", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
