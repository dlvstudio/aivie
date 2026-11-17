"""§TK_073 — RAG + SeaLLM: kiến trúc lai (grounding của ta + LLM hiểu-thật).

Đường: prompt → [RETRIEVE] BM25 truy đoạn Wikipedia THẬT (viwiki) → [GENERATE]
nạp đoạn cho SeaLLM-7B (Ollama) sinh có căn cứ → [VERIFY] đối chiếu câu trả lời
với đoạn nguồn (đánh dấu phần KHÔNG có trong nguồn = nghi ảo giác).

Hệ 54M của ta (W_skip/kNN) hay-đúng-nhưng-chép; SeaLLM hiểu-nhưng-hay-bịa. RAG nối
2 thế mạnh: kho factual tiếng Việt của ta neo SeaLLM khỏi bịa, SeaLLM lo hiểu+mượt.

    PYTHONPATH=src python scripts/rag_seallm.py --demo
    PYTHONPATH=src python scripts/rag_seallm.py --q "chó là gì"
"""
import sys, time, json, re, math, argparse, urllib.request
from collections import defaultdict, Counter

VIWIKI = "data/text_corpus/viwiki.jsonl"
OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "hf.co/SeaLLMs/SeaLLM-7B-v2.5-GGUF:Q4_K_M"
_TOK = re.compile(r"\w+", re.UNICODE)


def tok(s):
    return _TOK.findall(s.lower())


class BM25:
    """BM25 Okapi thuần Python trên lead (định nghĩa đầu bài)."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.N = len(docs_tokens)
        self.len = [len(d) for d in docs_tokens]
        self.avg = sum(self.len) / max(self.N, 1)
        self.tf = []            # per-doc term->count
        df = defaultdict(int)
        self.inv = defaultdict(list)   # term -> [doc_id]
        for i, d in enumerate(docs_tokens):
            c = Counter(d); self.tf.append(c)
            for t in c:
                df[t] += 1; self.inv[t].append(i)
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def top(self, query, k=3):
        q = tok(query)
        scores = defaultdict(float)
        for t in q:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i in self.inv[t]:
                tf = self.tf[i][t]
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return sorted(scores, key=scores.get, reverse=True)[:k]


def load_corpus(max_docs, lead_chars=200, snippet_chars=600):
    leads = []; snippets = []; T0 = time.time()
    for i, line in enumerate(open(VIWIKI)):
        if i >= max_docs:
            break
        try:
            t = json.loads(line)
        except Exception:
            continue
        if not isinstance(t, str) or len(t) < 40:
            continue
        leads.append(tok(t[:lead_chars]))
        snippets.append(t[:snippet_chars].replace("\n", " "))
        if i % 50000 == 0:
            print(f"[{time.time()-T0:.0f}s] nạp {i} bài…", flush=True)
    print(f"[{time.time()-T0:.0f}s] {len(snippets)} bài; dựng BM25…", flush=True)
    bm = BM25(leads)
    print(f"[{time.time()-T0:.0f}s] BM25 sẵn sàng", flush=True)
    return bm, snippets


def seallm(prompt, num_predict=180, temperature=0.2):
    body = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict}}
    r = urllib.request.urlopen(urllib.request.Request(
        OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=180)
    return json.load(r)["response"].strip()


def rag_answer(bm, snippets, query, k=3, verify=True):
    ids = bm.top(query, k=k)
    ctx = "\n\n".join(f"[Nguồn {j+1}] {snippets[i]}" for j, i in enumerate(ids))
    prompt = (
        "Bạn là trợ lý tiếng Việt. CHỈ dựa vào các nguồn dưới đây để trả lời, "
        "ngắn gọn 1-3 câu. Nếu nguồn không có thông tin, nói 'Không tìm thấy trong nguồn.'\n\n"
        f"{ctx}\n\nCâu hỏi: {query}\nTrả lời:")
    ans = seallm(prompt)
    result = {"query": query, "answer": ans, "sources": [snippets[i][:120] for i in ids]}
    if verify:
        # kiểm thô: từ nội dung (>3 ký tự) trong câu trả lời có xuất hiện ở nguồn không
        src_tokens = set(tok(" ".join(snippets[i] for i in ids)))
        ans_content = [w for w in tok(ans) if len(w) > 3]
        unsup = [w for w in ans_content if w not in src_tokens]
        result["grounded_ratio"] = round(1 - len(unsup) / max(len(ans_content), 1), 2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--max-docs", type=int, default=285000)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    bm, snippets = load_corpus(a.max_docs)
    qs = ([a.q] if a.q else
          ["chó là gì", "Hà Nội là gì", "internet là gì", "tế bào là gì",
           "biến đổi khí hậu là gì", "trí tuệ nhân tạo là gì", "mặt trời là gì",
           "sông Hồng ở đâu"] if a.demo else [])
    for q in qs:
        t = time.time()
        r = rag_answer(bm, snippets, q, k=a.k)
        print(f"\n### «{q}»  ({time.time()-t:.1f}s, grounded={r.get('grounded_ratio')})", flush=True)
        print(f"  → {r['answer']}", flush=True)
        print(f"  nguồn: {r['sources'][0][:90]}…", flush=True)


if __name__ == "__main__":
    main()
