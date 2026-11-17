"""§TK_103 · RAG cho HybridLM 1.3B của TA (thay QLoRA-3B trong assistant_ours.py).

Đường đi (tái dùng tầng truy-hồi model-agnostic):
  [RETRIEVE] BM25F viwiki (retrieval_v2)  →  [FOCUS] cửa-sổ idf-weighted
  →  [GROUND] dán nguồn vào khuôn SFT 'Hỏi: …\\nĐáp: '
  →  [GENERATE] HybridLM 1.3B SFT (MPS, dừng <EOS>)  →  [VERIFY] grounded_ratio

Vì sao: VMLU cho thấy 1.3B gần-ngẫu-nhiên về TRI-THỨC (bịa) nhưng DIỄN-ĐẠT tiếng Việt
tốt. RAG nạp sự-thật từ ngoài → model chỉ cần trích/diễn-đạt (việc nó làm được).

    PYTHONPATH=scripts:src .venv/bin/python scripts/rag_hybrid.py --demo
    PYTHONPATH=scripts:src .venv/bin/python scripts/rag_hybrid.py \\
        --ckpt checkpoints/hf_hybrid/hybrid_1p3b_assistant_v5.pt \\
        --q "Sao Hỏa có mấy vệ tinh?" --k 3
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np, torch

sys.path.insert(0, "scripts")
from retrieval_v2 import load_corpus_v2, tok, STOP, extract_title
from train_hybrid_scale import HybridLM, HAVE_KERNEL
from nn_aivie.core.tokenizer import BPETokenizer

V = 32002


def load_model(ckpt, dev):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    c, state, step = obj["cfg"], obj["state"], obj.get("step")
    del obj
    ae = c.get("attn_every", 0)
    attn = list(range(ae - 1, c["layers"], ae)) if ae and ae > 0 else []
    m = HybridLM(V, c["d"], c["layers"], attn, c.get("d_state", 16),
                 c.get("heads", 16), c.get("kv_heads"))
    m.load_state_dict(state); del state
    m.eval().to(dev)
    return m, c, step


@torch.no_grad()
def generate(m, tok, dev, question, n=200, temp=0.3, top_p=0.9, rep_pen=1.15,
             recent=64, ctx=1024, eos=None):
    """Khuôn SFT: 'Hỏi: {question}\\n' + 'Đáp: ' (mỗi đoạn encode riêng như build_example)."""
    ids = tok.encode("Hỏi: " + question.strip() + "\n") + tok.encode("Đáp: ")
    n0 = len(ids)
    rng = np.random.default_rng(0)
    for _ in range(n):
        x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=dev)
        lo = m(x)[0, -1].float().cpu().numpy().astype(np.float64)
        for t in set(ids[-recent:]):
            if 0 <= t < V:
                lo[t] -= rep_pen
        lo = lo / max(temp, 1e-6)
        lo -= lo.max()
        p = np.exp(lo); p /= p.sum()
        order = np.argsort(p)[::-1]
        cdf = np.cumsum(p[order])
        keep = order[: max(1, int(np.searchsorted(cdf, top_p)) + 1)]
        pk = p[keep] / p[keep].sum()
        nxt = int(rng.choice(keep, p=pk))
        if eos is not None and nxt == eos:
            break
        ids.append(nxt)
    return tok.decode(ids[n0:]).strip()


class RagHybrid:
    def __init__(self, ckpt, max_docs=285000, dev=None):
        self.dev = dev or ("mps" if torch.backends.mps.is_available() else "cpu")
        t = time.time()
        self.bm, self.snips = load_corpus_v2(max_docs)
        print(f"[+{time.time()-t:.0f}s] BM25F viwiki nạp ({len(self.snips)} bài)", flush=True)
        t = time.time()
        self.m, self.cfg, self.step = load_model(ckpt, self.dev)
        self.tok = BPETokenizer.load("checkpoints/bpe_v1.json")
        self.eos = self.tok.token_to_id["<EOS>"]
        npar = sum(p.numel() for p in self.m.parameters())
        print(f"[+{time.time()-t:.0f}s] HybridLM SFT step={self.step} {npar/1e9:.3f}B "
              f"trên {self.dev} · kernel={HAVE_KERNEL}\n", flush=True)

    def _focus(self, text, query, win=700, step=140):
        """Cửa-sổ ~win ký-tự có mật-độ từ-query cao nhất (idf/(1+count) — từ hiếm định-vị tốt)."""
        qset = {t for t in tok(query) if t not in STOP and len(t) >= 2}
        if not qset or len(text) <= win:
            return text[:win]
        low = text.lower()
        wt = {t: self.bm.idf.get(t, 0.0) / (1 + low.count(t)) for t in qset}
        best_s, best_sc = 0, -1.0
        for s in range(0, len(low) - win + 1, step):
            w = low[s:s + win]
            sc = sum(wt[t] for t in qset if t in w)
            if sc > best_sc:
                best_sc, best_s = sc, s
        snip = text[best_s:best_s + win]
        return snip if best_s == 0 else "…" + snip

    def answer(self, query, k=3, max_tokens=200):
        ids = self.bm.top(query, k=k)
        ctx = "\n".join(f"[Nguồn {j+1} — {extract_title(self.snips[i])}] {self._focus(self.snips[i], query)}"
                        for j, i in enumerate(ids))
        # nhồi ngữ-cảnh + câu hỏi vào phần "Hỏi:" của khuôn SFT
        q = (f"Dựa vào các nguồn dưới đây, trả lời ngắn gọn bằng tiếng Việt. "
             f"Nếu nguồn không có, nói không tìm thấy.\n{ctx}\nCâu hỏi: {query}")
        t = time.time()
        ans = generate(self.m, self.tok, self.dev, q, n=max_tokens, eos=self.eos)
        # verify: từ nội-dung (>3 ký-tự) trong câu trả lời có trong nguồn không
        src = set(tok(" ".join(self.snips[i] for i in ids)))
        content = [w for w in tok(ans) if len(w) > 3]
        unsup = [w for w in content if w not in src]
        gr = round(1 - len(unsup) / max(len(content), 1), 2)
        return {"answer": ans, "grounded": gr, "sec": round(time.time() - t, 1),
                "src": [self.snips[i][:90] for i in ids]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf_hybrid/hybrid_1p3b_assistant_v5.pt")
    ap.add_argument("--q", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--max-docs", type=int, default=285000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    r = RagHybrid(a.ckpt, a.max_docs)
    qs = ([a.q] if a.q else
          ["Sao Hỏa có mấy vệ tinh?",
           "Biến đổi khí hậu là gì?",
           "Sông Mê Kông chảy qua những nước nào?",
           "Trí tuệ nhân tạo là gì?"] if a.demo else [])
    for q in qs:
        res = r.answer(q, k=a.k, max_tokens=a.max_tokens)
        print(f"{'='*72}\n❓ {q}   ({res['sec']}s · grounded={res['grounded']})\n{'-'*72}", flush=True)
        print(f"💬 {res['answer']}", flush=True)
        print(f"📚 {res['src'][0]}…\n", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
