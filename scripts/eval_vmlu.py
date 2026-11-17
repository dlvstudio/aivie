"""§TK_100 · M0 — EVAL HARNESS VMLU (cổng go/no-go tiếng Việt).

VMLU = 10,880 MCQ / 58 môn (STEM·Nhân-văn·Xã-hội·Khác). `validation` CÓ đáp-án → chấm
accuracy; `test` ẩn đáp-án → xuất file nộp leaderboard. Nguồn HF: Hamana0509/VMLU
(schema: id "MÔN-XXXX", question, choices ["A. …"], answer "C").

Bốn backend:
  • dummy      — chọn A / ngẫu-nhiên → TEST đường-ống ngay (không cần model)
  • hf         — model transformers cục-bộ: sinh vài token → trích chữ-cái
  • openai     — endpoint tương-thích OpenAI (vLLM server) → cho scale
  • ours       — HybridLM 1.3B của ta (checkpoint {cfg,state}); chấm bằng
                 LOGLIKELIHOOD trên token chữ-cái " A".." E" (1 forward/câu, không sinh)

CHẤM checkpoint của TA (M4/MPS):
    PYTHONPATH=scripts:src .venv/bin/python scripts/eval_vmlu.py --backend ours \
        --ckpt checkpoints/hf_hybrid/hybrid_1p3b_assistant_v5.pt \
        --tokenizer checkpoints/bpe_v1.json --split validation --limit 1000

SMOKE (kiểm plumbing + chấm, không cần model):
    PYTHONPATH=src .venv/bin/python scripts/eval_vmlu.py --backend dummy --limit 200

CHẤM model cục-bộ (khi đã có checkpoint):
    PYTHONPATH=src .venv/bin/python scripts/eval_vmlu.py --backend hf \
        --model <path/hf> --split validation --n-shot 0

CHẤM qua vLLM + XUẤT NỘP:
    python scripts/eval_vmlu.py --backend openai --base-url http://localhost:8000/v1 \
        --model my-vi-llm --split test --out out/vmlu_submission.json
"""
import sys, os, re, json, time, argparse, random, collections
T0 = time.time()
LETTERS = "ABCDE"

INSTRUCT = ("Đây là câu hỏi trắc nghiệm. Hãy chọn MỘT đáp án đúng và chỉ trả lời bằng "
            "một chữ cái (A, B, C, D hoặc E).")


def q_block(row, with_answer=False):
    ch = "\n".join(row["choices"])
    s = f"Câu hỏi: {row['question']}\n{ch}\nĐáp án:"
    if with_answer:
        s += f" {row['answer']}"
    return s


def build_prompt(row, shots, ctx_text=None):
    parts = [INSTRUCT, ""]
    if ctx_text:
        parts.append("Thông tin tham khảo:")
        parts.append(ctx_text)
        parts.append("")
    for sh in shots:
        parts.append(q_block(sh, with_answer=True)); parts.append("")
    parts.append(q_block(row))
    return "\n".join(parts)


class RagCtx:
    """Nhồi ngữ-cảnh viwiki (BM25F) vào prompt MCQ → đo RAG có nâng accuracy không."""
    def __init__(self, k=3, max_docs=285000, win=500, step=120):
        import sys as _s
        _s.path.insert(0, "scripts")
        from retrieval_v2 import load_corpus_v2, tok, STOP
        self._tok, self._STOP = tok, STOP
        self.bm, self.snips = load_corpus_v2(max_docs)
        self.k, self.win, self.step = k, win, step
        print(f"[{time.time()-T0:.0f}s] RAG: BM25F viwiki {len(self.snips)} bài · k={k}", flush=True)

    def _focus(self, text, query):
        qset = {t for t in self._tok(query) if t not in self._STOP and len(t) >= 2}
        if not qset or len(text) <= self.win:
            return text[:self.win]
        low = text.lower()
        wt = {t: self.bm.idf.get(t, 0.0) / (1 + low.count(t)) for t in qset}
        best_s, best_sc = 0, -1.0
        for s in range(0, len(low) - self.win + 1, self.step):
            w = low[s:s + self.win]
            sc = sum(wt[t] for t in qset if t in w)
            if sc > best_sc:
                best_sc, best_s = sc, s
        return text[best_s:best_s + self.win]

    def context(self, question):
        ids = self.bm.top(question, k=self.k)
        return "\n".join(f"- {self._focus(self.snips[i], question)}" for i in ids)


def extract_letter(text, n_choices):
    valid = LETTERS[:n_choices]
    t = text.strip().upper()
    m = re.search(r"(?:ĐÁP\s*ÁN|ANSWER|CHỌN)[:\s]*([A-E])", t)
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.search(r"\b([A-E])\b", t)
    if m and m.group(1) in valid:
        return m.group(1)
    for c in t:
        if c in valid:
            return c
    return valid[0]                                       # fallback: không đoán được → chữ đầu


# ── BACKENDS ───────────────────────────────────────────────────────────────────────
def make_predictor(a):
    if a.backend == "dummy":
        rng = random.Random(0)
        return lambda prompt, n: (LETTERS[:n][0] if a.dummy_fixed else rng.choice(LETTERS[:n]))

    if a.backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        tk = AutoTokenizer.from_pretrained(a.model)
        md = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype="auto").to(dev).eval()
        print(f"[{time.time()-T0:.0f}s] hf model nạp trên {dev}", flush=True)

        def pred(prompt, n):
            msgs = [{"role": "user", "content": prompt}]
            try:
                ids = tk.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            except Exception:
                ids = tk(prompt, return_tensors="pt").input_ids
            ids = ids.to(dev)
            with torch.no_grad():
                out = md.generate(ids, max_new_tokens=8, do_sample=False,
                                  pad_token_id=tk.eos_token_id)
            return extract_letter(tk.decode(out[0, ids.shape[1]:], skip_special_tokens=True), n)
        return pred

    if a.backend == "openai":
        import urllib.request
        url = a.base_url.rstrip("/") + "/chat/completions"
        hdr = {"Content-Type": "application/json"}
        if a.api_key:
            hdr["Authorization"] = f"Bearer {a.api_key}"

        def pred(prompt, n):
            body = json.dumps({"model": a.model, "temperature": 0, "max_tokens": 8,
                               "messages": [{"role": "user", "content": prompt}]}).encode()
            req = urllib.request.Request(url, data=body, headers=hdr)
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return extract_letter(r["choices"][0]["message"]["content"], n)
        return pred

    if a.backend == "hf-ll":
        # Model HF bất-kỳ (vd Zamba2-2.7B-Instruct), chấm LOGLIKELIHOOD chữ-cái như backend `ours`:
        # prompt bọc chat-template → token kế-tiếp sau "assistant\n" là chữ-cái đáp-án (Zamba2: 1 token/chữ).
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if "zamba2" in a.model.lower():
            sys.path.insert(0, "scripts")
            from test_zamba2 import patch_zamba2_block_id
            patch_zamba2_block_id()                           # bug transformers 5.16 (TK_103 §13)
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dt = torch.bfloat16 if dev == "cuda" else (torch.float16 if dev == "mps" else torch.float32)
        tk = AutoTokenizer.from_pretrained(a.model)
        md = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
        print(f"[{time.time()-T0:.0f}s] hf-ll nạp {a.model} trên {dev} · loglik chữ-cái (chat-template)", flush=True)

        @torch.no_grad()
        def pred(prompt, n):
            pre = tk.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, tokenize=False)
            ids = tk(pre, add_special_tokens=False).input_ids
            cand = []
            for L in LETTERS[:n]:                             # token đầu tiên của chữ-cái sau tiền-tố
                full = tk(pre + L, add_special_tokens=False).input_ids
                if full[:len(ids)] != ids or len(full) != len(ids) + 1:
                    raise SystemExit(f"chữ '{L}' không phải đúng 1 token sau tiền-tố — đổi cách chấm")
                cand.append(full[-1])
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            lo = md(input_ids=x, use_cache=False).logits[0, -1].float()
            return LETTERS[int(lo[torch.tensor(cand, device=dev)].argmax())]
        return pred

    if a.backend == "ours":
        import torch
        sys.path.insert(0, "scripts")
        from train_hybrid_scale import HybridLM, HAVE_KERNEL
        from nn_aivie.core.tokenizer import BPETokenizer
        if not a.ckpt:
            raise SystemExit("--backend ours cần --ckpt <checkpoint .pt>")
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        obj = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        c, state, step = obj["cfg"], obj["state"], obj.get("step")
        del obj
        ae = c.get("attn_every", 0)
        attn = list(range(ae - 1, c["layers"], ae)) if ae and ae > 0 else []
        V = 32002
        md = HybridLM(V, c["d"], c["layers"], attn, c.get("d_state", 16),
                      c.get("heads", 16), c.get("kv_heads"))
        md.load_state_dict(state); del state
        md.eval().to(dev)
        tok = BPETokenizer.load(a.tokenizer)
        # token chữ-cái (có dấu cách phía trước — khớp cách nối sau "Đáp án:")
        letter_ids = [tok.encode(" " + L)[0] for L in LETTERS]
        maxctx = int(c.get("ctx", 2048))
        npar = sum(p.numel() for p in md.parameters())
        print(f"[{time.time()-T0:.0f}s] ours nạp trên {dev} · step={step} · {npar/1e9:.3f}B "
              f"· kernel={HAVE_KERNEL} · loglik chữ-cái", flush=True)

        @torch.no_grad()
        def pred(prompt, n):
            ids = tok.encode(prompt)[-maxctx:]
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            lo = md(x)[0, -1].float()                         # logits token kế-tiếp
            cand = torch.tensor(letter_ids[:n], device=dev)
            return LETTERS[int(lo[cand].argmax())]
        return pred

    raise SystemExit(f"backend lạ: {a.backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["dummy", "hf", "openai", "ours", "hf-ll"], default="dummy")
    ap.add_argument("--ckpt", default=None, help="checkpoint HybridLM cho backend=ours")
    ap.add_argument("--tokenizer", default="checkpoints/bpe_v1.json", help="BPE cho backend=ours")
    ap.add_argument("--rag", action="store_true", help="nhồi ngữ-cảnh viwiki (BM25F) vào prompt")
    ap.add_argument("--rag-k", type=int, default=3, help="số nguồn viwiki mỗi câu")
    ap.add_argument("--rag-docs", type=int, default=285000)
    ap.add_argument("--repo", default="Hamana0509/VMLU")
    ap.add_argument("--split", default="validation", help="validation=chấm · test=xuất nộp")
    ap.add_argument("--n-shot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = toàn bộ")
    ap.add_argument("--model", default=None, help="đường-dẫn HF hoặc tên model API")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--dummy-fixed", action="store_true", help="dummy luôn chọn A (thay vì ngẫu-nhiên)")
    ap.add_argument("--out", default=None, help="lưu dự-đoán JSON (bắt buộc cho split=test)")
    ap.add_argument("--shuffle", action="store_true",
                    help="xáo thứ-tự câu (seed 0): acc tạm-tính giữa chừng = ước-lượng không thiên-lệch mọi môn")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(a.repo, split=a.split)
    shots = list(load_dataset(a.repo, split="train"))[:a.n_shot] if a.n_shot else []
    rows = list(ds)
    if a.shuffle:
        random.Random(0).shuffle(rows)                    # dataset xếp theo môn → cắt/dừng giữa chừng sẽ lệch
    rows = rows[:a.limit] if a.limit else rows
    print(f"[{time.time()-T0:.0f}s] {a.repo}/{a.split}: {len(rows)} câu · {a.n_shot}-shot · backend={a.backend}", flush=True)

    predict = make_predictor(a)
    rag = RagCtx(k=a.rag_k, max_docs=a.rag_docs) if a.rag else None
    has_ans = all(r.get("answer") for r in rows[:5])      # validation có; test rỗng
    per = collections.defaultdict(lambda: [0, 0])         # môn → [đúng, tổng]
    preds = {}
    correct = total = 0
    for i, r in enumerate(rows):
        n = len(r["choices"])
        ctx_text = rag.context(r["question"]) if rag else None
        p = predict(build_prompt(r, shots, ctx_text), n)
        preds[r["id"]] = p
        subj = r["id"].split("-")[0]
        if has_ans:
            gold = r["answer"].strip().upper()
            hit = int(p == gold)
            correct += hit; total += 1
            per[subj][0] += hit; per[subj][1] += 1
        if (i + 1) % (50 if a.shuffle else 200) == 0:
            acc = f"{100*correct/total:.1f}%" if total else "—"
            print(f"[{time.time()-T0:.0f}s]   {i+1}/{len(rows)} · acc {acc}", flush=True)

    if a.out or a.split == "test":
        out = a.out or "out/vmlu_submission.json"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        # định-dạng nộp VMLU: [{"id":..., "answer":...}]
        with open(out, "w", encoding="utf-8") as f:
            json.dump([{"id": k, "answer": v} for k, v in preds.items()], f, ensure_ascii=False, indent=2)
        print(f"[{time.time()-T0:.0f}s] LƯU dự-đoán {out} ({len(preds)} câu)", flush=True)

    print(f"\n[{time.time()-T0:.0f}s] ===== KẾT QUẢ =====", flush=True)
    if has_ans:
        print(f"  ACCURACY TỔNG: {100*correct/total:.2f}%  ({correct}/{total})", flush=True)
        print(f"  (mốc: SeaLLM-7B-v2.5 ≈ 53.3 · VNPTAI-14B ≈ 55.8)", flush=True)
        worst = sorted(per.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]))[:8]
        print("  8 môn YẾU nhất (môn-id: acc, n):", flush=True)
        for s, (c, t) in worst:
            print(f"     {s}: {100*c/t:5.1f}%  (n={t})", flush=True)
    else:
        print("  split không có đáp-án → chỉ xuất file nộp (không chấm tại-chỗ).", flush=True)


if __name__ == "__main__":
    main()
