"""TK_105 · Giai đoạn 3 — GHÉP ĐẦU-CUỐI: hỏi tiếng Việt → Zamba2-7B nghĩ nháp (tiếng Anh, ẩn) → 1.3B đáp tiếng Việt.

    câu hỏi Việt ─┬─► [cầu dịch-vào v2: token mềm + câu Việt thô] ─► Zamba2-7B (đóng băng) tự viết NHÁP lời giải tiếng Anh
                  │                                                    (ẩn, người dùng không thấy; 1.3B không đọc chữ nháp)
                  │                         ẩn Zamba2 trên [câu hỏi + nháp] ─┐
                  └─► 1.3B (đóng băng) "Hỏi: …\\nĐáp: " ──[6 cross-attention có cổng = cầu dịch-ra]──► câu trả lời TIẾNG VIỆT
Vì sao có nháp: TK_104 — "viết lời giải >> nghĩ thầm"; Zamba2 đọc bản Anh giải đúng 18/18 toán eval_v1 (Bước 0).

Train (rẻ, KHÔNG cần Zamba2 sinh chữ): GSM8K có cặp Anh–Việt — Zamba2 đọc [câu hỏi Việt + lời giải Anh gốc như thể chính
nó đã nghĩ], 1.3B học viết lời giải Việt (Vi-GSM8K) qua cross-attention (khởi tạo từ cầu dịch-ra gđ2); trộn 30% câu dịch
DailyDialog (giữ khả năng dịch-ra). Học: cross-attention + trộn tầng; mọi thứ khác đóng băng.

Đo (lúc chạy thật Zamba2 TỰ sinh nháp): toán Vi-GSM8K test (100) + toán eval_v1 (18) → đáp số cuối; sự kiện + thực thể bịa
eval_v1 → Gemma chấm; so 1.3B một mình và "Zamba2 tự nháp tiếng Anh" (đáp số trong nháp).

    /workspace/zenv/bin/python scripts/tk105_stage3.py train --xattn ckpt/tk105/bridge_out.pt [--bridge-in ckpt/tk105/bridge_in_v2.pt]
    /workspace/zenv/bin/python scripts/tk105_stage3.py eval --xattn ckpt/tk105/stage3.pt [--bridge-in …]
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys, time
sys.path[:0] = ["scripts", "src"]
import torch
import torch.nn.functional as F
import tk105_bridge as T1
from tk105_bridge2 import Bridge2, POS, PROMPT as PROMPT_DICH
from tk105_bridge import log, OUT

SUF_MATH = "\nSolve the problem above step by step in English. End with: The answer is <number>.<|im_end|>\n<|im_start|>assistant\n"
SUF_QA = ("\nAnswer the question above in English, briefly. If you do not know, or the thing does not exist, say you do not know."
          "<|im_end|>\n<|im_start|>assistant\n")


def num_last(s):
    m = re.findall(r"-?\d[\d.,]*", s or "")
    if not m:
        return None
    x = m[-1].rstrip(".,")
    x = re.sub(r"[.,](?=\d{3}\b)", "", x).replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return None


def gsm_pairs():
    from datasets import load_dataset
    vi = load_dataset("longhoang06/Vi-GSM8K", split="train")
    en = load_dataset("openai/gsm8k", "main")
    out = {"train": [], "test": []}
    for split, off in (("train", 0), ("test", len(en["train"]))):
        for j, e in enumerate(en[split]):
            v = vi[off + j]
            g_en = num_last(e["answer"].split("####")[-1])
            g_vi = num_last(v["answer"])
            if g_en is None or g_en != g_vi:              # căn hàng theo chỉ số — chỉ giữ cặp cùng đáp số
                continue
            sol = re.sub(r"<<[^>]*>>", "", e["answer"]).split("####")[0].strip() + f"\nThe answer is {e['answer'].split('####')[-1].strip()}."
            out[split].append({"vi_q": v["question"].strip(), "vi_a": v["answer"].strip(), "en_a": sol, "gold": g_en})
    return out


class P3(T1.Parts):
    def __init__(self, a):
        super().__init__(a)
        self.max_tgt = 384                                    # nháp/lời giải dài hơn câu dịch
        from nn_aivie.core.vocab import Vocabulary
        self.voc = Vocabulary(self.tok_vi)
        self.eos = self.tok_vi.token_to_id["<EOS>"]
        enc = lambda s: self.ztok(s, add_special_tokens=False).input_ids
        self.suf_math, self.suf_qa = enc(SUF_MATH), enc(SUF_QA)
        nl = self.z.config.num_hidden_layers
        self.br = Bridge2(nl, [nl // 2, (3 * nl) // 4, nl]).to(self.dev)
        self.br.load_state_dict(torch.load(a.xattn, map_location=self.dev))
        self.bin = None
        if a.bridge_in:
            self.bin = T1.Bridge(emb_rms=self.emb_rms).to(self.dev)
            self.bin.load_state_dict(torch.load(a.bridge_in, map_location=self.dev)); self.bin.eval()
            for p in self.bin.parameters():
                p.requires_grad_(False)

    def zin(self, vi_qs, suf, targets=None):
        """embedding đầu vào Zamba2: (token mềm +) câu Việt thô + chỉ dẫn (+ nháp/đích)."""
        self.suf = suf
        with torch.no_grad():
            if self.bin is not None:
                h, lens = self.hidden_vi(vi_qs)
                soft = self.bin(h)
            else:
                soft, lens = None, [0] * len(vi_qs)
            return self.build(soft, lens, targets=targets, raw_vi=vi_qs)

    @torch.no_grad()
    def zstate(self, emb, mask):
        hs = self.z(inputs_embeds=emb, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.br.zstate(hs).to(torch.bfloat16), mask.bool()

    @torch.no_grad()
    def zstate_text(self, en):
        ids = [self.ztok(s, add_special_tokens=False).input_ids[:160] for s in en]
        Tm = max(len(x) for x in ids)
        X = torch.zeros(len(ids), Tm, dtype=torch.long, device=self.dev)
        M = torch.zeros(len(ids), Tm, dtype=torch.long, device=self.dev)
        for i, x in enumerate(ids):
            X[i, :len(x)] = torch.tensor(x); M[i, :len(x)] = 1
        hs = self.z(input_ids=X, attention_mask=M, output_hidden_states=True, use_cache=False).hidden_states
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.br.zstate(hs).to(torch.bfloat16), M.bool()

    def fwd(self_, ids, H, M):
        """1.3B (self_.ours = thân đóng băng) + cross-attention đọc ẩn Zamba2 (H=None → 1.3B một mình)."""
        m = self_.ours
        x = m.emb(ids)
        for i, (blk, nm) in enumerate(zip(m.blocks, m.norms)):
            if i < POS[0]:
                with torch.no_grad():
                    x = x + blk(nm(x))
            else:
                x = x + torch.utils.checkpoint.checkpoint(lambda z, b=blk, n=nm: b(n(z)), x, use_reentrant=False)
            if H is not None and str(i) in self_.br.xa:
                x = self_.br.xa[str(i)](x, H, M)
        return m.norm_f(x) @ m.emb.weight.T

    def prompt_ids(self, q):
        return self.tok_vi.encode("Hỏi: " + q.strip() + "\n") + self.tok_vi.encode("Đáp: ")

    def batch_ours(self, qs, answers):
        rows, labs = [], []
        for q, a_ in zip(qs, answers):
            p = self.prompt_ids(q) if q is not None else self.tok_vi.encode(PROMPT_DICH) + self.tok_vi.encode("Đáp: ")
            t = self.tok_vi.encode(a_)[:320] + [self.eos]
            rows.append(p + t); labs.append([-100] * len(p) + t)
        Tm = max(len(r) for r in rows)
        X = torch.zeros(len(rows), Tm, dtype=torch.long, device=self.dev)
        Y = torch.full((len(rows), Tm), -100, dtype=torch.long, device=self.dev)
        for i, (r, l) in enumerate(zip(rows, labs)):
            X[i, :len(r)] = torch.tensor(r); Y[i, :len(l)] = torch.tensor(l)
        return X, Y


def train(a):
    P = P3(a)
    P.rep_pen = 1.0
    data = gsm_pairs()
    dd = T1.pairs("train")
    log(f"GSM8K Anh–Việt khớp: train {len(data['train'])} · test {len(data['test'])} · cầu dịch-vào {'có' if P.bin else 'không'}")
    params = list(P.br.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    rng = random.Random(4)
    for s in range(a.steps):
        lr = a.lr * min(1, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * s / a.steps)) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        if rng.random() < 0.3:                                    # giữ khả năng dịch-ra (gđ2)
            b = rng.sample(dd, a.bs)
            H, M = P.zstate_text([e for _, e in b])
            X, Y = P.batch_ours([None] * a.bs, [v for v, _ in b])
        else:
            b = rng.sample(data["train"], a.bs)
            emb, mask, _ = P.zin([x["vi_q"] for x in b], P.suf_math, targets=[x["en_a"] for x in b])
            H, M = P.zstate(emb, mask)
            X, Y = P.batch_ours([x["vi_q"] for x in b], [x["vi_a"] for x in b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = P.fwd(X[:, :-1], H, M)
        loss = F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), Y[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s % 50 == 0 or s == a.steps - 1:
            g = float(torch.stack([x.g.float().mean() for x in P.br.xa.values()]).mean())
            log(f"bước {s}/{a.steps} · loss {loss.item():.3f} · cổng {g:.2f}")
    os.makedirs(OUT, exist_ok=True)
    torch.save(P.br.state_dict(), f"{OUT}/{a.tag}.pt")
    log(f"### LƯU {OUT}/{a.tag}.pt")


@torch.no_grad()
def zamba_draft(P, qs, suf, n):
    emb, mask, _ = P.zin(qs, suf)
    return T1.greedy(P, emb, mask, n=n)


@torch.no_grad()
def ours_answer(P, qs, H, M, n):
    seqs = [P.prompt_ids(q) for q in qs]
    p_len = [len(s) for s in seqs]
    done = [False] * len(qs)
    for _ in range(n):
        Tm = max(len(s) for s in seqs)
        X = torch.zeros(len(seqs), Tm, dtype=torch.long, device=P.dev)
        for i, s in enumerate(seqs):
            X[i, :len(s)] = torch.tensor(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = P.fwd(X, H, M)
        for i, s in enumerate(seqs):
            if done[i]:
                continue
            l_ = lo[i, len(s) - 1].float()
            if P.rep_pen > 1.0:                                # phạt lặp (gđ3 v1 hay lặp câu tới hết độ dài)
                rec = torch.tensor(sorted(set(s[max(p_len[i], len(s) - 64):])), dtype=torch.long, device=l_.device)
                if len(rec):
                    v = l_[rec]
                    l_[rec] = torch.where(v > 0, v / P.rep_pen, v * P.rep_pen)
            t = int(l_.argmax())
            if t == P.eos:
                done[i] = True
            else:
                s.append(t)
        if all(done):
            break
    return [P.voc.detokenize(s[pl:]).strip() for s, pl in zip(seqs, p_len)]


def pipeline(P, qs, suf, n_draft, n_ans, bs=12):
    """chạy THẬT: Zamba2 tự viết nháp → ẩn trên [câu hỏi + nháp] → 1.3B đáp tiếng Việt qua cross-attention."""
    drafts, answers = [], []
    for i in range(0, len(qs), bs):
        q = qs[i:i + bs]
        d = zamba_draft(P, q, suf, n_draft)
        emb, mask, _ = P.zin(q, suf, targets=d)
        H, M = P.zstate(emb, mask)
        drafts += d
        answers += ours_answer(P, q, H, M, n_ans)
        log(f"  {min(i + bs, len(qs))}/{len(qs)}")
    return drafts, answers


def evaluate(a):
    P = P3(a)
    P.rep_pen = a.rep_pen
    res = {}
    data = gsm_pairs()
    ev = [json.loads(l) for l in open("data/eval_clean/eval_v1.jsonl", encoding="utf-8")]
    math_items = data["test"][: a.n] + [{"vi_q": r["q"], "gold": num_last(r["gold"]), "src": "eval_v1"} for r in ev if r["type"] == "math"]
    qs = [x["vi_q"] for x in math_items]
    drafts, answers = pipeline(P, qs, P.suf_math, 320, a.max_ans)
    alone = []
    for i in range(0, len(qs), 12):                               # 1.3B một mình (không cross-attention)
        alone += ours_answer(P, qs[i:i + 12], None, None, a.max_ans)
    rows = []
    for x, d, ans, al in zip(math_items, drafts, answers, alone):
        rows.append({**x, "draft": d, "answer": ans, "alone": al,
                     "ok_ghep": num_last(ans) == x["gold"], "ok_nhap": num_last(d) == x["gold"], "ok_1p3b": num_last(al) == x["gold"]})
    for src in ("gsm", "eval_v1"):
        rs = [r for r in rows if (r.get("src") == "eval_v1") == (src == "eval_v1")]
        res[f"toan_{src}"] = {"n": len(rs), **{k: sum(r[k] for r in rs) for k in ("ok_ghep", "ok_nhap", "ok_1p3b")}}
        log(f"toán {src}: ghép {res[f'toan_{src}']['ok_ghep']}/{len(rs)} · nháp Zamba2 {res[f'toan_{src}']['ok_nhap']} · 1.3B một mình {res[f'toan_{src}']['ok_1p3b']}")
    qa = [r for r in ev if r["type"] in ("fact", "unanswerable")]
    d2, a2 = pipeline(P, [r["q"] for r in qa], P.suf_qa, 120, 200)
    sys.path.insert(0, "scripts")
    from eval_clean import judge_one
    from collections import Counter
    cnt = Counter()
    qa_rows = []
    for r, d, ans in zip(qa, d2, a2):
        try:
            j = judge_one({**r, "answer": ans}, "gemma4:26b-a4b-it-qat")
            cnt[(r["type"], str(j.get("verdict", "?")).lower())] += 1
        except Exception as e:
            j = {"err": str(e)[:80]}
        qa_rows.append({**r, "draft": d, "answer": ans, "j": j})
    res["qa"] = {f"{t}/{v}": c for (t, v), c in cnt.items()}
    log(f"sự kiện + thực thể bịa: {res['qa']}")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"tom_tat": res, "toan": rows[:40] + [r for r in rows if r.get("src") == "eval_v1"], "qa": qa_rows},
              open(f"{OUT}/eval3_{a.tag}.json", "w"), ensure_ascii=False, indent=1)
    log(f"### GĐ3 ĐÁNH GIÁ XONG → {OUT}/eval3_{a.tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--base", default="ckpt/hybrid_1p3b_chat_v2.pt")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--xattn", required=True, help="cầu dịch-ra (gđ2 để train, gđ3 để đo)")
    ap.add_argument("--bridge-in", default="", help="cầu dịch-vào v2 (bổ sung); '' = Zamba2 chỉ đọc câu Việt thô")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tag", default="stage3")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rep-pen", type=float, default=1.0)
    ap.add_argument("--max-ans", type=int, default=360)
    a = ap.parse_args()
    torch.manual_seed(0)
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
