"""TK_105 · Giai đoạn 2 — CẦU DỊCH-RA: Zamba2-7B đọc (tiếng Anh) → 1.3B NÓI tiếng Việt, nhìn ẩn Zamba2 qua cross-attention.

    câu Anh ──► Zamba2-7B (đóng băng, không gradient) ──ẩn 3584, trộn 3 tầng (học trọng số)──┐
    "Hỏi: Dịch sang tiếng Việt.\\nĐáp: " ──► 1.3B (đóng băng) ──[sau khối 7,15,…,47: x += σ(g)·XAttn(x, H_z)]──► câu Việt
XAttn khởi tạo lớp ra = 0 → lúc đầu y hệt 1.3B gốc (không quên); ~140M tham số học.
1.3B KHÔNG thấy câu Anh — mọi nội dung phải đi qua cross-attention từ Zamba2 → đo đúng "cầu dịch-ra truyền ý".

Thước đo: BLEU/chrF Anh→Việt trên câu GIỮ RIÊNG (DailyDialog test):
    bridge   : 1.3B nói qua cầu (cái ta xây)
    zamba_vi : Zamba2 tự dịch sang tiếng Việt (mốc — Zamba2 viết tiếng Việt yếu)

    /workspace/zenv/bin/python scripts/tk105_bridge2.py train --steps 3000
    /workspace/zenv/bin/python scripts/tk105_bridge2.py eval --xattn ckpt/tk105/bridge_out.pt
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
sys.path[:0] = ["scripts", "src"]
import torch
import torch.nn as nn
import torch.nn.functional as F
from tk105_bridge import pairs, ZAMBA, OUT, log

POS = [7, 15, 23, 31, 39, 47]
PROMPT = "Hỏi: Dịch sang tiếng Việt.\n"


class XAttn(nn.Module):
    def __init__(self, d=2048, dkv=3584, heads=16):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.nq, self.nkv = nn.RMSNorm(d), nn.LayerNorm(dkv)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(dkv, d, bias=False)
        self.v = nn.Linear(dkv, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o.weight)                       # đồng nhất lúc chèn
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, x, H, kmask):
        B, T, D = x.shape
        xn = self.nq(x)
        Hn = self.nkv(H)
        q = self.q(xn).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(Hn).view(B, -1, self.h, self.hd).transpose(1, 2)
        v = self.v(Hn).view(B, -1, self.h, self.hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=kmask[:, None, None, :])
        self.g = torch.sigmoid(self.gate(xn))
        return x + self.g * self.o(a.transpose(1, 2).reshape(B, T, D))


class Bridge2(nn.Module):
    def __init__(self, n_zlayers, pick):
        super().__init__()
        self.pick = pick                                   # tầng Zamba2 lấy ẩn
        self.mix = nn.Parameter(torch.zeros(len(pick)))
        self.xa = nn.ModuleDict({str(i): XAttn() for i in POS})

    def zstate(self, hs):
        w = torch.softmax(self.mix, 0)
        return sum(w[j] * hs[min(i, len(hs) - 1)].float() for j, i in enumerate(self.pick))


class P2:
    def __init__(self, a, dev="cuda"):
        import expert_ffn_1p3b as E
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from nn_aivie.core.vocab import Vocabulary
        E.BASE = a.base
        self.ours, _ = E.load_trunk(dev)
        self.ours.eval()
        self.tok = E.tok_()
        self.voc = Vocabulary(self.tok)
        self.eos = self.tok.token_to_id["<EOS>"]
        self.ztok = AutoTokenizer.from_pretrained(ZAMBA)
        self.z = AutoModelForCausalLM.from_pretrained(ZAMBA, dtype=torch.bfloat16, device_map=dev).eval()
        for p in self.z.parameters():
            p.requires_grad_(False)
        nl = self.z.config.num_hidden_layers
        self.pick = [nl // 2, (3 * nl) // 4, nl]           # giữa, ¾, cuối
        self.dev = dev
        self.prompt = self.tok.encode(PROMPT) + self.tok.encode("Đáp: ")

    @torch.no_grad()
    def zhidden(self, en):
        ids = [self.ztok(s, add_special_tokens=False).input_ids[:128] for s in en]
        T = max(len(x) for x in ids)
        X = torch.full((len(ids), T), self.ztok.pad_token_id or 0, dtype=torch.long, device=self.dev)
        M = torch.zeros(len(ids), T, dtype=torch.bool, device=self.dev)
        for i, x in enumerate(ids):
            X[i, :len(x)] = torch.tensor(x); M[i, :len(x)] = True
        out = self.z(input_ids=X, attention_mask=M.long(), output_hidden_states=True, use_cache=False)
        return out.hidden_states, M

    def forward_ours(self, br, ids, H, M):
        m = self.ours
        x = m.emb(ids)
        for i, (blk, nm) in enumerate(zip(m.blocks, m.norms)):
            if i < POS[0]:
                with torch.no_grad():
                    x = x + blk(nm(x))
            else:
                x = x + torch.utils.checkpoint.checkpoint(lambda z, b=blk, n=nm: b(n(z)), x, use_reentrant=False)
            if str(i) in br.xa:
                x = br.xa[str(i)](x, H, M)
        return m.norm_f(x) @ m.emb.weight.T

    def batch_ids(self, vi_list):
        rows, labs = [], []
        for vi in vi_list:
            t = self.tok.encode(vi)[:96] + [self.eos]
            rows.append(self.prompt + t); labs.append([-100] * len(self.prompt) + t)
        T = max(len(r) for r in rows)
        X = torch.zeros(len(rows), T, dtype=torch.long, device=self.dev)
        Y = torch.full((len(rows), T), -100, dtype=torch.long, device=self.dev)
        for i, (r, l) in enumerate(zip(rows, labs)):
            X[i, :len(r)] = torch.tensor(r); Y[i, :len(l)] = torch.tensor(l)
        return X, Y


def train(a):
    P = P2(a)
    br = Bridge2(P.z.config.num_hidden_layers, P.pick).to(P.dev)
    log(f"nạp xong · cross-attn {sum(p.numel() for p in br.parameters())/1e6:.0f}M · tầng Zamba2 {P.pick}")
    tr = pairs("train"); random.Random(0).shuffle(tr)
    opt = torch.optim.AdamW(br.parameters(), lr=a.lr, weight_decay=0.01)
    rng = random.Random(1)
    for s in range(a.steps):
        lr = a.lr * min(1, (s + 1) / 100) * (0.5 * (1 + math.cos(math.pi * s / a.steps)) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = rng.sample(tr, a.bs)
        hs, M = P.zhidden([e for _, e in batch])
        X, Y = P.batch_ids([v for v, _ in batch])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            H = br.zstate(hs).to(torch.bfloat16)
            lo = P.forward_ours(br, X[:, :-1], H, M)
        loss = F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), Y[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(br.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s % 50 == 0 or s == a.steps - 1:
            g = float(torch.stack([x.g.float().mean() for x in br.xa.values()]).mean())
            log(f"bước {s}/{a.steps} · loss {loss.item():.3f} · cổng {g:.2f} · trộn tầng {torch.softmax(br.mix, 0).tolist()}")
        if a.save_every and s and s % a.save_every == 0:
            os.makedirs(OUT, exist_ok=True); torch.save(br.state_dict(), f"{OUT}/{a.tag}.pt")
    os.makedirs(OUT, exist_ok=True)
    torch.save(br.state_dict(), f"{OUT}/{a.tag}.pt")
    log(f"### LƯU {OUT}/{a.tag}.pt")


@torch.no_grad()
def gen_ours(P, br, en, n=80):
    hs, M = P.zhidden(en)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        H = br.zstate(hs).to(torch.bfloat16)
    seqs = [list(P.prompt) for _ in en]
    done = [False] * len(en)
    for _ in range(n):
        T = max(len(s) for s in seqs)
        X = torch.zeros(len(seqs), T, dtype=torch.long, device=P.dev)
        for i, s in enumerate(seqs):
            X[i, :len(s)] = torch.tensor(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = P.forward_ours(br, X, H, M)
        for i, s in enumerate(seqs):
            if done[i]:
                continue
            t = int(lo[i, len(s) - 1].argmax())
            if t == P.eos:
                done[i] = True
            else:
                s.append(t)
        if all(done):
            break
    return [P.voc.detokenize(s[len(P.prompt):]).strip() for s in seqs]


@torch.no_grad()
def gen_zamba_vi(P, en, n=100):
    out = []
    for s in en:
        msgs = [{"role": "user", "content": f"Translate into Vietnamese. Output only the translation.\n\n{s}"}]
        x = P.ztok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        x = (x["input_ids"] if not torch.is_tensor(x) else x).to(P.dev)
        y = P.z.generate(x, max_new_tokens=n, do_sample=False)
        out.append(P.ztok.decode(y[0, x.shape[1]:], skip_special_tokens=True).strip())
    return out


def evaluate(a):
    import sacrebleu
    P = P2(a)
    br = Bridge2(P.z.config.num_hidden_layers, P.pick).to(P.dev)
    br.load_state_dict(torch.load(a.xattn, map_location=P.dev)); br.eval()
    ps = pairs("test")[: a.n]
    ref = [v for v, _ in ps]
    res = {}
    hyp = []
    for i in range(0, len(ps), 16):
        hyp += gen_ours(P, br, [e for _, e in ps[i:i + 16]])
    res["bridge"] = hyp
    res["zamba_vi"] = gen_zamba_vi(P, [e for _, e in ps])
    summ = {}
    for k, h in res.items():
        summ[k] = {"bleu": round(sacrebleu.corpus_bleu(h, [ref]).score, 1), "chrf": round(sacrebleu.corpus_chrf(h, [ref]).score, 1)}
        log(f"{k}: BLEU {summ[k]['bleu']} · chrF {summ[k]['chrf']} (n={len(ps)})")
    summ["vi_du"] = [{"en": e, "ref": v, "bridge": b, "zamba_vi": z}
                     for (v, e), b, z in list(zip(ps, res["bridge"], res["zamba_vi"]))[:8]]
    os.makedirs(OUT, exist_ok=True)
    json.dump(summ, open(f"{OUT}/eval2_{a.tag}.json", "w"), ensure_ascii=False, indent=1)
    log(f"### ĐÁNH GIÁ GĐ2 XONG → {OUT}/eval2_{a.tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--base", default="ckpt/hybrid_1p3b_chat_v2.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--tag", default="bridge_out")
    ap.add_argument("--xattn", default="")
    ap.add_argument("--n", type=int, default=150, help="số câu giữ riêng (Zamba2 tự dịch chậm ~8s/câu)")
    a = ap.parse_args()
    torch.manual_seed(0)
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
