"""TK_105 · Giai đoạn 1 — CẦU DỊCH-VÀO: 1.3B (đọc tiếng Việt) → token mềm → Zamba2-7B (nghĩ bằng "tiếng Anh").

    câu Việt ──BPE ta──► 1.3B khối 0..L (đóng băng) ──ẩn 2048──► CẦU (LN → MLP 2048→4096→3584, ~25M, HỌC)
            ──token mềm (1 token Việt = 1 token mềm)──► Zamba2-7B-Instruct (đóng băng) ──► viết bản tiếng Anh

Mục tiêu học (kiểu MindMerger gđ 1): Zamba2 nhận token mềm trong khuôn chat
    <|im_start|>user\n [token mềm] \nTranslate the text above into English.<|im_end|>\n<|im_start|>assistant\n  →  bản Anh
loss chỉ trên bản Anh; gradient đi XUYÊN Zamba2 đóng băng về cầu.

Thước đo "Zamba2 hiểu đúng ý": BLEU/chrF của bản Anh sinh ra so với bản Anh chuẩn, trên câu GIỮ RIÊNG:
    soft   : qua cầu (cái ta xây)
    vi_raw : Zamba2 đọc thẳng câu Việt bằng bộ token của nó (mốc dưới — pilot B: Zamba2 yếu tiếng Việt)
Dữ liệu: DailyDialog Anh–Việt (data/giaotiep/dd_envi.jsonl, chia theo hội thoại) + đề eval_v1 (Việt → Anh chuẩn của Gemma).

    /workspace/zenv/bin/python scripts/tk105_bridge.py train --steps 3000      # box H100 (venv có transformers đã vá)
    /workspace/zenv/bin/python scripts/tk105_bridge.py eval --bridge ckpt/tk105/bridge_in.pt
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
sys.path[:0] = ["scripts", "src"]
import torch
import torch.nn as nn
import torch.nn.functional as F

ZAMBA = "Zyphra/Zamba2-7B-Instruct"
OUT = "ckpt/tk105"
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:.0f}s]", *a, flush=True)


# ───────── dữ liệu song ngữ ─────────
def pairs(split):
    out = []
    for l in open("data/giaotiep/dd_envi.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if (r["split"] == "train") != (split == "train"):
            continue
        for vi, en in zip(r["vi"], r["en"]):
            vi, en = vi.strip(), en.strip()
            if 3 <= len(vi.split()) <= 60 and en:
                out.append((vi, en))
    return out


def evalv1_pairs():
    p = "data/zamba_ceiling/evalv1_en.jsonl"
    return [(r["q"], r["q_en"]) for r in map(json.loads, open(p, encoding="utf-8"))] if os.path.exists(p) else []


# ───────── mô hình ─────────
class Bridge(nn.Module):
    def __init__(self, d_in=2048, d_out=3584, hid=4096, emb_rms=1.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hid)
        self.fc2 = nn.Linear(hid, d_out)
        # khớp độ lớn embedding thật của Zamba2 — CỐ ĐỊNH (v1 để học: tự phình ×3,4 → Zamba2 coi là nhiễu)
        self.register_buffer("scale", torch.tensor(float(emb_rms)))

    def forward(self, h):
        y = self.fc2(F.gelu(self.fc1(self.ln(h))))
        return self.scale * y / (y.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)


class Parts:
    """1.3B đóng băng (đọc Việt, tới khối L) + Zamba2 đóng băng + bộ token hai bên."""
    def __init__(self, a, dev="cuda"):
        import expert_ffn_1p3b as E
        from transformers import AutoModelForCausalLM, AutoTokenizer
        E.BASE = a.base
        self.ours, _ = E.load_trunk(dev)
        self.ours.eval()
        self.tok_vi = E.tok_()
        self.L = a.layer
        self.ztok = AutoTokenizer.from_pretrained(ZAMBA)
        self.z = AutoModelForCausalLM.from_pretrained(ZAMBA, dtype=torch.bfloat16, device_map=dev).eval()
        for p in self.z.parameters():
            p.requires_grad_(False)
        self.zemb = self.z.get_input_embeddings()
        self.dev = dev
        enc = lambda s: self.ztok(s, add_special_tokens=False).input_ids
        self.pre = enc("<|im_start|>user\n")
        self.suf = enc("\nTranslate the text above into English.<|im_end|>\n<|im_start|>assistant\n")
        self.end = enc("<|im_end|>")
        self.nl = enc("\n")
        self.emb_rms = float(self.zemb.weight.float().pow(2).mean(-1).sqrt().mean())

    @torch.no_grad()
    def hidden_vi(self, texts):
        """ẩn sau khối L của 1.3B cho từng câu Việt (đệm phải; Mamba nhân quả → đệm phải không ảnh hưởng token thật)."""
        ids = [self.tok_vi.encode(t)[:96] for t in texts]
        T = max(len(x) for x in ids)
        X = torch.zeros(len(ids), T, dtype=torch.long, device=self.dev)
        for i, x in enumerate(ids):
            X[i, :len(x)] = torch.tensor(x)
        m = self.ours
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = m.emb(X)
            for i, (blk, nm) in enumerate(zip(m.blocks, m.norms)):
                if i > self.L:
                    break
                x = x + blk(nm(x))
        return x.float(), [len(x) for x in ids]

    def build(self, soft, lens, targets=None, raw_vi=None):
        """ghép embedding: pre + (token mềm | câu Việt thô) + suf (+ đích) — đệm phải. Trả embeds, mask, nhãn."""
        E_ = lambda ids: self.zemb(torch.tensor(ids, device=self.dev))
        seqs, labs = [], []
        for b in range(len(lens)):
            if soft is None:                                   # chỉ câu Việt thô (mốc)
                mid = E_(self.ztok(raw_vi[b], add_special_tokens=False).input_ids[:192])
            elif raw_vi is None:                               # chỉ token mềm (v1)
                mid = soft[b, :lens[b]].to(torch.bfloat16)
            else:                                              # BỔ SUNG (v2, kiểu MindMerger): token mềm + câu Việt thô
                mid = torch.cat([soft[b, :lens[b]].to(torch.bfloat16), E_(self.nl),
                                 E_(self.ztok(raw_vi[b], add_special_tokens=False).input_ids[:192])])
            parts = [E_(self.pre), mid, E_(self.suf)]
            lab = [-100] * (len(self.pre) + mid.shape[0] + len(self.suf))
            if targets is not None:
                t = self.ztok(targets[b], add_special_tokens=False).input_ids[:getattr(self, "max_tgt", 96)] + self.end
                parts.append(E_(t)); lab += t
            seqs.append(torch.cat(parts)); labs.append(lab)
        T = max(s.shape[0] for s in seqs)
        D = seqs[0].shape[1]
        emb = torch.zeros(len(seqs), T, D, dtype=torch.bfloat16, device=self.dev)
        mask = torch.zeros(len(seqs), T, dtype=torch.long, device=self.dev)
        Y = torch.full((len(seqs), T), -100, dtype=torch.long, device=self.dev)
        for b, (s, l) in enumerate(zip(seqs, labs)):
            emb[b, :len(s)] = s; mask[b, :len(s)] = 1; Y[b, :len(l)] = torch.tensor(l)
        return emb, mask, Y


def train(a):
    P = Parts(a)
    br = Bridge(emb_rms=P.emb_rms).to(P.dev)
    log(f"nạp xong · cầu {sum(p.numel() for p in br.parameters())/1e6:.1f}M · ẩn tầng {a.layer} · emb_rms {P.emb_rms:.3f}")
    # lan truyền ngược XUYÊN 7B: checkpoint từng tầng (không thì ~50GB kích hoạt, đụng Gemma đang sinh dữ liệu)
    P.z.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    P.z.train()
    tr = pairs("train"); random.Random(0).shuffle(tr)
    log(f"cặp Việt–Anh train {len(tr)}")
    opt = torch.optim.AdamW(br.parameters(), lr=a.lr, weight_decay=0.01)
    rng = random.Random(1)
    for s in range(a.steps):
        lr = a.lr * min(1, (s + 1) / 100) * (0.5 * (1 + math.cos(math.pi * s / a.steps)) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = rng.sample(tr, a.bs)
        h, lens = P.hidden_vi([v for v, _ in batch])
        soft = br(h)
        emb, mask, Y = P.build(soft, lens, targets=[e for _, e in batch],
                               raw_vi=[v for v, _ in batch] if a.augment else None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = P.z(inputs_embeds=emb, attention_mask=mask, use_cache=False).logits
        loss = F.cross_entropy(lo[:, :-1].float().reshape(-1, lo.size(-1)), Y[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(br.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s % 50 == 0 or s == a.steps - 1:
            log(f"bước {s}/{a.steps} · loss {loss.item():.3f} · lr {lr:.2e}")
        if a.save_every and s and s % a.save_every == 0:
            os.makedirs(OUT, exist_ok=True); torch.save(br.state_dict(), f"{OUT}/{a.tag}.pt")
    os.makedirs(OUT, exist_ok=True)
    torch.save(br.state_dict(), f"{OUT}/{a.tag}.pt")
    log(f"### LƯU {OUT}/{a.tag}.pt")


@torch.no_grad()
def greedy(P, emb, mask, n=80):
    """sinh tham lam không cache (câu ngắn) — chắc chắn đúng với inputs_embeds + đệm phải."""
    B = emb.shape[0]
    lens = mask.sum(1)
    out = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=P.dev)
    end = P.end[0]
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = P.z(inputs_embeds=emb, attention_mask=mask, use_cache=False).logits
        nxt = lo[torch.arange(B), lens - 1].argmax(-1)
        T = emb.shape[1]
        if int(lens.max()) >= T:
            emb = torch.cat([emb, torch.zeros(B, 1, emb.shape[2], dtype=emb.dtype, device=P.dev)], 1)
            mask = torch.cat([mask, torch.zeros(B, 1, dtype=mask.dtype, device=P.dev)], 1)
        e = P.zemb(nxt)
        for b in range(B):
            if done[b]:
                continue
            t = int(nxt[b])
            if t == end or t == P.ztok.eos_token_id:
                done[b] = True; continue
            out[b].append(t)
            emb[b, lens[b]] = e[b]; mask[b, lens[b]] = 1
        lens = mask.sum(1)
        if bool(done.all()):
            break
    return [P.ztok.decode(o, skip_special_tokens=True).strip() for o in out]


def evaluate(a):
    import sacrebleu
    P = Parts(a)
    br = None
    if a.bridge:
        br = Bridge(emb_rms=P.emb_rms).to(P.dev)
        br.load_state_dict(torch.load(a.bridge, map_location=P.dev)); br.eval()
    sets = {"dd_test": pairs("test")[: a.n], "evalv1": evalv1_pairs()[: a.n]}
    res = {}
    for name, ps in sets.items():
        if not ps:
            continue
        for mode in (["soft"] if br else []) + ["vi_raw"]:
            hyp = []
            for i in range(0, len(ps), 16):
                chunk = ps[i:i + 16]
                vis = [v for v, _ in chunk]
                if mode == "soft":
                    h, lens = P.hidden_vi(vis)
                    emb, mask, _ = P.build(br(h), lens, raw_vi=vis if a.augment else None)
                else:
                    emb, mask, _ = P.build(None, [0] * len(vis), raw_vi=vis)
                hyp += greedy(P, emb, mask)
            ref = [e for _, e in ps]
            r = {"bleu": round(sacrebleu.corpus_bleu(hyp, [ref]).score, 1),
                 "chrf": round(sacrebleu.corpus_chrf(hyp, [ref]).score, 1), "n": len(ps),
                 "vi_du": [{"vi": v, "ref": e, "hyp": h_} for (v, e), h_ in list(zip(ps, hyp))[:6]]}
            res[f"{name}/{mode}"] = r
            log(f"{name} · {mode}: BLEU {r['bleu']} · chrF {r['chrf']} (n={len(ps)})")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/eval_{a.tag}.json", "w"), ensure_ascii=False, indent=1)
    log(f"### ĐÁNH GIÁ XONG → {OUT}/eval_{a.tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--base", default="ckpt/hybrid_1p3b_chat_v2.pt")
    ap.add_argument("--layer", type=int, default=23, help="lấy ẩn sau khối này của 1.3B")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--tag", default="bridge_in")
    ap.add_argument("--bridge", default="")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--augment", action="store_true", help="v2: Zamba2 nhận token mềm + câu Việt thô (bổ sung, không thay thế)")
    a = ap.parse_args()
    torch.manual_seed(0)
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
