"""TK_104 · Thí-nghiệm BTX (Branch-Train-MiX): mỗi môn MỘT expert, rồi ghép thành mạng chuyên-gia-hỗn-hợp.

Ý-tưởng của user: các mạng chuyên môn riêng (truyện, hội-thoại, dịch, toán…) → đóng băng → ghép qua một tầng định-tuyến
→ train trên dữ-liệu trộn. Bản BTX (Meta 2024): mọi expert RẼ NHÁNH từ CÙNG một model nền (để chung nền ngôn-ngữ),
ghép TRONG TỪNG TẦNG (attention lấy trung-bình, FFN của từng expert thành một lựa-chọn, bộ định-tuyến chọn top-2 cho
TỪNG token).

  prep                       tách kho v2 + toán thành dòng token theo MÔN (out/btx/data/*.npy)
  seed                       nền chung: 2.000 bước trên dữ-liệu trộn
  expert --domain D          rẽ nhánh từ nền, 3.000 bước CHỈ môn D
  dense                      đối-chứng: model đặc cùng cỡ, CÙNG tổng số bước (16.000) trên dữ-liệu trộn
  merge --experts a,b,c --mode frozen|full [--init moe.pt]   ghép + train 2.000 bước trên dữ-liệu trộn
  eval  tag…                 sai-số (loss) theo từng môn

    python curriculum/btx/btx.py prep && python curriculum/btx/btx.py seed && …   (xem run_btx.sh)
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "curriculum"))
sys.path.insert(0, os.path.join(ROOT, "curriculum", "giaotiep"))
from loop_lm import Block, rope  # noqa: E402

OUT = "out/btx"
DOM = {"truyen": ["story_en", "story_vi"], "hoithoai": ["dialog_en", "dialog_vi"], "dich": ["dich_ev", "dich_ve"],
       "toan": ["toan"], "laptrinh": ["laptrinh"]}
CTX, BS, D, L, H = 512, 64, 512, 8, 8


# ── dữ-liệu ──────────────────────────────────────────────────────────────────────────
def prep(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("data/giaotiep/tok.json")
    eos = tok.token_to_id("<eos>")
    os.makedirs(f"{OUT}/data", exist_ok=True)
    for split, path in (("train", "data/giaotiep/train.jsonl"), ("val", "data/giaotiep/val.jsonl")):
        by = {}
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            by.setdefault(r["type"], []).append(r["text"])
        lv = range(1, 6)
        rows = [json.loads(l) for k in lv for l in open(f"data/curriculum/toan/lop{k}_{'train' if split == 'train' else 'test'}.jsonl",
                                                         encoding="utf-8")]
        by["toan"] = [f"Đề: {r['q']}\nGiải: {r['sol']}" for r in rows]
        for dom, types in DOM.items():
            if dom == "laptrinh":                                # dựng riêng bằng prep_code
                continue
            texts = [t for ty in types for t in by.get(ty, [])]
            ids = np.fromiter((i for e in tok.encode_batch(texts) for i in (e.ids + [eos])), dtype=np.uint16)
            np.save(f"{OUT}/data/{dom}_{split}.npy", ids)
            print(f"{dom:9s} {split:5s} {len(texts):7d} tài-liệu · {len(ids)/1e6:6.1f}M token", flush=True)


def prep_code(a):
    """Môn LẬP-TRÌNH (tiếng Anh): Magicoder-OSS-Instruct-75K (nhiều ngôn-ngữ) + CodeFeedback (50K đầu)."""
    from datasets import load_dataset
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("data/giaotiep/tok.json")
    eos = tok.token_to_id("<eos>")
    texts = []
    for r in load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train"):
        texts.append(f"### Task ({r['lang']})\n{r['problem'].strip()}\n### Solution\n{r['solution'].strip()}")
    for i, r in enumerate(load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)):
        if i >= 50000:
            break
        texts.append(f"### Task ({r['lang']})\n{r['query'].strip()}\n### Solution\n{r['answer'].strip()}")
    import random
    random.Random(104).shuffle(texts)
    for split, part in (("val", texts[:1500]), ("train", texts[1500:])):
        ids = np.fromiter((i for e in tok.encode_batch(part) for i in (e.ids + [eos])), dtype=np.uint16)
        np.save(f"{OUT}/data/laptrinh_{split}.npy", ids)
        print(f"laptrinh  {split:5s} {len(part):7d} tài-liệu · {len(ids)/1e6:6.1f}M token", flush=True)


class Data:
    def __init__(self, doms, split="train"):
        self.s = {d: np.load(f"{OUT}/data/{d}_{split}.npy", mmap_mode="r") for d in doms}
        self.doms = list(doms)
        self.rng = np.random.default_rng(0)
        w = np.array([len(self.s[d]) ** 0.5 for d in self.doms])     # trộn theo CĂN BẬC HAI kích-thước môn
        self.w = w / w.sum()

    def batch(self, n=BS, dom=None, dev="cuda"):
        ds = [dom] * n if dom else self.rng.choice(self.doms, n, p=self.w)
        rows = []
        for d in ds:
            s = self.s[d]
            i = int(self.rng.integers(0, len(s) - CTX - 1))
            rows.append(np.asarray(s[i:i + CTX + 1], dtype=np.int64))
        return torch.from_numpy(np.stack(rows)).to(dev)


# ── model ────────────────────────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, V, blocks):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(D)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.aux = 0.0

    def forward(self, ids):
        x = self.emb(ids)
        self.aux = 0.0
        for b in self.blocks:
            x = b(x)
            self.aux = self.aux + getattr(b, "aux", 0.0)
        return self.norm(x) @ self.emb.weight.T


class FFN(nn.Module):
    def __init__(self, w12, w3):
        super().__init__()
        self.w12, self.w3 = w12, w3

    def forward(self, x):
        g, u = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(g) * u)


class MoEBlock(nn.Module):
    """Attention + chuẩn-hoá = TRUNG-BÌNH các expert; FFN = E expert; bộ định-tuyến chọn top-k cho từng token."""
    def __init__(self, blocks, k=2):
        super().__init__()
        b0 = blocks[0]
        self.h, self.hd, self.k = b0.h, b0.hd, k
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.qkv, self.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        with torch.no_grad():
            for name in ("n1", "n2", "qkv", "o"):
                for p_dst, *p_src in zip(getattr(self, name).parameters(), *[getattr(b, name).parameters() for b in blocks]):
                    p_dst.copy_(torch.stack([p.data for p in p_src]).mean(0))
        self.experts = nn.ModuleList([FFN(b.w12, b.w3) for b in blocks])
        self.router = nn.Linear(D, len(blocks), bias=True)
        nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)   # khởi-đầu: chia đều
        self.aux = 0.0

    def add_expert(self, block):
        """HỌC TÍCH-LUỸ: cắm thêm một expert mới (expert cũ giữ nguyên)."""
        self.experts.append(FFN(block.w12, block.w3))
        old_w, old_b = self.router.weight.data, self.router.bias.data
        self.router = nn.Linear(D, len(self.experts), bias=True).to(old_w.device)
        with torch.no_grad():                                # expert mới bắt đầu TẮT (lệch −10) → cắm vào không đổi đầu ra
            self.router.weight.zero_(); self.router.weight[: old_w.shape[0]] = old_w
            self.router.bias.fill_(-10.0); self.router.bias[: old_b.shape[0]] = old_b

    def forward(self, x):
        b, l, d = x.shape
        q, k_, v = self.qkv(self.n1(x)).view(b, l, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k_), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, l, d))
        hN = self.n2(x)
        logits = self.router(hN).float()                      # (b,l,E)
        p = logits.softmax(-1)
        top, idx = p.topk(self.k, -1)
        w = torch.zeros_like(p).scatter(-1, idx, top / top.sum(-1, keepdim=True))
        E = len(self.experts)
        frac = F.one_hot(idx[..., 0], E).float().mean((0, 1))     # cân-bằng tải (Switch)
        self.aux = E * (frac * p.mean((0, 1))).sum()
        self.last_w = w.detach()
        out = sum(w[..., e:e + 1].to(hN.dtype) * self.experts[e](hN) for e in range(E))
        return x + out


def new_dense(V):
    return GPT(V, [Block(D, H) for _ in range(L)])


def load_dense(path, V, dev):
    m = new_dense(V)
    m.load_state_dict(torch.load(path, map_location="cpu")["state"])
    return m.to(dev)


# ── train ────────────────────────────────────────────────────────────────────────────
def train(m, data, steps, lr, tag, dev, dom=None, params=None, aux_w=0.01, log_every=500, val=None, best_on=None):
    """best_on = tên môn → DỪNG SỚM: giữ trạng-thái có sai-số tốt nhất trên môn đó (đánh-giá mỗi log_every bước)."""
    best = (float("inf"), None)
    params = [p for p in (params if params is not None else m.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    T0 = time.time()
    hist = []
    for s in range(steps):
        lr_s = lr * min(1, (s + 1) / 100) * min(1.0, (steps - s) / (0.2 * steps))   # phẳng, giảm 20% cuối
        for g in opt.param_groups:
            g["lr"] = lr_s
        xb = data.batch(dom=dom, dev=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(xb[:, :-1])
        loss = F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), xb[:, 1:].reshape(-1)) + aux_w * m.aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if s % log_every == 0 or s == steps - 1:
            v = evaluate(m, val, dev) if val else {}
            hist.append({"step": s, "loss": round(float(loss), 4), "val": v})
            print(f"[{tag}] bước {s}/{steps} · loss {float(loss):.3f} · {time.time()-T0:.0f}s · {v}", flush=True)
            if best_on and v.get(best_on, 1e9) < best[0]:
                best = (v[best_on], {k: t.detach().clone() for k, t in m.state_dict().items()}, s)
    if best_on and best[1] is not None:
        m.load_state_dict(best[1])
        print(f"[{tag}] dừng sớm: giữ bước {best[2]} ({best_on} {best[0]})", flush=True)
        hist.append({"best_step": best[2]})
    return hist


@torch.no_grad()
def evaluate(m, val, dev, n=24, reps=4):
    m.eval()
    out = {}
    for d in val.doms:
        ls = []
        for r in range(reps):
            xb = val.batch(n, dom=d, dev=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(xb[:, :-1])
            ls.append(float(F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), xb[:, 1:].reshape(-1))))
        out[d] = round(sum(ls) / len(ls), 4)
    m.train()
    return out


def save(m, tag, extra=None):
    torch.save({"state": m.state_dict(), **(extra or {})}, f"{OUT}/{tag}.pt")


def vocab():
    from tokenizers import Tokenizer
    return Tokenizer.from_file("data/giaotiep/tok.json").get_vocab_size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--domain", default="")
    ap.add_argument("--experts", default="")
    ap.add_argument("--mode", default="frozen", choices=["frozen", "full"])
    ap.add_argument("--init", default="", help="merge: bắt đầu từ MoE có sẵn rồi CẮM THÊM expert")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("tags", nargs="*")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    if a.cmd == "prep":
        return prep(a)
    if a.cmd == "prep_code":
        return prep_code(a)
    V = vocab()
    allD = list(DOM)
    val = Data(allD, "val")
    log = {"cmd": a.cmd, "args": vars(a)}
    if a.cmd == "seed":
        m = new_dense(V).to(dev)
        log["hist"] = train(m, Data(allD), a.steps or 2000, 6e-4, "seed", dev, val=val)
        save(m, "seed")
    elif a.cmd == "expert":
        m = load_dense(f"{OUT}/seed.pt", V, dev)
        log["hist"] = train(m, Data([a.domain]), a.steps or 3000, 3e-4, f"exp_{a.domain}", dev, dom=a.domain, val=val,
                            log_every=250, best_on=a.domain)
        save(m, f"exp_{a.domain}")
    elif a.cmd == "dense":
        m = load_dense(f"{OUT}/seed.pt", V, dev)                      # cùng nền → tổng bước = 2.000 + steps
        log["hist"] = train(m, Data(allD), a.steps or 17000, 3e-4, "dense", dev, val=val, log_every=2000)
        save(m, "dense")
    elif a.cmd == "merge":
        names = a.experts.split(",")
        exps = [load_dense(f"{OUT}/exp_{n}.pt", V, dev) for n in names]
        if a.init:                                                   # CẮM THÊM expert vào MoE đã train
            moe = GPT(V, [MoEBlock([e.blocks[i] for e in exps[:-1]]) for i in range(L)]).to(dev)
            moe.load_state_dict(torch.load(f"{OUT}/{a.init}.pt", map_location="cpu")["state"])
            for i, blk in enumerate(moe.blocks):
                blk.add_expert(exps[-1].blocks[i])
        else:
            moe = GPT(V, [MoEBlock([e.blocks[i] for e in exps]) for i in range(L)]).to(dev)
            with torch.no_grad():                                    # bảng-từ & chuẩn-hoá cuối = trung-bình
                moe.emb.weight.copy_(torch.stack([e.emb.weight for e in exps]).mean(0))
                moe.norm.weight.copy_(torch.stack([e.norm.weight for e in exps]).mean(0))
        moe.to(dev)
        if a.mode == "frozen" or a.init:                              # ĐÓNG BĂNG mọi FFN expert
            for blk in moe.blocks:
                for ex in blk.experts:
                    for p in ex.parameters():
                        p.requires_grad_(False)
        doms = allD if not a.init else allD
        ntr = sum(p.numel() for p in moe.parameters() if p.requires_grad)
        print(f"[{a.tag}] MoE {len(names)} expert · {sum(p.numel() for p in moe.parameters())/1e6:.0f}M tham-số · "
              f"đang học {ntr/1e6:.1f}M", flush=True)
        log["before"] = evaluate(moe, Data(allD, "val"), dev, n=32, reps=8)
        print(f"[{a.tag}] ngay sau khi ghép: {log['before']}", flush=True)
        trd = Data([d for d in allD if d in names] if not a.init else [d for d in allD if d in names])
        log["hist"] = train(moe, trd, a.steps or 2000, 2e-4, a.tag, dev, val=val, log_every=500)
        # định-tuyến: mỗi môn dùng expert nào
        route = {}
        with torch.no_grad():
            for d in allD:
                xb = val.batch(16, dom=d, dev=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    moe(xb[:, :-1])
                w = torch.stack([blk.last_w.mean((0, 1)) for blk in moe.blocks]).mean(0)
                route[d] = {n: round(float(x), 3) for n, x in zip(names, w)}
        log["route"] = route
        print(f"[{a.tag}] định-tuyến: {route}", flush=True)
        save(moe, a.tag, {"experts": names})
    elif a.cmd == "eval":
        for t in a.tags:
            print(t, json.load(open(f"{OUT}/{t}.json"))["final"], flush=True)
        return
    m_ = moe if a.cmd == "merge" else m
    log["final"] = evaluate(m_, Data(allD, "val"), dev, n=32, reps=8)   # CÙNG bộ đoạn cho mọi model
    print(f"[{a.tag or a.cmd}] CUỐI: {log['final']}", flush=True)
    json.dump(log, open(f"{OUT}/{a.tag or (a.cmd + ('_' + a.domain if a.domain else ''))}.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
