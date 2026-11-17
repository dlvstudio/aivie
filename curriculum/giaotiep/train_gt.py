"""TK_104 · Môn GIAO-TIẾP: train model song-ngữ Việt–Anh TỪ ĐẦU trên kho build_corpus.py.

Áp-dụng bài học vòng Toán: TRỘN NGẪU-NHIÊN mọi loại tài-liệu; tuỳ-chọn LỚN DẦN (--grow: số tầng theo 5
giai-đoạn, tầng mới khởi-tạo đồng-nhất). Khối Transformer dùng lại từ curriculum/loop_lm.py (không lặp).
Đóng gói tài-liệu nối bằng <eos> thành cửa-sổ --ctx token, loss trên mọi token.

    python curriculum/giaotiep/train_gt.py --d 512 --layers 8 --steps 8000 --tag gt_base
    → out/curriculum/gt/<tag>.json (val-loss theo loại tài-liệu) · <tag>.pt · mẫu sinh thử
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loop_lm import Block

D = "data/giaotiep"
SAMPLES = [
    "A: Chào bạn, hôm nay bạn thế nào?\nB:",
    "A: Hi! Are you free this weekend?\nB:",
    "A: Anh ơi, cho em hỏi đường ra ga tàu với ạ.\nB:",
    "Dịch sang tiếng Việt: I'm sorry, I can't come to the party tonight.\n=>",
    "Translate to English: Hôm nay trời mưa to quá, mình ở nhà thôi.\n=>",
    "Ngày xửa ngày xưa, có một cô bé tên là Lan.",
    "Once upon a time, there was a little dog named Max.",
]


class GPT(nn.Module):
    def __init__(self, V, d, layers, heads):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.RMSNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.emb.weight.T

    def grow(self, n, dev):
        d, h = self.emb.weight.shape[1], self.blocks[0].h
        new = []
        for _ in range(n):
            b = Block(d, h).to(dev)
            nn.init.zeros_(b.o.weight); nn.init.zeros_(b.w3.weight)   # đồng-nhất lúc chèn
            self.blocks.append(b); new += list(b.parameters())
        return new


@torch.no_grad()
def generate(m, tok, prompt, dev, n=80, temp=0.7, eos=1):
    ids = tok.encode(prompt).ids
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(n):
        lo = m(torch.tensor([ids[-1024:]], device=dev))[0, -1].float() / temp
        nxt = int(torch.multinomial(torch.softmax(lo, -1).cpu(), 1, generator=g))
        if nxt == eos:
            break
        ids.append(nxt)
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--grow", default="", help="số tầng theo 5 giai-đoạn, vd 4,4,8,8,8")
    ap.add_argument("--only", default="", help="chỉ train các loại này (vd story_vi,dialog_vi) — thí-nghiệm đơn-ngữ")
    ap.add_argument("--data-dir", default=D, help="thư-mục kho (train.jsonl, val.jsonl, tok.json)")
    ap.add_argument("--dialog-repeat", type=int, default=1,
                    help="lặp tài-liệu HỘI-THOẠI (dialog_* + dịch từng lượt) k lần — hội-thoại ít (~1%% token) so với truyện")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{a.data_dir}/tok.json")
    V, EOS = tok.get_vocab_size(), tok.token_to_id("<eos>")
    only = set(a.only.split(",")) if a.only else None

    T0 = time.time()
    train_docs = [json.loads(l) for l in open(f"{a.data_dir}/train.jsonl", encoding="utf-8")]
    train_docs = [r for r in train_docs if not only or r["type"] in only]
    if a.dialog_repeat > 1:
        dl = [r for r in train_docs if r["type"].startswith("dialog") or (r["type"].startswith("dich") and "\n=>\n" not in r["text"])]
        train_docs += dl * (a.dialog_repeat - 1)
        random.Random(a.seed).shuffle(train_docs)
        print(f"  lặp hội-thoại ×{a.dialog_repeat}: +{len(dl) * (a.dialog_repeat - 1)} tài-liệu", flush=True)
    enc = tok.encode_batch([r["text"] for r in train_docs])
    stream = np.fromiter((t for e in enc for t in (e.ids + [EOS])), dtype=np.uint16)
    val = {}
    for r in map(json.loads, open(f"{a.data_dir}/val.jsonl", encoding="utf-8")):
        val.setdefault(r["type"], []).append(r["text"])
    val_streams = {k: np.fromiter((t for e in tok.encode_batch(v) for t in (e.ids + [EOS])), dtype=np.uint16)
                   for k, v in val.items()}
    print(f"[{a.tag}] {len(train_docs)} tài-liệu · {len(stream)/1e6:.1f}M token train · {time.time()-T0:.0f}s", flush=True)

    grow = [int(x) for x in a.grow.split(",")] if a.grow else None
    m = GPT(V, a.d, grow[0] if grow else a.layers, a.heads).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    lr_at = lambda s: a.lr * min(1, (s + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
    torch.backends.cuda.matmul.allow_tf32 = True

    def batch(s, n):
        ix = rng.integers(0, len(s) - a.ctx - 1, n)
        x = np.stack([s[i:i + a.ctx + 1] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(dev)

    @torch.no_grad()
    def val_loss():
        m.eval(); out = {}
        for k, s in val_streams.items():
            if len(s) <= a.ctx + 1:
                continue
            xb = batch(s, 32)
            with amp:
                lo = m(xb[:, :-1])
            out[k] = round(float(F.cross_entropy(lo.float().reshape(-1, V), xb[:, 1:].reshape(-1))), 4)
        m.train(); return out

    log = {"args": vars(a), "evals": []}
    layer_steps, toks = 0, 0
    for step in range(a.steps + 1):
        if grow:
            want = grow[min(4, step * 5 // a.steps)]
            if len(m.blocks) < want:
                opt.add_param_group({"params": m.grow(want - len(m.blocks), dev)})
                print(f"  ⇑ lớn dần @{step}: {len(m.blocks)} tầng", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            v = val_loss()
            log["evals"].append({"step": step, "tokens_M": round(toks / 1e6, 1), "val": v})
            print(f"[{time.time()-T0:.0f}s] bước {step} · {toks/1e6:.0f}M token · val {v}", flush=True)
        if step == a.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        xb = batch(stream, a.bs)
        with amp:
            lo = m(xb[:, :-1])
        loss = F.cross_entropy(lo.float().reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        layer_steps += len(m.blocks); toks += a.bs * a.ctx
    m.eval()
    log["samples"] = {p: generate(m, tok, p, dev, eos=EOS) for p in SAMPLES}
    log["params"] = sum(p.numel() for p in m.parameters())
    log["layer_steps"] = layer_steps
    log["sec"] = round(time.time() - T0)
    os.makedirs("out/curriculum/gt", exist_ok=True)
    json.dump(log, open(f"out/curriculum/gt/{a.tag}.json", "w"), ensure_ascii=False, indent=1)
    torch.save({"args": vars(a), "state": m.state_dict()}, f"out/curriculum/gt/{a.tag}.pt")
    print(f"[{a.tag}] XONG {log['sec']}s · {log['params']/1e6:.1f}M tham-số", flush=True)
    for p, s in log["samples"].items():
        print(f"--- {s}", flush=True)


if __name__ == "__main__":
    main()
