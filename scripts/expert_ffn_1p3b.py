"""TK_104 §13.5 · CẮM KỸ-NĂNG MỚI vào model 1.3B mà KHÔNG QUÊN: chèn tầng FFN chuyên-gia vào thân Mamba+attention
ĐÓNG BĂNG (bài-học BTX: expert đóng băng + định-tuyến = học tích-luỹ không quên).

HybridLM 1.3B = 48 khối (Mamba, attention ở 7,15,…,47), KHÔNG có FFN → chèn 6 tầng GatedFFN sau mỗi khối attention
(thân 2,5B TK_107 = 96 khối, 12 attention → 12 tầng, mỗi tầng nửa cỡ — xem set_pos/hid_eff):
    x ← x + g(x) · FFN(RMSNorm(x))      g = sigmoid(cổng(RMSNorm(x)))  ∈ (0,1), từng token
  A  thân đóng băng, CHỈ FFN học code (cổng mở hẳn); FFN khởi-tạo lớp ra = 0 → lúc đầu y hệt model gốc
  B  FFN cũng đóng băng, CHỈ cổng học trên dữ-liệu TRỘN (code + hội-thoại/trợ-lý) → code mở, tiếng Việt đóng

    PYTHONPATH=scripts:src python scripts/expert_ffn_1p3b.py prep
    … trainA | trainB | loss | mbpp --which base|A|B | evalv1   (xem run_expert.sh)
"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, tempfile, time
sys.path[:0] = ["scripts", "src"]
import torch
import torch.utils.checkpoint                  # NGC torch không tự nạp submodule này (Mac thì có)
import torch.nn as nn
import torch.nn.functional as F

BASE = "ckpt/hybrid_1p3b_chat.pt"
OUT = "ckpt/expert_code"
POS = [7, 15, 23, 31, 39, 47]                                 # sau mỗi khối attention — load_trunk cập nhật TẠI CHỖ
HID = 4096                                                    # theo cfg (thân 96 khối TK_107 → 12 điểm)
REF_N = 6                                                     # số điểm cắm của thân 1.3B gốc: mọi --hid tính theo mốc này


def set_pos(c):
    """điểm cắm = các khối attention của thân (sửa POS tại chỗ → `from expert_ffn_1p3b import POS` vẫn thấy)."""
    import train_hybrid_scale as ths
    POS[:] = ths.attn_layers_of(c)


def hid_eff(hid):
    """cỡ ẩn THẬT của mỗi FFN: --hid là cỡ quy về 6 điểm; 12 điểm → mỗi FFN nửa cỡ, tổng tham số expert giữ nguyên."""
    return max(256, hid * REF_N // len(POS))


class CodeLoRA(nn.Module):
    """TK_108 C8 (22/10): LoRA lên KHỐI TRỘN của thân (wq/wk/wv/wo attention · in_proj/out_proj Mamba) — chỗ thân "đọc và nối"
    các phần của đề; expert FFN không chạm tới được (C7: Python/JS nghiệp vụ vẫn 0% dù loss C7 giảm một nửa).
    W x + (B A x)·(α/r), B khởi 0 → lúc đầu y hệt thân. Cờ CodeLoRA.on riêng (KHÔNG dùng chung cờ LoRA đọc ảnh của vlm_1p3b)."""
    on = True

    def __init__(self, base, r=32, alpha=None):
        super().__init__()
        import math
        self.base, self.r, self.s = base, r, (alpha or r) / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        if not CodeLoRA.on:
            return self.base(x)
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.s

    @property
    def weight(self):                             # lõi CUDA mamba_ssm đọc THẲNG in_proj.weight / out_proj.weight
        w = self.base.weight
        return w if not CodeLoRA.on else w + (self.B @ self.A).to(w.dtype) * self.s

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


MIXERS = ("wq", "wk", "wv", "wo", "in_proj", "out_proj")


def add_code_lora(trunk, r, start=0, attn_only=False):
    """cắm CodeLoRA lên lớp chiếu của các khối từ `start` → danh sách tham số học."""
    ps = []
    for i, blk in enumerate(trunk.blocks):
        if i < start:
            continue
        for nm in MIXERS:
            base = getattr(blk, nm, None)
            if not isinstance(base, nn.Linear) or (attn_only and nm in ("in_proj", "out_proj")):
                continue
            setattr(blk, nm, CodeLoRA(base, r))
            ps += [getattr(blk, nm).A, getattr(blk, nm).B]
    return ps


def code_lora_state(trunk):
    return {f"{i}.{nm}.{k}": getattr(getattr(b, nm), k).detach().to(torch.bfloat16).cpu()
            for i, b in enumerate(trunk.blocks) for nm in MIXERS
            if isinstance(getattr(b, nm, None), CodeLoRA) for k in ("A", "B")}


def load_code_lora(trunk, sd):
    """cắm lại CodeLoRA theo đúng khoá đã lưu ("<khối>.<lớp>.A/B"), r đọc từ A."""
    for k in sorted({k.rsplit(".", 1)[0] for k in sd}):
        i, nm = k.split(".")
        blk = trunk.blocks[int(i)]
        base = getattr(blk, nm)
        A = sd[f"{k}.A"]
        lo = CodeLoRA(base, A.shape[0])
        with torch.no_grad():
            lo.A.copy_(A.to(lo.A.dtype)); lo.B.copy_(sd[f"{k}.B"].to(lo.B.dtype))
        setattr(blk, nm, lo)
    return trunk


class GatedFFN(nn.Module):
    def __init__(self, d, hid=None, exact=False):
        hid = hid if exact else hid_eff(hid or HID)              # exact: cỡ ẩn THẬT (expert lớn đọc từ file)
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.w12 = nn.Linear(d, 2 * hid, bias=False)
        self.w3 = nn.Linear(hid, d, bias=False)
        nn.init.zeros_(self.w3.weight)                        # đồng-nhất lúc chèn
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -2.0)
        self.use_gate = False

    def forward(self, x):
        h = self.norm(x)
        g_, u = self.w12(h).chunk(2, -1)
        y = self.w3(F.silu(g_) * u)
        if self.use_gate:
            self.g = torch.sigmoid(self.gate(h).float()).to(y.dtype)
            y = self.g * y
        return x + y


def load_trunk(dev, kernel=True):
    import train_hybrid_scale as ths
    if not kernel:
        ths.HAVE_KERNEL = False
    obj = torch.load(BASE, map_location="cpu", weights_only=False, mmap=True)
    c = obj["cfg"]
    m = ths.build_from_cfg(c, obj["state"]["emb.weight"].shape[0])       # đúng attn_layers + rope_base của thân
    m.load_state_dict(obj["state"])
    set_pos(c)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(dev), c


class WithExperts(nn.Module):
    def __init__(self, trunk, d):
        super().__init__()
        self.t = trunk
        self.ad = nn.ModuleDict({str(i): GatedFFN(d) for i in POS})

    def forward(self, ids):
        t = self.t
        x = t.emb(ids)
        for i, (blk, nm) in enumerate(zip(t.blocks, t.norms)):
            if i < POS[0]:
                with torch.no_grad():                            # trước tầng chuyên-gia đầu: khỏi lưu gradient
                    x = x + blk(nm(x))
            else:
                x = x + torch.utils.checkpoint.checkpoint(lambda z, b=blk, n=nm: b(n(z)), x, use_reentrant=False)
            if str(i) in self.ad:
                x = self.ad[str(i)](x)
        return t.norm_f(x) @ t.emb.weight.T


# ── dữ-liệu ──────────────────────────────────────────────────────────────────────────
def prep(a):
    from datasets import load_dataset
    rows = []
    for r in load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train"):
        rows.append({"messages": [{"role": "user", "content": r["problem"].strip()},
                                  {"role": "assistant", "content": r["solution"].strip()}], "src": "code"})
    random.Random(104).shuffle(rows)
    os.makedirs("data/code_sft", exist_ok=True)
    for name, part in (("val", rows[:400]), ("train", rows[400:30400])):
        with open(f"data/code_sft/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    src = "data/sft_chat/chat_mix.jsonl"
    if not os.path.exists(src):                                # TK_107 P3: kho chat cũ không còn → lấy chat của sft_v2
        src = "data/sft_v2/sft_v2.jsonl"
    chat = [json.loads(l) for l in open(src, encoding="utf-8")]
    chat = [r for r in chat if not str(r.get("source", "")).startswith("math")] or chat
    random.Random(7).shuffle(chat)
    for name, part in (("chat_val", chat[:400]), ("chat_mix", chat[400:12400])):
        with open(f"data/code_sft/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("code train 30000 · val 400 · chat_mix 12000 · chat_val 400")


def load_ex(path, tok, max_len=1536):
    from train_hybrid_sft import build_example
    out = []
    for l in open(path, encoding="utf-8"):
        ids, lab = build_example(json.loads(l)["messages"], tok, max_len)
        if any(x != -100 for x in lab):
            out.append((ids, lab))
    return out


def batch(exs, rng, bs, dev):
    pick = [exs[rng.randrange(len(exs))] for _ in range(bs)]
    L = max(len(i) for i, _ in pick)
    X = torch.zeros(bs, L, dtype=torch.long); Y = torch.full((bs, L), -100, dtype=torch.long)
    for j, (i, l) in enumerate(pick):
        X[j, :len(i)] = torch.tensor(i); Y[j, :len(l)] = torch.tensor(l)
    return X.to(dev), Y.to(dev)


def lm_loss(m, X, Y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(X[:, :-1])
    return F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), Y[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def val_loss(m, exs, dev, n=200, bs=4):
    m.eval(); rng = random.Random(0); tot = 0.0
    for _ in range(n // bs):
        X, Y = batch(exs, rng, bs, dev)
        tot += float(lm_loss(m, X, Y))
    m.train()
    return round(tot / (n // bs), 4)


def tok_():
    from nn_aivie.core.tokenizer import BPETokenizer
    return BPETokenizer.load("checkpoints/bpe_v1.json")


def train(a):
    dev = "cuda"
    tok = tok_()
    trunk, c = load_trunk(dev)
    m = WithExperts(trunk, c["d"]).to(dev)
    code_tr, code_va = load_ex("data/code_sft/train.jsonl", tok), load_ex("data/code_sft/val.jsonl", tok)
    chat_va = load_ex("data/code_sft/chat_val.jsonl", tok)
    log = {}
    groups = None
    if a.cmd == "trainA":
        params = [p for k, p in m.ad.named_parameters() if not k.endswith(("gate.weight", "gate.bias"))]
        data, steps, lr = [code_tr], a.steps or 1200, 3e-4
    elif a.cmd == "trainVI":
        # CẦU NỐI tiếng Việt: FFN + cổng học tiếp (LR nhỏ cho FFN) trên trộn ĐỀU code-tiếng-Việt / code-tiếng-Anh / hội-thoại
        m.ad.load_state_dict(torch.load(a.adapters))
        for blk in m.ad.values():
            blk.use_gate = True
        vi_all = load_ex("data/code_sft/vi_code.jsonl", tok)
        vi_tr, vi_va = vi_all[100:], vi_all[:100]
        chat_tr = load_ex("data/code_sft/chat_mix.jsonl", tok)
        ffn = [p for k, p in m.ad.named_parameters() if ".gate." not in k]
        gate = [p for k, p in m.ad.named_parameters() if ".gate." in k]
        params = ffn + gate
        groups = [{"params": ffn, "lr": 5e-5, "base": 5e-5}, {"params": gate, "lr": 1e-3, "base": 1e-3}]
        data, steps, lr = [vi_tr, code_tr, chat_tr], a.steps or 600, 1e-3
        log["vi_before"] = val_loss(m, vi_va, dev)
    else:
        m.ad.load_state_dict(torch.load(a.adapters or f"{OUT}/A.pt"))
        for blk in m.ad.values():
            blk.use_gate = True
        for k, p in m.ad.named_parameters():
            p.requires_grad_(k.split(".", 1)[1].startswith("gate"))
        params = [p for p in m.ad.parameters() if p.requires_grad]
        chat_tr = load_ex("data/code_sft/chat_mix.jsonl", tok)
        data, steps, lr = [code_tr, chat_tr], a.steps or 500, 1e-3
    log["before"] = {"code": val_loss(m, code_va, dev), "chat": val_loss(m, chat_va, dev)}
    print(f"[{a.cmd}] trước: {log['before']} · học {sum(p.numel() for p in params)/1e6:.1f}M tham-số", flush=True)
    opt = torch.optim.AdamW(groups or params, lr=lr, weight_decay=0.0 if a.cmd == "trainB" else 0.1)
    rng = random.Random(1); T0 = time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = g.get("base", lr) * min(1, (s + 1) / 50) * min(1.0, (steps - s) / (0.2 * steps))
        loss = 0.0
        for _ in range(a.accum):
            X, Y = batch(rng.choice(data), rng, a.bs, dev)       # B: trộn đều code / tiếng Việt
            l = lm_loss(m, X, Y) / a.accum
            l.backward(); loss += float(l)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s % 100 == 0 or s == steps - 1:
            print(f"[{a.cmd}] bước {s}/{steps} · loss {loss:.3f} · {time.time()-T0:.0f}s", flush=True)
    log["after"] = {"code": val_loss(m, code_va, dev), "chat": val_loss(m, chat_va, dev)}
    if a.cmd == "trainVI":
        log["vi_after"] = val_loss(m, vi_va, dev)
    if a.cmd in ("trainB", "trainVI"):                                       # cổng mở bao nhiêu trên code vs tiếng Việt
        gs = {}
        with torch.no_grad():
            for name, exs in (("code", code_va), ("chat", chat_va)):
                X, _ = batch(exs, random.Random(3), 4, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m(X[:, :-1])
                gs[name] = round(float(torch.stack([b.g.float().mean() for b in m.ad.values()]).mean()), 3)
        log["gate_mean"] = gs
    print(f"[{a.cmd}] sau: {log['after']} {log.get('gate_mean', '')}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    name = a.tag or {"trainA": "A", "trainB": "B", "trainVI": "VI"}[a.cmd]
    torch.save(m.ad.state_dict(), f"{OUT}/{name}.pt")
    json.dump(log, open(f"{OUT}/{a.cmd}_{name}.json", "w"), indent=1)


def train_pt(a):
    """BỒI ĐẮP KIẾN-THỨC CODE: dòng token đóng gói (data/code_pt: py_*.npy code Python lọc kiểu Phi-1 + inst_000.npy bài tập
    có lời-giải, trộn 3:1), cửa-sổ 2.048, thân 1.3B ĐÓNG BĂNG, chỉ FFN chuyên-gia học. Lịch WSD (phẳng, giảm 15% cuối).
    Ảnh chụp ở các mốc --snap (token) → tự chấm MBPP (cổng mở hẳn) bằng tiến-trình riêng trong lúc train chạy tiếp."""
    import glob, numpy as np
    dev = "cuda"
    trunk, c = load_trunk(dev)
    m = WithExperts(trunk, c["d"]).to(dev)
    params = [p for k, p in m.ad.named_parameters() if ".gate." not in k]
    print(f"[pt] FFN chuyên-gia {sum(p.numel() for p in params)/1e6:.0f}M tham-số · thân {BASE}", flush=True)
    py = [np.load(f, mmap_mode="r") for f in sorted(glob.glob("data/code_pt/py_*.npy"))]
    inst = np.load("data/code_pt/inst_000.npy", mmap_mode="r")
    va = [inst[-2_000_000:], py[-1][-2_000_000:]]                  # đuôi giữ riêng để đo
    inst_tr, py_tr = inst[:-2_000_000], py[:-1] + [py[-1][:-2_000_000]]
    rng = np.random.default_rng(0)
    L = a.ctx + 1

    def window(arr):
        i = int(rng.integers(0, len(arr) - L))
        return np.asarray(arr[i:i + L], dtype=np.int64)

    def get_batch(bs):
        rows = []
        for _ in range(bs):
            if rng.random() < 0.25:
                rows.append(window(inst_tr))
            else:
                w = np.array([len(x) for x in py_tr], dtype=np.float64)
                rows.append(window(py_tr[int(rng.choice(len(py_tr), p=w / w.sum()))]))
        return torch.from_numpy(np.stack(rows)).to(dev)

    @torch.no_grad()
    def vloss():
        m.eval(); out = []
        g = np.random.default_rng(5)
        for arr in va:
            tot = 0.0
            for _ in range(8):
                xb = torch.from_numpy(np.stack([np.asarray(arr[j:j + L], dtype=np.int64)
                                                for j in g.integers(0, len(arr) - L, 4)])).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = m(xb[:, :-1])
                tot += float(F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), xb[:, 1:].reshape(-1)))
            out.append(round(tot / 8, 4))
        m.train()
        return {"inst": out[0], "py": out[1]}

    steps = a.tokens // (a.bs * a.ctx)
    snaps = sorted(int(x) for x in a.snap.split(",") if x)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    os.makedirs(OUT, exist_ok=True)
    log = {"args": vars(a), "hist": [], "start": vloss()}
    print(f"[pt] {steps} bước × {a.bs}×{a.ctx} = {a.tokens/1e6:.0f}M token · trước: {log['start']}", flush=True)
    T0, tok_seen, si = time.time(), 0, 0
    for s in range(steps):
        lr = a.lr * min(1, (s + 1) / 300) * min(1.0, (steps - s) / (0.15 * steps))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        xb = get_batch(a.bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(xb[:, :-1])
        loss = F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), xb[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        tok_seen += a.bs * a.ctx
        if s % 200 == 0:
            el = time.time() - T0
            print(f"[pt] bước {s}/{steps} · {tok_seen/1e6:.0f}M token · loss {float(loss):.3f} · "
                  f"{tok_seen/max(el,1):.0f} tok/s · còn ~{(steps-s)*el/max(s,1)/3600:.1f} giờ", flush=True)
        if s % 2000 == 0 and s:
            v = vloss(); log["hist"].append({"tokens": tok_seen, "val": v}); print(f"[pt] val {v}", flush=True)
            json.dump(log, open(f"{OUT}/pt_log.json", "w"), indent=1)
        if si < len(snaps) and tok_seen >= snaps[si]:
            path = f"{OUT}/pt_{snaps[si]//1_000_000}M.pt"
            torch.save(m.ad.state_dict(), path)
            subprocess.Popen(f"PYTHONPATH=scripts:src python -W ignore -u scripts/expert_ffn_1p3b.py mbpp --which A "
                             f"--adapters {path} --hid {HID} --base {BASE} --tag pt_{snaps[si]//1_000_000}M "
                             f"> /workspace/mbpp_pt_{snaps[si]//1_000_000}M.log 2>&1", shell=True)
            print(f"[pt] ẢNH CHỤP {path} → đang chấm MBPP song-song", flush=True)
            si += 1
    torch.save(m.ad.state_dict(), f"{OUT}/pt_final.pt")
    log["final"] = vloss()
    json.dump(log, open(f"{OUT}/pt_log.json", "w"), indent=1)
    print(f"[pt] XONG · {log['final']}", flush=True)


# ── suy-luận từng bước (Runner thuần-PyTorch) có tầng chuyên-gia ─────────────────────
def adapter_runner(which, dev, path=None):
    import hybrid_infer as hi
    m, c = load_trunk(dev, kernel=False)
    m.eval()
    ad = None
    if which in ("A", "B"):
        ad = nn.ModuleDict({str(i): GatedFFN(c["d"]) for i in POS}).to(dev)
        ad.load_state_dict(torch.load(path or f"{OUT}/{which}.pt", map_location=dev))
        for blk in ad.values():
            blk.use_gate = which == "B"
        ad.eval()

    class R(hi.Runner):
        @torch.no_grad()
        def prefill(self, ids, all_logits=False, state=None, vis=None):   # vis: hàm sinh chung (có ảnh) truyền vào — runner chữ bỏ qua
            x = self.m.emb(torch.tensor([ids], dtype=torch.long, device=self.dev))
            pos0 = state["pos"] if state else 0
            olds = state["layers"] if state else [None] * len(self.m.blocks)
            sts = []
            for i, (blk, nm, st) in enumerate(zip(self.m.blocks, self.m.norms, olds)):
                o, st2 = (hi.attn_prefill(blk, nm(x), st, pos0) if isinstance(blk, hi.ScaledAttn)
                          else hi.mamba_prefill(blk, nm(x), st))
                x = x + o; sts.append(st2)
                if ad is not None and str(i) in ad:
                    x = ad[str(i)](x)
            lo = self._logits(x if all_logits else x[:, -1:])
            return lo[0], {"layers": sts, "pos": pos0 + len(ids)}

        @torch.no_grad()
        def step(self, tok, state):
            x = self.m.emb(torch.tensor([[tok]], dtype=torch.long, device=self.dev))
            new, pos = [], state["pos"]
            for i, (blk, nm, st) in enumerate(zip(self.m.blocks, self.m.norms, state["layers"])):
                o, st2 = (hi.attn_step(blk, nm(x), st, pos) if isinstance(blk, hi.ScaledAttn)
                          else hi.mamba_step(blk, nm(x), st))
                x = x + o; new.append(st2)
                if ad is not None and str(i) in ad:
                    x = ad[str(i)](x)
            return self._logits(x)[0, -1], {"layers": new, "pos": pos + 1}
    return R(m, dev)


def answer(r, tok, q, n=400):
    from nn_aivie.core.vocab import Vocabulary
    eos = tok.token_to_id["<EOS>"]
    ids = tok.encode("Hỏi: " + q.strip() + "\n") + tok.encode("Đáp: ")
    out = list(r.generate(ids, n=n, temp=0, rep_pen=1.1, eos=eos))
    return Vocabulary(tok).detokenize(out).strip()


def mbpp(a):
    """MBPP (Python có test): model viết hàm → chạy test thật → % bài QUA."""
    from datasets import load_dataset
    dev = "cuda"
    tok = tok_()
    r = adapter_runner(a.which, dev, a.adapters or None)
    ds = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))[: a.n]
    ok, rows = 0, []
    for k, ex in enumerate(ds):
        q = f"{ex['text']}\nYour code should pass this test:\n{ex['test_list'][0]}\nWrite the Python function."
        resp = answer(r, tok, q)
        m_ = re.search(r"```(?:python)?\n(.*?)```", resp, re.S)
        code = m_.group(1) if m_ else resp
        prog = code + "\n\n" + "\n".join(ex["test_list"]) + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog)
        try:
            p_ = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10)
            passed = p_.returncode == 0
        except subprocess.TimeoutExpired:
            passed = False
        os.unlink(f.name)
        ok += passed
        rows.append({"task": ex["task_id"], "pass": passed, "resp": resp[:800]})
        if k % 20 == 0:
            print(f"[mbpp {a.which}] {k}/{len(ds)} · qua {ok}", flush=True)
    res = {"which": a.which, "adapters": a.adapters, "n": len(ds), "pass": ok, "pass_rate": round(ok / len(ds), 3),
           "def_rate": round(sum("def " in x["resp"] for x in rows) / len(rows), 3)}
    print(json.dumps(res), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({**res, "rows": rows}, open(f"{OUT}/mbpp_{a.tag or a.which}.json", "w"), ensure_ascii=False, indent=1)


ASK = ["Cho tôi đoạn code Python đệ quy tính giai thừa.",
       "Viết hàm Python kiểm tra một chuỗi có phải là chuỗi đối xứng (palindrome) không, giải thích bằng tiếng Việt.",
       "Làm sao để đọc một file CSV trong Python và tính trung bình một cột?",
       "Viết hàm Python sắp xếp danh sách theo thuật toán sắp xếp nổi bọt.",
       "Soạn giúp tôi email xin nghỉ phép 2 ngày gửi sếp vì việc gia đình.",
       "Thủ đô của Việt Nam là gì?"]


def ask(a):
    """Hỏi thử các câu tiếng Việt (code + không phải code) → xem cổng mở đúng chỗ, trả lời tiếng Việt."""
    tok = tok_()
    r = adapter_runner(a.which, "cuda", a.adapters or None)
    out = []
    for q in ASK:
        ans = answer(r, tok, q, n=300)
        out.append({"q": q, "a": ans})
        print("HỎI:", q, "\nĐÁP:", ans[:700], "\n" + "=" * 60, flush=True)
    json.dump(out, open(f"{OUT}/ask_{a.tag or a.which}.json", "w"), ensure_ascii=False, indent=1)


def evalv1(a):
    """eval_v1 (bộ đánh-giá sạch tiếng Việt) với tầng chuyên-gia → kiểm KHÔNG QUÊN (chấm bằng eval_clean.py judge)."""
    dev = "cuda"
    tok = tok_()
    r = adapter_runner(a.which, dev, a.adapters or None)
    qs = [json.loads(l) for l in open("data/eval_clean/eval_v1.jsonl", encoding="utf-8")]
    tag = a.tag or f"expert{a.which}"
    with open(f"out/evalclean_{tag}.jsonl", "w", encoding="utf-8") as f:
        for q in qs:
            t0 = time.time()
            ans = answer(r, tok, q["q"], n=500)
            f.write(json.dumps({**q, "tag": tag, "answer": ans, "sec": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
    print(f"[evalv1 {a.which}] xong → out/evalclean_{tag}.jsonl", flush=True)


def main():
    global BASE, HID
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "trainA", "trainB", "trainPT", "trainVI", "mbpp", "evalv1", "ask"])
    ap.add_argument("--base", default=BASE, help="checkpoint thân 1.3B (đóng băng)")
    ap.add_argument("--hid", type=int, default=HID, help="cỡ ẩn của mỗi FFN chuyên-gia (quy về 6 điểm cắm, xem hid_eff)")
    ap.add_argument("--adapters", default="", help="file trọng-số FFN chuyên-gia để nạp")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tokens", type=int, default=1_100_000_000)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--snap", default="300000000,600000000")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--which", default="B", choices=["base", "A", "B"])
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    BASE, HID = a.base, a.hid
    torch.manual_seed(0)
    if a.cmd == "prep":
        return prep(a)
    if a.cmd in ("trainA", "trainB", "trainVI"):
        return train(a)
    if a.cmd == "ask":
        return ask(a)
    if a.cmd == "trainPT":
        return train_pt(a)
    if a.cmd == "mbpp":
        return mbpp(a)
    return evalv1(a)


if __name__ == "__main__":
    main()
