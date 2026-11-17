"""§TK_100 · M1 — TRAIN HYBRID (Mamba+Attention) TỪ-SCRATCH, QUY-MÔ ~1.3B trên 1×H100.

Nối kiến-trúc đã thắng ở M4 ([[mamba-ssm-tren-m4]]): khối Mamba xen vài tầng attention.
Nâng cho GPU thật:
  • ĐƯỜNG CUDA NHANH — dùng `mamba-ssm` (selective-scan kernel) khi có; KHÔNG có thì
    rơi về khối Mamba thuần-PyTorch của ta (train_mamba.MambaBlock) để CHẠY/TEST tại-chỗ.
  • Attention = RoPE + SDPA (FlashAttention trên CUDA) + GQA — thay mask L×L thủ-công.
  • Dataloader đọc SHARD uint16 (từ build_pretrain_pipeline.py), gói ngữ-cảnh liền mạch.
  • bf16 autocast · grad-checkpointing · fused-AdamW · cosine-LR + warmup · grad-accum ·
    grad-clip · resume opt-state · eval bits held-out · log tok/s + MFU.

Chinchilla ~1.3B: D≈26B token. Trên 1×H100 ≈ 1 tuần, ~$500.

CÀI trên H100 (ghim version — xem TK_102): causal-conv1d==1.4.0 (từ git) + mamba-ssm==2.2.2
CHẠY THẬT (baseline bs=8, KHÔNG --grad-ckpt/--compile — đo thật H100: 2 cờ đó CHẬM hơn,
VRAM dư 64/80GB nên không cần; ~24-26K tok/s, MFU ~20%):
    PYTHONPATH=src python scripts/train_hybrid_scale.py --data data/pretrain_vi \
        --ctx 2048 --d 2048 --layers 48 --attn-every 8 --heads 16 \
        --bs 8 --grad-accum 8 --lr 3e-4 --tokens 18e9 \
        --out checkpoints/hybrid_1p3b.pt \
        --hf-repo donaldluong/nn-aivie-hybrid-1p3b-ckpt --hf-every 5000
    # HF WRITE token: $HF_HOME/token hoặc env HF_TOKEN — ĐỪNG dán vào chat/git.
SMOKE tại-chỗ (CPU/MPS, cấu-hình tí hon):
    PYTHONPATH=src .venv/bin/python scripts/train_hybrid_scale.py --data data/pretrain_vi_smoke \
        --ctx 128 --d 128 --layers 4 --attn-every 2 --heads 4 --bs 4 --grad-accum 2 --steps 20
"""
import sys, os, glob, time, math, argparse
sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint                  # NGC torch không tự nạp (1 GPU, không DDP → grad-ckpt sập)
from train_mamba import MambaBlock, RMSNorm                 # tái-dùng khối Mamba thuần + RMSNorm của ta
from hf_backup_ckpt import maybe_hf_backup, start_hf_backup, wait_hf_backups, warn_if_no_hf_repo
T0 = time.time()

# ── Đường CUDA nhanh: khối Mamba chính-thức (kernel selective-scan) nếu cài được ──
try:
    from mamba_ssm import Mamba as _MambaKernel
    HAVE_KERNEL = True
except Exception:
    HAVE_KERNEL = False


def make_mamba(d, d_state):
    """Kernel CUDA khi có (nhanh) → nếu không, khối thuần-PyTorch của ta (đúng, chạy MPS/CPU)."""
    if HAVE_KERNEL:
        return _MambaKernel(d_model=d, d_state=d_state, d_conv=4, expand=2)
    return MambaBlock(d, d_state)


# ── Attention: RoPE + SDPA (FlashAttention trên CUDA) + GQA ────────────────────────
def rope(x, base=10000.0):                                  # x: (b,h,l,hd)
    b, h, l, hd = x.shape
    pos = torch.arange(l, device=x.device, dtype=torch.float32)
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)                             # (l, hd/2)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


class ScaledAttn(nn.Module):
    def __init__(self, d, heads, kv_heads=None, rope_base=10000.0):
        super().__init__()
        self.h, self.kv = heads, kv_heads or heads
        self.rope_base = float(rope_base)                              # P0: ngữ cảnh dài → base lớn (ABF, vd 1e6)
        self.hd = d // heads
        self.wq = nn.Linear(d, heads * self.hd, bias=False)
        self.wk = nn.Linear(d, self.kv * self.hd, bias=False)
        self.wv = nn.Linear(d, self.kv * self.hd, bias=False)
        self.wo = nn.Linear(heads * self.hd, d, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        q = self.wq(x).view(b, l, self.h, self.hd).transpose(1, 2)     # (b,h,l,hd)
        k = self.wk(x).view(b, l, self.kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(b, l, self.kv, self.hd).transpose(1, 2)
        q, k = rope(q, self.rope_base), rope(k, self.rope_base)
        if self.kv != self.h:                                          # GQA: nhân bản kv
            r = self.h // self.kv
            k = k.repeat_interleave(r, 1); v = v.repeat_interleave(r, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)     # FlashAttention khi CUDA
        return self.wo(o.transpose(1, 2).reshape(b, l, self.h * self.hd))


class HybridLM(nn.Module):
    def __init__(self, V, d, n_layer, attn_layers=(), d_state=16, heads=16, kv_heads=None, rope_base=10000.0):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        aset = set(attn_layers)
        self.attn_layers = sorted(aset)
        self.blocks = nn.ModuleList([
            ScaledAttn(d, heads, kv_heads, rope_base) if i in aset else make_mamba(d, d_state)
            for i in range(n_layer)])
        self.norms = nn.ModuleList([RMSNorm(d) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d); self.V = V; self.grad_ckpt = False

    def forward(self, ids):
        x = self.emb(ids)
        for blk, nm in zip(self.blocks, self.norms):
            if self.grad_ckpt and self.training:
                x = x + torch.utils.checkpoint.checkpoint(
                    lambda t, blk=blk, nm=nm: blk(nm(t)), x, use_reentrant=False)
            else:
                x = x + blk(nm(x))
        return self.norm_f(x) @ self.emb.weight.T                       # logits weight-tied


def attn_layers_of(c):
    """danh sách khối attention: cfg["attn_layers"] (bản nuôi lớn, P0) → hoặc suy từ attn_every (bản 1.3B gốc)."""
    if c.get("attn_layers") is not None:
        return list(c["attn_layers"])
    ae = c.get("attn_every", 0)
    return list(range(ae - 1, c["layers"], ae)) if ae and ae > 0 else []


def build_from_cfg(c, V=32002):
    """dựng HybridLM đúng cấu hình checkpoint (cả bản 1.3B cũ lẫn bản nuôi lớn 96 khối / RoPE dài)."""
    return HybridLM(V, c["d"], c["layers"], attn_layers_of(c), c.get("d_state", 16), c.get("heads", 16),
                    c.get("kv_heads"), c.get("rope_base", 10000.0))


# ── Dataloader: shard uint16 gói ngữ-cảnh liền mạch ────────────────────────────────
class ShardData:
    def __init__(self, path, ctx, val_tokens=200_000):
        self.paths = sorted(glob.glob(os.path.join(path, "shard_*.npy")))
        if not self.paths:
            raise SystemExit(f"không thấy shard trong {path} (chạy build_pretrain_pipeline.py trước)")
        self.mm = [np.load(p, mmap_mode="r") for p in self.paths]
        self.lens = [len(m) for m in self.mm]
        self.ctx = ctx
        self.total = sum(self.lens)
        # val = đuôi shard cuối; train = phần còn lại
        self.val_from = max(0, self.lens[-1] - val_tokens)
        print(f"[{time.time()-T0:.0f}s] data: {len(self.paths)} shard · {self.total/1e6:.1f}M token · "
              f"val={min(val_tokens,self.lens[-1])/1e3:.0f}K token", flush=True)

    def batch(self, bs, split, dev, gen):
        xs = []
        for _ in range(bs):
            si = int(gen.integers(0, len(self.mm)))
            m, ln = self.mm[si], self.lens[si]
            last = si == len(self.mm) - 1
            if split == "val":
                lo, hi = self.val_from, ln - self.ctx - 1
            else:
                lo, hi = 0, (self.val_from if last else ln) - self.ctx - 1
            if hi <= lo:
                si = 0; m, ln = self.mm[0], self.lens[0]; lo, hi = 0, ln - self.ctx - 1
            off = int(gen.integers(lo, max(lo + 1, hi)))
            xs.append(np.asarray(m[off:off + self.ctx + 1], dtype=np.int64))
        b = torch.from_numpy(np.stack(xs)).to(dev, non_blocking=True)
        return b[:, :-1], b[:, 1:]


class MixData:
    """trộn nhiều thư mục shard theo trọng số: "dirA:0.5,dirB:0.3,dirC:0.2" (P0: web lọc / tài liệu dài / Anh / code…)."""
    def __init__(self, spec, ctx, val_tokens=200_000):
        parts = [x.rsplit(":", 1) if ":" in x and x.rsplit(":", 1)[1].replace(".", "").isdigit() else (x, "1")
                 for x in spec.split(",")]
        self.srcs = [ShardData(p, ctx, val_tokens) for p, _ in parts]
        w = np.array([float(x) for _, x in parts]); self.w = w / w.sum()
        self.names = [p for p, _ in parts]

    def batch(self, bs, split, dev, gen):
        k = gen.choice(len(self.srcs), size=bs, p=self.w)
        xs, ys = [], []
        for j in range(len(self.srcs)):
            n = int((k == j).sum())
            if n:
                x, y = self.srcs[j].batch(n, split, dev, gen); xs.append(x); ys.append(y)
        return torch.cat(xs), torch.cat(ys)


def lr_at(step, warmup, total, lr, min_lr, sched="cosine", decay_frac=0.1):
    """cosine (bản 1.3B gốc) | wsd: ấm → GIỮ lr → hạ tuyến tính ở decay_frac cuối (P0: kéo dài được bất cứ lúc nào)."""
    if step < warmup:
        return lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    if sched == "wsd":
        d0 = int(total * (1 - decay_frac))
        return lr if step < d0 else lr - (lr - min_lr) * (step - d0) / max(1, total - d0)
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * r))


MODEL_KEYS = ("d", "layers", "attn_every", "attn_layers", "d_state", "heads", "kv_heads", "rope_base", "new_blocks",
              "expanded_from")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pretrain_vi")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--attn-every", type=int, default=8, help="cứ N tầng đặt 1 attention (hybrid)")
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=None, help="GQA; None=MHA")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=0, help="0 = suy từ --tokens")
    ap.add_argument("--tokens", type=float, default=26e9, help="tổng token đích (Chinchilla ~1.3B → 26e9)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--out", default="checkpoints/hybrid_1p3b.pt")
    ap.add_argument("--hf-repo", default="",
                    help="HF model repo để backup ckpt (cần token WRITE). Rỗng = không đẩy.")
    ap.add_argument("--hf-every", type=int, default=5000,
                    help="đẩy HF mỗi N step (0=tắt). Nên là bội của --save-every.")
    ap.add_argument("--hf-name", default="hybrid_1p3b.pt",
                    help="path_in_repo (rolling latest trên HF)")
    # ── P0 (nuôi lớn 2,6B + ngữ cảnh dài) ──
    ap.add_argument("--rope-base", type=float, default=0, help="đổi RoPE base (ABF, vd 1e6 cho 8K); 0 = giữ của ckpt")
    ap.add_argument("--sched", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--decay-frac", type=float, default=0.1, help="wsd: phần cuối hạ lr")
    ap.add_argument("--anneal-data", default="", help="wsd: trộn dữ liệu RIÊNG cho đoạn hạ lr (kiểu SmolLM3: dồn dữ liệu chất "
                    "lượng cao lúc 'chốt'); trống → đọc file <out bỏ .pt>.anneal nếu có")
    ap.add_argument("--train-new-only", action="store_true", help="chỉ học khối mới (cfg new_blocks), khối cũ đóng băng")
    ap.add_argument("--reset-step", action="store_true", help="init-from nhưng đếm bước lại từ 0 (giai đoạn mới)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # nhiều GPU: torchrun --nproc_per_node=8 … (mỗi GPU 1 bản model, chia batch; rank 0 ghi log/lưu)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if ddp:
        import torch.distributed as dist
        local = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        dev = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    else:
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    main_ = rank == 0
    log = (lambda *x, **k: print(*x, **k)) if main_ else (lambda *x, **k: None)
    use_bf16 = dev.startswith("cuda")
    data = MixData(a.data, a.ctx) if "," in a.data or ":" in a.data else ShardData(a.data, a.ctx)
    af = a.out[:-3] + ".anneal" if a.out.endswith(".pt") else a.out + ".anneal"
    aspec = a.anneal_data or (open(af).read().strip() if os.path.exists(af) else "")
    adata = (MixData(aspec, a.ctx) if "," in aspec or ":" in aspec else ShardData(aspec, a.ctx)) if aspec else None
    V = 32002
    tok_per_step = a.bs * a.grad_accum * a.ctx * world
    steps = a.steps or int(a.tokens // tok_per_step)

    resume_opt, step0, obj = None, 0, None
    if a.init_from and os.path.exists(a.init_from):
        obj = torch.load(a.init_from, map_location="cpu", weights_only=False)
    if obj is not None and "layers" in obj["cfg"]:                      # dựng ĐÚNG cấu hình ckpt (1.3B hoặc 2,6B nuôi lớn)
        mc = {k: obj["cfg"][k] for k in MODEL_KEYS if k in obj["cfg"]}
    else:
        mc = {"d": a.d, "layers": a.layers, "attn_every": a.attn_every, "d_state": a.d_state, "heads": a.heads,
              "kv_heads": a.kv_heads}
    mc.setdefault("rope_base", 10000.0)
    if a.rope_base:
        mc["rope_base"] = a.rope_base                                   # ABF: đổi base cho ngữ cảnh dài
    mc["attn_layers"] = attn_layers_of(mc)                              # luôn lưu danh sách tường minh
    m = build_from_cfg(mc, V).to(dev)
    m.grad_ckpt = a.grad_ckpt
    if obj is not None:
        m.load_state_dict(obj["state"])
        if not a.reset_step:
            resume_opt, step0 = obj.get("opt"), obj.get("step", 0)
        log(f"[{time.time()-T0:.0f}s] INIT {a.init_from} @step {step0} · {mc.get('layers')} khối · rope {mc.get('rope_base', 1e4):.0e}", flush=True)
        del obj
    if a.train_new_only:
        new = set(mc.get("new_blocks", []))
        assert new, "--train-new-only cần ckpt có cfg new_blocks (expand_depth.py)"
        for n_, p in m.named_parameters():
            p.requires_grad_(n_.startswith(("blocks.", "norms.")) and int(n_.split(".")[1]) in new)
    params = [p for p in m.parameters() if p.requires_grad]
    npar = sum(p.numel() for p in m.parameters()); ntrain = sum(p.numel() for p in params)
    try:
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=use_bf16)
    except TypeError:
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    if resume_opt:
        opt.load_state_dict(resume_opt)
    core = m
    if a.compile and use_bf16:
        m = torch.compile(m)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        m = DDP(m, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))] if use_bf16 else None)
    attn_layers = core.attn_layers

    kern = "CUDA-kernel" if HAVE_KERNEL else "thuần-PyTorch(fallback)"
    log(f"[{time.time()-T0:.0f}s] dev={dev}×{world} · HYBRID {npar/1e9:.3f}B (học {ntrain/1e9:.3f}B) · "
        f"d={mc['d']} L={mc['layers']} attn@{attn_layers} · Mamba={kern}", flush=True)
    log(f"[{time.time()-T0:.0f}s] ctx={a.ctx} bs={a.bs}×ga{a.grad_accum}×{world}gpu=eff-token/step={tok_per_step} "
        f"· steps={steps} (~{steps*tok_per_step/1e9:.1f}B token) · lịch {a.sched} · bf16={use_bf16}", flush=True)
    if a.hf_repo and a.hf_every > 0 and a.hf_every % a.save_every != 0:
        print(f"⚠️ --hf-every {a.hf_every} không chia hết --save-every {a.save_every} "
              "→ một số mốc sẽ bỏ (backup chỉ chạy ngay sau lúc LƯU).", flush=True)
    if main_:
        warn_if_no_hf_repo(a.hf_repo, a.hf_every)

    gen = np.random.default_rng(a.seed + 1000 * rank + 7919 * step0); vgen = np.random.default_rng(999)   # chạy tiếp: dòng dữ liệu KHÁC (trước lặp lại y hệt từ đầu)
    actx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)
    ccfg = {**{k: v for k, v in vars(a).items() if k not in ("rope_base",)}, **mc}   # cfg lưu: train + model (mc ưu tiên)

    def save(path, step):
        if main_:
            torch.save({"cfg": ccfg, "state": core.state_dict(), "opt": opt.state_dict(), "step": step}, path)
            log(f"[{time.time()-T0:.0f}s] LƯU {path} @step {step}", flush=True)

    @torch.no_grad()
    def val_bits(nb=20):
        core.eval(); tot = 0.0                                           # chạy trên bản gốc (không qua DDP)
        vgen = np.random.default_rng(999)                                # CÙNG lô val mỗi lần đo (trước: rng chạy tiếp →
        for _ in range(nb):                                              #  thành phần nhóm đổi, bits dao động ±0,1)
            x, y = data.batch(a.bs, "val", dev, vgen)
            with actx:
                lo = core(x)
            tot += float(F.cross_entropy(lo.reshape(-1, V), y.reshape(-1)))
        core.train(); return tot / nb / math.log(2)

    if main_:
        log(f"[{time.time()-T0:.0f}s] bits TRƯỚC: {val_bits(5):.3f}", flush=True)
    d0 = int(steps * (1 - a.decay_frac)) if a.sched == "wsd" else None
    t1 = time.time(); seen = 0
    for step in range(step0 + 1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.warmup, steps, a.lr, a.min_lr, a.sched, a.decay_frac)
        opt.zero_grad(set_to_none=True)
        lsum = 0.0
        src = adata if (adata is not None and d0 is not None and step >= d0) else data
        if src is adata and (step == d0 or step == step0 + 1):
            log(f"[{time.time()-T0:.0f}s] step {step}: HẠ LR + đổi sang dữ liệu chốt: {aspec}", flush=True)
        for k in range(a.grad_accum):
            x, y = src.batch(a.bs, "train", dev, gen)
            sync = (not ddp) or k == a.grad_accum - 1
            ctx_ = m.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx_:
                with actx:
                    lo = m(x)
                    loss = F.cross_entropy(lo.reshape(-1, V), y.reshape(-1)) / a.grad_accum
                loss.backward()
            lsum += loss.detach().item() * a.grad_accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); seen += tok_per_step
        if step % a.eval_every == 0 and main_:
            dt = (time.time() - t1) / (step - step0)
            tps = tok_per_step / dt
            mfu = 6 * npar * tps / (989e12 * world) * 100 if use_bf16 else 0
            log(f"[{time.time()-T0:.0f}s] step {step}/{steps} · loss {lsum/a.grad_accum:.3f} · "
                f"bits {val_bits():.3f} · {tps/1e3:.1f}K tok/s"
                + (f" · MFU~{mfu:.0f}%" if mfu else "") + f" · lr {opt.param_groups[0]['lr']:.2e}"
                + (f" · VRAM {torch.cuda.max_memory_allocated() / 2**30:.0f}GB" if torch.cuda.is_available() else ""), flush=True)
        if d0 is not None and step == d0 - 1:                         # WSD: lưu bản "ổn định" trước khi hạ lr → kéo dài sau
            save(a.out.replace(".pt", "_stable.pt"), step)
        if step % a.save_every == 0:
            save(a.out, step)
            if main_:
                maybe_hf_backup(a.out, a.hf_repo, step, a.hf_every, a.hf_name)
    if steps % a.save_every:                                             # bước cuối chưa lưu thì lưu
        save(a.out, steps)
    if main_:
        log(f"[{time.time()-T0:.0f}s] XONG · LƯU {a.out} · bits cuối {val_bits():.3f}", flush=True)
    if ddp:                                                              # đồng bộ + giải phóng nhóm NCCL TRƯỚC khi đẩy HF:
        import torch.distributed as dist                                 # đẩy 20–30GB mất >10 phút, các rank khác chờ ở
        dist.barrier(); dist.destroy_process_group()                     # barrier sẽ dính NCCL timeout 600s (A1 2026-10-07)
    if main_ and a.hf_repo:
        wait_hf_backups()
        log(f"[{time.time()-T0:.0f}s] HF backup mốc cuối @step {steps}", flush=True)
        start_hf_backup(a.out, a.hf_repo, steps, a.hf_name, wait=True)


if __name__ == "__main__":
    main()
