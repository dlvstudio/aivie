"""§TK_086 — MAMBA (Selective SSM) TỐI GIẢN thuần PyTorch, chạy MPS/M4.

Gói `mamba-ssm` chính thức cần CUDA kernel → KHÔNG chạy Mac. Đây là bản tự-cài
(kiểu mamba-minimal): selective scan bằng vòng-lặp tuần tự (chậm hơn kernel CUDA
nhưng ĐÚNG + chạy MPS). Mục đích: KIỂM mamba có TRAIN ở ctx DÀI (256/512) nơi
transformer đụng tường graph-compile MPS (ctx≤128) không.

    PYTHONPATH=src .venv/bin/python scripts/train_mamba.py --ctx 256 --steps 300
"""
import argparse, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
T0 = time.time()


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class MambaBlock(nn.Module):
    def __init__(self, d, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_inner = expand * d
        self.dt_rank = dt_rank or math.ceil(d / 16)
        self.in_proj = nn.Linear(d, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))          # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)
        self.d_state = d_state; self.d_conv = d_conv

    def ssm(self, x):                                     # x: (b,l,d_inner)
        A = -torch.exp(self.A_log)                        # (d_inner, n)
        db = self.x_proj(x)                               # (b,l, dt_rank+2n)
        dt, B, C = db.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                 # (b,l,d_inner)
        dA = torch.exp(dt.unsqueeze(-1) * A)              # (b,l,d_inner,n)
        dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)   # (b,l,d_inner,n)
        b, l, di, n = dA.shape
        h = torch.zeros(b, di, n, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(l):                                # SELECTIVE SCAN tuần tự
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1)                            # (b,l,d_inner)
        return y + x * self.D

    def forward(self, x):                                 # x: (b,l,d)
        b, l, _ = x.shape
        xz = self.in_proj(x)
        xx, z = xz.chunk(2, dim=-1)                       # (b,l,d_inner) mỗi
        xx = self.conv1d(xx.transpose(1, 2))[:, :, :l].transpose(1, 2)  # causal conv
        xx = F.silu(xx)
        y = self.ssm(xx) * F.silu(z)                      # gated
        return self.out_proj(y)


class AttnBlock(nn.Module):
    """§TK_087 — tầng self-attention nhân-quả (cho hybrid Jamba-nhỏ)."""
    def __init__(self, d, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.0)
    def forward(self, x):
        l = x.shape[1]
        cm = torch.triu(torch.ones(l, l, dtype=torch.bool, device=x.device), 1)
        o, _ = self.attn(x, x, x, attn_mask=cm, need_weights=False)
        return o


class Mamba(nn.Module):
    def __init__(self, V, d, n_layer, E0=None, d_state=16, attn_layers=(), heads=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)         # init nhỏ (kiểu GPT) — tránh logits nổ (weight-tied)
        attn_set = set(attn_layers)
        self.blocks = nn.ModuleList([
            AttnBlock(d, heads) if i in attn_set else MambaBlock(d, d_state)
            for i in range(n_layer)])
        self.is_attn = [i in attn_set for i in range(n_layer)]
        self.norms = nn.ModuleList([RMSNorm(d) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d); self.V = V
    def forward(self, ids, return_hidden=False):
        x = self.emb(ids)
        for blk, nm in zip(self.blocks, self.norms):
            x = x + blk(nm(x))                             # residual
        x = self.norm_f(x)
        logits = x @ self.emb.weight.T                     # weight-tied logits
        if return_hidden:                                  # §TK_096 combiner cần trạng-thái-ẩn làm đầu vào
            return logits, x
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--attn-layers", default="", help="§TK_087 chỉ số tầng attention (vd '2') → hybrid Jamba; rỗng=Mamba thuần")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4, help="§TK_095 cộng-dồn gradient qua N micro-batch → batch hiệu-dụng=bs*N, CÙNG RAM (KHÔNG nhanh hơn, chỉ gradient mượt hơn). Mặc-định 4 (eff bs=16); đặt 1 để tắt")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--data", default="checkpoints/enc_data_long512.npz")
    ap.add_argument("--init-from", default=None, help="§TK_094 warm-start/RESUME từ checkpoint .pt (kiến trúc lấy từ đó)")
    ap.add_argument("--out", default="checkpoints/mamba_small.pt")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = np.load(a.data); X = d["X"][:, :a.ctx]; L = d["L"]; E0 = d["E"]
    V = E0.shape[0]
    keep = np.where(L >= a.ctx // 2)[0]                    # đủ dài
    rng = np.random.default_rng(0); rng.shuffle(keep)
    ho, tr = keep[:400], keep[400:]
    Xt = torch.tensor(X, dtype=torch.long)
    attn_layers = [int(x) for x in a.attn_layers.split(",") if x.strip() != ""]
    resume_opt = None                                 # §TK_095 opt-state để nạp lại sau khi tạo optimizer
    if a.init_from:                                   # §TK_094 RESUME: kiến trúc + trọng số từ checkpoint
        obj0 = torch.load(a.init_from, map_location="cpu"); c0 = obj0["cfg"]
        a.d = c0["d"]; a.layers = c0["layers"]; a.d_state = c0["d_state"]; a.heads = c0.get("heads", 4)
        # suy attn-layers từ STATE (cfg cũ có thể lưu rỗng); cập nhật a.attn_layers để cfg mới lưu ĐÚNG
        import re as _re
        _al = sorted({int(_m.group(1)) for _k in obj0["state"]
                      for _m in [_re.match(r"blocks\.(\d+)\.attn\.", _k)] if _m})
        attn_layers = _al or [int(x) for x in str(c0.get("attn_layers", "")).split(",") if x.strip() != ""]
        a.attn_layers = ",".join(str(x) for x in attn_layers)
        m = Mamba(V, a.d, a.layers, None, a.d_state, attn_layers, a.heads).to(dev)
        m.load_state_dict(obj0["state"])
        resume_opt = obj0.get("opt")                  # §TK_095 opt-state (None nếu checkpoint cũ chưa lưu)
        print(f"[{time.time()-T0:.0f}s] RESUME từ {a.init_from} (d={a.d} L={a.layers} attn={attn_layers})"
              + (" +opt-state" if resume_opt else " (opt-state KHÔNG có → warm-up lại)"), flush=True)
    else:
        m = Mamba(V, a.d, a.layers, E0, a.d_state, attn_layers, a.heads).to(dev)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    if resume_opt is not None:                         # §TK_095 nạp lại momentum/adaptive-LR → resume LIỀN MẠCH (không nảy bits)
        opt.load_state_dict(resume_opt)
        print(f"[{time.time()-T0:.0f}s] nạp opt-state → resume liền mạch (không warm-up lại)", flush=True)
    tag = f"HYBRID(attn@{attn_layers})" if attn_layers else "Mamba thuần"
    eff_bs = a.bs * max(1, a.grad_accum)
    print(f"[{time.time()-T0:.0f}s] dev={dev} {tag} {npar/1e6:.1f}M · ctx={a.ctx} d={a.d} L={a.layers} state={a.d_state} · bs={a.bs}×ga{max(1,a.grad_accum)}=eff{eff_bs}", flush=True)
    print(f"[{time.time()-T0:.0f}s] train={len(tr)} ho={len(ho)} V={V}", flush=True)

    @torch.no_grad()
    def ho_bits(cap=64):
        m.eval(); tot = 0.0; c = 0
        for i in range(0, min(len(ho), cap), a.bs):
            ids = Xt[ho[i:i + a.bs]].to(dev)
            lo = m(ids)
            l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
            tot += float(l); c += 1
        m.train(); return tot / max(c, 1) / math.log(2)

    t1 = time.time()
    print(f"[{time.time()-T0:.0f}s] bits TRƯỚC: {ho_bits():.3f}", flush=True)
    ga = max(1, a.grad_accum)                              # §TK_095 số micro-batch cộng-dồn
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        loss_sum = 0.0
        for _ in range(ga):                               # cộng-dồn gradient: batch hiệu-dụng = bs*ga, cùng RAM
            ids = Xt[tr[rng.integers(0, len(tr), a.bs)]].to(dev)
            lo = m(ids)
            loss = F.cross_entropy(lo[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1)) / ga
            loss.backward()                               # grad TÍCH-LŨY (không zero giữa chừng)
            loss_sum += loss.detach().item() * ga         # detach → tránh UserWarning requires_grad
        loss_val = loss_sum / ga                          # loss trung bình để in
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0:
            dt = (time.time() - t1) / step
            print(f"[{time.time()-T0:.0f}s] step {step} ({dt*1000:.0f}ms/step) loss {loss_val:.3f} | bits {ho_bits():.3f}", flush=True)
            torch.save({"cfg": vars(a), "state": m.state_dict(), "opt": opt.state_dict()}, a.out)
    torch.save({"cfg": vars(a), "state": m.state_dict(), "opt": opt.state_dict()}, a.out)
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
