"""Suy-luận TỪNG BƯỚC cho HybridLM 1.3B (Mamba + attention) — không tính lại cả ngữ-cảnh mỗi token.

Vì sao: khối Mamba thuần-PyTorch (dùng trên Mac, không có kernel CUDA) quét tuần-tự bằng vòng
lặp Python; vòng sinh cũ (gen_hybrid_scale.gen) gọi model(ids[-ctx:]) cho MỖI token → mỗi token
quét lại toàn bộ ngữ-cảnh ở cả 42 tầng Mamba (~5 tok/s, càng dài càng chậm).
Mamba là mô-hình HỒI-QUY → giữ trạng-thái:
  • tầng Mamba : (cửa-sổ conv d_conv-1 đầu-vào cuối, trạng-thái SSM h (b,d_inner,n))
  • tầng attn  : KV-cache (k đã xoay RoPE theo vị-trí tuyệt-đối, v)
prefill() nạp prompt 1 lần (cùng phép tính như forward gốc) rồi step() mỗi token O(1)/tầng Mamba.
Kiểm khớp số với forward gốc: `python scripts/hybrid_infer.py --check`.
"""
from __future__ import annotations
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from train_hybrid_scale import HybridLM, ScaledAttn
from train_mamba import MambaBlock

V = 32002


def rope_at(x, pos0, base=10000.0):
    """= rope() của train_hybrid_scale nhưng vị-trí bắt đầu từ pos0 (x: (b,h,l,hd))."""
    b, h, l, hd = x.shape
    pos = torch.arange(pos0, pos0 + l, device=x.device, dtype=torch.float32)
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


# ── Mamba ────────────────────────────────────────────────────────────────────────────
def _ssm_params(blk, xc):
    A = -torch.exp(blk.A_log)
    dt, B, C = blk.x_proj(xc).split([blk.dt_rank, blk.d_state, blk.d_state], dim=-1)
    return A, F.softplus(blk.dt_proj(dt)), B, C


def mamba_prefill(blk: MambaBlock, x, st=None):
    """Nạp cả đoạn x; st=(conv_state,h) để NẠP TIẾP từ trạng-thái có sẵn (lượt chat sau)."""
    b, l, _ = x.shape
    k = blk.d_conv - 1
    xx, z = blk.in_proj(x).chunk(2, dim=-1)
    conv_in = xx.transpose(1, 2)                                    # (b,di,l)
    if st is None:
        xc = blk.conv1d(conv_in)[:, :, :l]
        h = torch.zeros(b, conv_in.shape[1], blk.d_state, device=x.device, dtype=x.dtype)
    else:                                                           # nối k đầu-vào cũ, conv KHÔNG pad
        conv_in = torch.cat([st[0], conv_in], -1)
        xc = F.conv1d(conv_in, blk.conv1d.weight, blk.conv1d.bias, groups=blk.conv1d.groups)
        h = st[1]
    xc = F.silu(xc.transpose(1, 2))
    A, dt, B, C = _ssm_params(blk, xc)
    dA = torch.exp(dt.unsqueeze(-1) * A)
    dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * xc.unsqueeze(-1)
    ys = []
    for t in range(l):
        h = dA[:, t] * h + dBx[:, t]
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, 1) + xc * blk.D
    conv_state = F.pad(conv_in, (max(0, k - conv_in.shape[-1]), 0))[:, :, -k:]   # k đầu-vào conv cuối
    return blk.out_proj(y * F.silu(z)), (conv_state, h)


def mamba_step(blk: MambaBlock, x, st):
    conv_state, h = st
    xx, z = blk.in_proj(x[:, 0]).chunk(2, dim=-1)                  # (b,di)
    win = torch.cat([conv_state, xx.unsqueeze(-1)], -1)             # (b,di,d_conv)
    xc = F.silu((win * blk.conv1d.weight[:, 0, :]).sum(-1) + blk.conv1d.bias)
    A, dt, B, C = _ssm_params(blk, xc)
    h = torch.exp(dt.unsqueeze(-1) * A) * h + dt.unsqueeze(-1) * B.unsqueeze(1) * xc.unsqueeze(-1)
    y = (h * C.unsqueeze(1)).sum(-1) + xc * blk.D
    return blk.out_proj(y * F.silu(z)).unsqueeze(1), (win[:, :, 1:], h)


# ── Attention ────────────────────────────────────────────────────────────────────────
def _qkv(blk: ScaledAttn, x, pos0):
    b, l, _ = x.shape
    q = blk.wq(x).view(b, l, blk.h, blk.hd).transpose(1, 2)
    k = blk.wk(x).view(b, l, blk.kv, blk.hd).transpose(1, 2)
    v = blk.wv(x).view(b, l, blk.kv, blk.hd).transpose(1, 2)
    base = getattr(blk, "rope_base", 10000.0)
    return rope_at(q, pos0, base), rope_at(k, pos0, base), v


def _attend(blk, q, k, v, causal):
    if blk.kv != blk.h:
        r = blk.h // blk.kv
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    b, _, l, _ = o.shape
    return blk.wo(o.transpose(1, 2).reshape(b, l, blk.h * blk.hd))


def attn_prefill(blk, x, st=None, pos0=0):
    q, k, v = _qkv(blk, x, pos0)
    if st is None:
        return _attend(blk, q, k, v, causal=True), (k, v)
    K, Vv = torch.cat([st[0], k], 2), torch.cat([st[1], v], 2)
    l, P = x.shape[1], st[0].shape[2]                               # token mới i thấy: mọi cache + mới ≤ i
    mask = torch.ones(l, P + l, dtype=torch.bool, device=x.device)
    mask[:, P:] = torch.tril(torch.ones(l, l, dtype=torch.bool, device=x.device))
    if blk.kv != blk.h:
        r = blk.h // blk.kv
        K2, V2 = K.repeat_interleave(r, 1), Vv.repeat_interleave(r, 1)
    else:
        K2, V2 = K, Vv
    o = F.scaled_dot_product_attention(q, K2, V2, attn_mask=mask)
    b = x.shape[0]
    return blk.wo(o.transpose(1, 2).reshape(b, l, blk.h * blk.hd)), (K, Vv)


def attn_step(blk, x, st, pos):
    q, k, v = _qkv(blk, x, pos)
    K, Vv = torch.cat([st[0], k], 2), torch.cat([st[1], v], 2)
    return _attend(blk, q, K, Vv, causal=False), (K, Vv)            # q dài 1 → nhìn mọi quá-khứ


# ── Model ────────────────────────────────────────────────────────────────────────────
class Runner:
    def __init__(self, model: HybridLM, dev, adapters=None):
        """adapters: {chỉ-số khối (str) → module x→x'} cắm SAU khối đó (expert GatedFFN, TK_104 §14)."""
        self.m, self.dev, self.ad = model, dev, adapters or {}
        assert all(isinstance(b, (MambaBlock, ScaledAttn)) for b in model.blocks), \
            "chỉ hỗ-trợ khối Mamba thuần-PyTorch (không kernel CUDA)"

    def _logits(self, x):
        return self.m.norm_f(x) @ self.m.emb.weight.T

    @torch.no_grad()
    def embed(self, ids, vis=None):
        """ids có thể chứa id ÂM = chỗ giữ token ẢNH (vlm_1p3b) → thay bằng các hàng của vis (n_ảnh_token, d)."""
        t = torch.tensor([ids], dtype=torch.long, device=self.dev)
        x = self.m.emb(t.clamp(min=0))
        if vis is not None:
            x = x.masked_scatter((t < 0).unsqueeze(-1), vis.to(x.dtype))
        return x

    @torch.no_grad()
    def prefill(self, ids, all_logits=False, state=None, vis=None):
        """Nạp ids; state = trạng-thái cũ → nạp TIẾP (chỉ ids mới) thay vì nạp lại cả hội-thoại."""
        x = self.embed(ids, vis)
        pos0 = state["pos"] if state else 0
        olds = state["layers"] if state else [None] * len(self.m.blocks)
        states = []
        for i, (blk, nm, st) in enumerate(zip(self.m.blocks, self.m.norms, olds)):
            if isinstance(blk, ScaledAttn):
                o, st2 = attn_prefill(blk, nm(x), st, pos0)
            else:
                o, st2 = mamba_prefill(blk, nm(x), st)
            x = x + o
            if str(i) in self.ad:
                x = self.ad[str(i)](x)
            states.append(st2)
        lo = self._logits(x if all_logits else x[:, -1:])
        return lo[0], {"layers": states, "pos": pos0 + len(ids)}

    @torch.no_grad()
    def step(self, tok, state):
        x = self.m.emb(torch.tensor([[tok]], dtype=torch.long, device=self.dev))
        new, pos = [], state["pos"]
        for i, (blk, nm, st) in enumerate(zip(self.m.blocks, self.m.norms, state["layers"])):
            if isinstance(blk, ScaledAttn):
                o, st2 = attn_step(blk, nm(x), st, pos)
            else:
                o, st2 = mamba_step(blk, nm(x), st)
            x = x + o
            if str(i) in self.ad:
                x = self.ad[str(i)](x)
            new.append(st2)
        return self._logits(x)[0, -1], {"layers": new, "pos": pos + 1}

    def generate(self, ids, n=256, temp=0.7, top_p=0.9, rep_pen=1.1, recent=64, eos=None,
                 seed=0, stop=lambda: False, state=None, hist=None, out=None, vis=None):
        """Sinh từng token (yield id). rep_pen: TRỪ vào logit các token vừa xuất-hiện (như gen cũ).
        state: nạp TIẾP ids từ trạng-thái cũ. out (dict): nhận trạng-thái cuối ở out['state']."""
        rng = np.random.default_rng(seed)
        lo, state = self.prefill(ids, state=state, vis=vis)
        lo = lo[-1]
        hist = list(hist or []) + list(ids)
        if out is not None:
            out["state"] = state
        for _ in range(n):
            if stop():
                return
            l = lo.float().cpu().numpy().astype(np.float64)
            for t in set(hist[-recent:]):
                if 0 <= t < V:
                    l[t] -= rep_pen
            if temp <= 0:
                nxt = int(l.argmax())
            else:
                l = (l - l.max()) / temp
                p = np.exp(l); p /= p.sum()
                order = np.argsort(p)[::-1]
                keep = order[: max(1, int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1)]
                nxt = int(rng.choice(keep, p=p[keep] / p[keep].sum()))
            if eos is not None and nxt == eos:
                return
            yield nxt
            hist.append(nxt)
            lo, state = self.step(nxt, state)
            if out is not None:
                out["state"] = state


def load(ckpt, dev):
    # Runner cần khối Mamba thuần-PyTorch (giữ trạng-thái). Trên CUDA, make_mamba sẽ chọn kernel mamba_ssm
    # → ép tắt. Trọng-số tương-thích (cùng tên in_proj/conv1d/x_proj/dt_proj/A_log/D/out_proj).
    import train_hybrid_scale
    train_hybrid_scale.HAVE_KERNEL = False
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    c, state, step = obj["cfg"], obj["state"], obj.get("step")
    del obj
    m = train_hybrid_scale.build_from_cfg(c, V)
    m.load_state_dict(state)
    del state
    return m.eval().to(dev), c, step


if __name__ == "__main__" and "--check" in sys.argv:
    # Kiểm khớp số: prefill(20) + step(20) phải cho logit giống forward gốc trên 40 token.
    import time
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if "--tiny" in sys.argv:
        m = HybridLM(V, 64, 8, [3, 7], 16, 4).eval().to(dev); step = "tí hon"
    else:
        m, _, step = load("checkpoints/hf_hybrid/hybrid_1p3b_assistant_v5.pt", dev)
    r = Runner(m, dev)
    ids = torch.randint(5, V, (40,)).tolist()
    with torch.no_grad():
        ref = m(torch.tensor([ids], device=dev))[0].float()
    lo, st = r.prefill(ids[:12], all_logits=True)
    lo2, st = r.prefill(ids[12:20], all_logits=True, state=st)       # nạp TIẾP 8 token
    got = [lo.float(), lo2.float()]
    t = time.time()
    for tok in ids[20:]:
        l, st = r.step(tok, st)
        got.append(l.float()[None])
    got = torch.cat(got)[:40]
    d = (got[:39] - ref[:39]).abs().max().item()
    print(f"model {step} · sai-lệch logit lớn nhất (39 vị-trí) = {d:.2e} · "
          f"argmax khớp {int((got[:39].argmax(-1) == ref[:39].argmax(-1)).sum())}/39 · "
          f"{20/(time.time()-t):.1f} bước/giây")
