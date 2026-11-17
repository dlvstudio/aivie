"""
TK_042 G10 — LM nơ-ron làm KÊNH TRÔI CHẢY (thay `W_seq`).

## Vì sao (số, không phải ý kiến)

| | bits/token | bits/byte |
|---|---:|---:|
| `W_seq` (kênh micro hiện tại) = bigram+unigram | **7.088** | 1.276 |
| LM 11.4M, 30k step, CÙNG corpus + held-out | **5.594** | **1.007** |

`W_seq` **CHÍNH LÀ** mắt xích 6.4-bits (TK_042 §0.1b) — kênh yếu nhất trong 4 kênh. G7 đo được
LM tốt hơn nó **1.494 bits/token** (gap train/eval +0.811 ⇒ đã vắt được data; 15/15 mốc eval
đều tốt nhất ⇒ không overfit). §7.8.

## Vì sao KHÔNG phải RAG nhét-`[FACT]`-vào-ngữ-cảnh

LM 11.4M gần như chắc chắn **không có in-context learning** (nó nổi lên theo QUY MÔ). Ở đây
**agenda lái nội dung TỪ BÊN NGOÀI** qua kênh `macro` — LM **không cần "hiểu"** khối FACT, nó
chỉ cần trôi chảy. Đó là lý do cách này hợp với LM nhỏ. TK_042 §7.10b.

## ⚠️ MỘT định nghĩa kiến trúc DUY NHẤT

Class `TinyLM` ở đây là nguồn DUY NHẤT. `scripts/train_neural_lm.py` **import từ đây**, không
tự định nghĩa. Nếu hai bên định nghĩa riêng thì `load_state_dict` lệch → hoặc nổ (may), hoặc
**nạp nhầm trọng số vào nhầm tầng rồi chạy tiếp** (im lặng — đúng lớp lỗi TK_041 §5.2).

torch là dependency **TUỲ CHỌN**: chỉ cần khi thật sự dùng kênh LM. Đường đếm không đụng tới.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _torch():
    try:
        import torch

        return torch
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Kênh LM (TK_042 G10) cần torch: `pip install torch`. "
            "Đường ĐẾM không cần — chỉ bật khi dùng `ar_lm_path`."
        ) from e


def build_model(V: int, d: int, nl: int, nh: int, ctx: int, p: float, ad: float):
    """Kiến trúc DUY NHẤT — train (`scripts/train_neural_lm.py`) và infer (đây) dùng chung."""
    torch = _torch()
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(s, d, h, p, ad):
            super().__init__()
            s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            s.attn = nn.MultiheadAttention(d, h, dropout=ad, batch_first=True)
            s.ff = nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p)
            )

        def forward(s, x, mask):
            h = s.ln1(x)
            a, _ = s.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            return x + s.ff(s.ln2(x))

    class TinyLM(nn.Module):
        def __init__(s, V, d, nl, nh, ctx, p, ad):
            super().__init__()
            s.tok = nn.Embedding(V, d)
            s.pos = nn.Embedding(ctx, d)
            s.drop = nn.Dropout(p)
            s.blocks = nn.ModuleList([Block(d, nh, p, ad) for _ in range(nl)])
            s.ln = nn.LayerNorm(d)
            s.head = nn.Linear(d, V, bias=False)
            s.head.weight = s.tok.weight  # tied — vocab 32k chiếm phần lớn tham số
            s.ctx = ctx

        def forward(s, x):
            T = x.shape[1]
            h = s.drop(s.tok(x) + s.pos(torch.arange(T, device=x.device)))
            m = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            for b in s.blocks:
                h = b(h, m)
            return s.head(s.ln(h))

    return TinyLM(V, d, nl, nh, ctx, p, ad)


def build_model_v2(
    V: int,
    d: int,
    nl: int,
    nh: int,
    ctx: int,
    p: float = 0.0,
    mem_slots: int = 0,
    mem_dk: int = 64,
    mem_topk: int = 32,
    mem_layer: int = -1,
    mos_k: int = 0,
):
    """Kiến trúc v2 (TK_042 §7.20) — RoPE + RMSNorm + SwiGLU (+ tuỳ chọn MemLayer from-scratch).

    Khác v1: KHÔNG có bảng vị trí (`pos.weight`) — RoPE xoay Q/K theo vị trí tương đối.
    Checkpoint v2 lưu dạng dict {"cfg", "state_dict"} — `NeuralLM.load` đọc cfg, KHÔNG đoán
    (nh không suy được từ shape: qkv gộp là (3d,d) bất kể mấy head — đoán là lớp lỗi im lặng
    TK_041 §5.2).

    Init: `nn.init.orthogonal_` cho MỌI ma trận 2D attention/FFN. Vì các head là các LÁT HÀNG
    của cùng một ma trận, ma trận gộp trực giao ⇒ không gian con các head TRỰC GIAO TỪNG ĐÔI
    ngay tại t=0 (câu hỏi tác giả 2026-07-19). Trong train, Muon giữ update đẳng hướng — cùng
    một trực giác, không cần ràng buộc cứng (đo-không-ép: chẩn đoán head-sim in cuối train).
    Embedding vẫn N(0,0.02) — bài học tied-weight (xem `_init` trong train script, bug
    2026-07-16: init N(0,1) → loss khởi đầu 60 bits).
    """
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        def __init__(s, dim, eps=1e-6):
            super().__init__()
            s.w = nn.Parameter(torch.ones(dim))
            s.eps = eps

        def forward(s, x):
            return s.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps)

    dk_head = d // nh
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dk_head, 2).float() / dk_head))
    t = torch.arange(ctx).float()
    freqs = torch.outer(t, inv_freq)  # (ctx, dk/2)
    COS, SIN = freqs.cos(), freqs.sin()

    def rope(x, T, offset=0):  # x: (B, nh, T, dk); offset = vị trí TUYỆT ĐỐI của token đầu
        # ⚠️ KV cache: token mới ở vị trí `past_len`, KHÔNG phải 0 — phải COS[offset:offset+T].
        # Dùng offset=0 sẽ khớp shape nhưng xoay SAI góc ⇒ lỗi im lặng (TK_041 §5.2).
        cos = COS[offset : offset + T].to(x.device)[None, None]
        sin = SIN[offset : offset + T].to(x.device)[None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    class Attn(nn.Module):
        """Tự viết để áp RoPE (nn.MultiheadAttention không cho sửa Q/K trước dot).

        `cache`=(past_k, past_v) hoặc None. Trả (out, new_cache). Prefill (past=None) → is_causal
        True. Decode (có past, T thường=1) → is_causal False (query nhìn TOÀN BỘ past+self)."""

        def __init__(s):
            super().__init__()
            s.qkv = nn.Linear(d, 3 * d, bias=False)
            s.wo = nn.Linear(d, d, bias=False)

        def forward(s, x, cache=None, offset=0):
            B, T, _ = x.shape
            q, k, v = s.qkv(x).chunk(3, dim=-1)
            q = rope(q.view(B, T, nh, dk_head).transpose(1, 2), T, offset)
            k = rope(k.view(B, T, nh, dk_head).transpose(1, 2), T, offset)
            v = v.view(B, T, nh, dk_head).transpose(1, 2)
            past = cache[0] is not None if cache is not None else False
            if past:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
            new_cache = (k, v) if cache is not None else None
            a = F.scaled_dot_product_attention(q, k, v, is_causal=not past)
            return s.wo(a.transpose(1, 2).reshape(B, T, d)), new_cache

    hid = ((int(8 * d / 3) + 63) // 64) * 64  # 2/3·4d, bội 64 — giữ tham số ≈ FFN 4d GELU

    class SwiGLU(nn.Module):
        def __init__(s):
            super().__init__()
            s.w1 = nn.Linear(d, hid, bias=False)
            s.w2 = nn.Linear(d, hid, bias=False)
            s.w3 = nn.Linear(hid, d, bias=False)

        def forward(s, x):
            return s.w3(F.silu(s.w1(x)) * s.w2(x))

    class MemLayer(nn.Module):
        """G18 §7.19d: memory PHẢI train CÙNG backbone từ đầu (ghép lên backbone đóng băng đã
        rớt 3 vòng). N nơ-ron (key,value), top-k nổ thưa — lan truyền kích thích học được."""

        def __init__(s):
            super().__init__()
            s.Wq = nn.Linear(d, mem_dk, bias=False)
            K0 = torch.randn(mem_slots, mem_dk)
            s.K = nn.Parameter(K0 / (K0.norm(dim=-1, keepdim=True) + 1e-8))
            s.V = nn.Parameter(torch.zeros(mem_slots, d))
            s.last_topi = None
            s._cap = False

        def forward(s, x):
            q = s.Wq(x)
            scores = q @ s.K.T / math.sqrt(mem_dk)
            topv, topi = scores.topk(mem_topk, dim=-1)
            if s._cap:
                s.last_topi = topi.detach()
            w = torch.softmax(topv, dim=-1)
            return (w.unsqueeze(-1) * s.V[topi]).sum(-2)

    class Block(nn.Module):
        def __init__(s, use_mem):
            super().__init__()
            s.n1, s.n2 = RMSNorm(d), RMSNorm(d)
            s.attn = Attn()
            s.ffn = MemLayer() if use_mem else SwiGLU()
            s.drop = nn.Dropout(p)

        def forward(s, x, cache=None, offset=0):
            a, new_cache = s.attn(s.n1(x), cache, offset)
            x = x + s.drop(a)
            return x + s.drop(s.ffn(s.n2(x))), new_cache

    class MoSHead(nn.Module):
        """G20 §7.21 — Mixture of Softmaxes (Yang 2018): phiên bản ĐÚNG TOÁN của "multi-head
        output". K ngữ-cảnh → K softmax → TRỘN SAU softmax. Vì softmax phi tuyến, Σπₖ·softmaxₖ
        KHÔNG rút về softmax(Σ) ⇒ hạng log-P THOÁT trần d (phá softmax-bottleneck). GIỮ tied
        (chỉ thêm K ma trận d×d + gating). Trả LOG-P (không phải logits) ⇒ loss=nll, eval KHÔNG
        log_softmax lại (cờ `returns_logprob`; chống double-softmax = lỗi im lặng TK_041 §5.2).

        ⚠️ CHƯA CHẠY THỰC — spec để dành (§7.21). Chỉ bật khi (1) data đã gỡ (2) rerun sạch mà
        bits vẫn kẹt VÀ chẩn đoán chỉ vào OUTPUT, không depth (§7.20a: nút đo được là depth+data)."""

        def __init__(s, emb_weight):
            super().__init__()
            s.emb = emb_weight  # tied: E = tok.weight (V,d) — dùng .T làm output
            s.ctxs = nn.Linear(d, mos_k * d, bias=False)  # K vector ngữ cảnh
            s.gate = nn.Linear(d, mos_k, bias=False)

        def forward(s, h):  # h: (B,T,d) SAU ln
            B, T, _ = h.shape
            pi = torch.softmax(s.gate(h), dim=-1)  # (B,T,K)
            hk = torch.tanh(s.ctxs(h)).view(B, T, mos_k, d)  # Yang dùng tanh
            logits_k = hk @ s.emb.T  # (B,T,K,V) tied
            probs = torch.softmax(logits_k, dim=-1)
            mixed = (pi.unsqueeze(-1) * probs).sum(dim=-2)  # (B,T,V) — TRỘN SAU softmax
            return torch.log(mixed + 1e-9)  # LOG-P

    class LMv2(nn.Module):
        def __init__(s):
            super().__init__()
            s.tok = nn.Embedding(V, d)
            s.blocks = nn.ModuleList(
                [Block(use_mem=(mem_slots > 0 and li == mem_layer)) for li in range(nl)]
            )
            s.ln = RMSNorm(d)
            s.returns_logprob = mos_k > 0
            if mos_k > 0:
                s.mos = MoSHead(s.tok.weight)
            else:
                s.head = nn.Linear(d, V, bias=False)
                s.head.weight = s.tok.weight  # tied
            s.ctx = ctx
            s.drop = nn.Dropout(p)

        def _head(s, h):
            return s.mos(h) if s.returns_logprob else s.head(h)

        def forward(s, x):  # đường TRAIN/EVAL — không cache, giữ nguyên chữ ký cũ
            h = s.drop(s.tok(x))
            for b in s.blocks:
                h, _ = b(h)
            return s._head(s.ln(h))

        def forward_cached(s, x, caches, offset):
            """Đường SINH có KV cache. caches = list[nl] (mỗi phần tử (k,v) hoặc (None,None)).
            Trả (out_head, new_caches). Prefill: offset=0, caches=[(None,None)]×nl, x=cả prompt.
            Decode: offset=past_len, x=token mới [B,1]. Đầu ra GIỐNG HỆT forward (không cache)."""
            h = s.tok(x)  # eval mode: dropout off
            new_caches = []
            for b, c in zip(s.blocks, caches):
                h, nc = b(h, c, offset)
                new_caches.append(nc)
            return s._head(s.ln(h)), new_caches

    model = LMv2()
    with torch.no_grad():
        nn.init.normal_(model.tok.weight, mean=0.0, std=0.02)  # tied-head: PHẢI 0.02
        for blk in model.blocks:
            nn.init.orthogonal_(blk.attn.qkv.weight)  # head-slices trực giao từng đôi @ t=0
            nn.init.orthogonal_(blk.attn.wo.weight)
            if isinstance(blk.ffn, SwiGLU):
                nn.init.orthogonal_(blk.ffn.w1.weight)
                nn.init.orthogonal_(blk.ffn.w2.weight)
                nn.init.orthogonal_(blk.ffn.w3.weight)
            else:
                nn.init.orthogonal_(blk.ffn.Wq.weight)
        if mos_k > 0:
            nn.init.orthogonal_(model.mos.ctxs.weight)
            nn.init.zeros_(model.mos.gate.weight)  # gate=0 ⇒ π đều lúc init (đối xứng K ngữ cảnh)
    return model


class NeuralLM:
    """Bọc TinyLM cho đường sinh: `probs(ids) -> np.ndarray[V]` đã chuẩn hoá max=1."""

    def __init__(self, model, ctx: int, device: str):
        self.model = model
        self.ctx = ctx
        self.device = device
        self._cache: dict[tuple[int, ...], np.ndarray] = {}

    def generate_cached(self, prompt_ids: list[int], n: int) -> list[int]:
        """Sinh greedy với KV cache (TK_042 §7.22). Prefill prompt MỘT lần, rồi decode từng
        token chỉ tính O(1) thay vì O(ctx). Đầu ra GIỐNG HỆT vòng argmax không cache (đã kiểm
        trong probe_kv_cache). CHỈ v2 (forward_cached). Trượt cửa sổ khi vượt ctx."""
        torch = _torch()
        m = self.model
        if not hasattr(m, "forward_cached"):
            raise RuntimeError("generate_cached chỉ hỗ trợ kiến trúc v2 (build_model_v2)")
        m.eval()
        ids = list(prompt_ids)
        nl = len(m.blocks)
        with torch.no_grad():
            caches = [(None, None)] * nl
            x = torch.tensor(ids[-self.ctx :], dtype=torch.long, device=self.device)[None]
            offset = 0
            out, caches = m.forward_cached(x, caches, offset)
            offset = x.shape[1]
            for _ in range(n):
                nxt = int(out[0, -1].argmax())
                ids.append(nxt)
                if offset >= self.ctx:  # cửa sổ đầy → prefill lại ctx token cuối (hiếm khi)
                    caches = [(None, None)] * nl
                    x = torch.tensor(ids[-self.ctx :], dtype=torch.long, device=self.device)[None]
                    offset = 0
                    out, caches = m.forward_cached(x, caches, offset)
                    offset = x.shape[1]
                else:
                    x = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                    out, caches = m.forward_cached(x, caches, offset)
                    offset += 1
        return ids

    def generate(
        self,
        prompt_ids: list[int],
        n: int,
        *,
        eos_id: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        seed: int | None = None,
    ) -> list[int]:
        """Sinh có bộ decode (§7.17d/G15 — argmax thuần → vòng lặp thoái hoá; đây là các núm
        CHỐNG lặp).  `temperature<=0` = greedy argmax (giữ hành vi cũ khi mọi núm mặc định).

        * `repetition_penalty` (CTRL 2019): chia logit token ĐÃ XUẤT (prompt+sinh) cho r nếu >0,
          nhân nếu <0 → dìm lặp mà không cấm cứng.  1.0 = tắt.  Khuyến nghị 1.1–1.3.
        * `no_repeat_ngram`=k: CẤM cứng token tạo lại k-gram đã xuất — cắt loop nguyên khối.
        * `top_k`/`top_p`: cắt đuôi phân bố (G15 v4 — chặn token rác đuôi) trước khi lấy mẫu.
        * `eos_id`: gặp thì dừng sớm.

        Trả CẢ prompt + token sinh (như `generate_cached`)."""
        torch = _torch()
        m = self.model
        m.eval()
        ids = list(prompt_ids)
        is_lp = getattr(m, "returns_logprob", False)  # MoS trả log-P sẵn
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        with torch.no_grad():
            for _ in range(n):
                x = torch.tensor(ids[-self.ctx :], dtype=torch.long, device=self.device)[None]
                logits = m(x)[0, -1].float().cpu()  # (V,) — CPU cho multinomial ổn định trên MPS
                if is_lp:
                    logits = logits.clone()  # đã là log-P; penalty/temp dưới đây vẫn hợp lệ
                if repetition_penalty != 1.0 and ids:
                    seen = torch.tensor(sorted(set(ids)), dtype=torch.long)
                    sl = logits[seen]
                    logits[seen] = torch.where(sl > 0, sl / repetition_penalty, sl * repetition_penalty)
                if no_repeat_ngram > 0 and len(ids) >= no_repeat_ngram - 1:
                    k = no_repeat_ngram
                    prefix = tuple(ids[-(k - 1) :]) if k > 1 else ()
                    banned = {
                        ids[i + k - 1]
                        for i in range(len(ids) - k + 1)
                        if tuple(ids[i : i + k - 1]) == prefix
                    }
                    for b in banned:
                        logits[b] = float("-inf")
                if temperature <= 0:  # greedy
                    nxt = int(logits.argmax())
                else:
                    logits = logits / temperature
                    if top_k > 0:
                        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                        logits[logits < kth] = float("-inf")
                    probs = torch.softmax(logits, -1)
                    if top_p < 1.0:
                        sp, si = torch.sort(probs, descending=True)
                        cum = torch.cumsum(sp, -1)
                        drop = cum - sp > top_p  # giữ tối thiểu tập cumprob ≥ top_p
                        sp[drop] = 0.0
                        probs = torch.zeros_like(probs).scatter_(0, si, sp)
                    nxt = int(torch.multinomial(probs, 1, generator=gen))
                ids.append(nxt)
                if eos_id is not None and nxt == eos_id:
                    break
        return ids

    @classmethod
    def load(cls, path: str | Path, V: int, *, d=256, nl=4, nh=4, ctx=256, device=None):
        """Nạp trọng số. **Suy `d`/`ctx` TỪ CHÍNH state_dict** — không tin tham số truyền vào.

        Nếu tin tham số truyền vào mà nó lệch, `load_state_dict(strict=True)` sẽ nổ (may) —
        nhưng shape có thể tình cờ khớp và nạp NHẦM. Đọc từ tensor là sự thật.

        v2 (RoPE): checkpoint là dict {"cfg", "state_dict"} — đọc cfg tường minh, vì `nh`
        KHÔNG suy được từ shape (qkv gộp là (3d,d) bất kể mấy head).
        """
        torch = _torch()
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "cfg" in obj:  # ---- v2 ----
            cfg, sd2 = obj["cfg"], obj["state_dict"]
            if cfg["V"] != V:
                raise ValueError(
                    f"LM vocab={cfg['V']:,} ≠ checkpoint vocab={V:,} — tokenizer KHÁC "
                    f"(TK_041 §5.2)."
                )
            dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
            model = build_model_v2(**cfg, p=0.0)
            model.load_state_dict(sd2)  # strict — lệch là nổ
            model.eval().to(dev)
            return cls(model, cfg["ctx"], dev)
        sd = obj  # ---- v1 legacy (state_dict trần) ----
        v_ck, d_ck = sd["tok.weight"].shape
        ctx_ck = sd["pos.weight"].shape[0]
        nl_ck = 1 + max(
            int(k.split(".")[1]) for k in sd if k.startswith("blocks.")
        )
        if v_ck != V:
            raise ValueError(
                f"LM vocab={v_ck:,} ≠ checkpoint vocab={V:,} — LM train bằng tokenizer KHÁC. "
                f"Mọi id sẽ trỏ sai mà KHÔNG crash (TK_041 §5.2)."
            )
        dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        model = build_model(v_ck, d_ck, nl_ck, nh, ctx_ck, 0.0, 0.0)
        model.load_state_dict(sd)  # strict — lệch là nổ, KHÔNG nạp nhầm im lặng
        model.eval().to(dev)
        return cls(model, ctx_ck, dev)

    def probs(self, ids: list[int]) -> np.ndarray:
        """P(next | ids[-ctx:]) chuẩn hoá max=1 — cùng quy ước với các kênh khác."""
        torch = _torch()
        key = tuple(ids[-self.ctx :])
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        with torch.no_grad():
            x = torch.tensor(key, dtype=torch.long, device=self.device)[None]
            lg = self.model(x)[0, -1]
            p = torch.softmax(lg, -1).float().cpu().numpy()
        p = p / (p.max() or 1.0)
        if len(self._cache) < 4096:
            self._cache[key] = p
        return p
