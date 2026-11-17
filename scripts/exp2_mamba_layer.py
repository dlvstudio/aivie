"""TK_108 · Thí nghiệm 2 — thay tầng cửa sổ trượt của Gemma 4 12B bằng Mamba-2, trên M4 (MLX).

Câu hỏi: đổi attention cửa sổ trượt → Mamba (khởi tạo từ chính Q/K/V/O của attention) thì mất bao nhiêu?
  - train P1 (so từng tầng, thầy cấp đầu vào) CHỈ bằng văn bản tiếng Việt
  - đo trên dữ liệu CHƯA THẤY: web VI, văn phòng VI (chat), code EN (chat), văn bản dài 8K
  - so: gốc ↔ bỏ hẳn attention tầng đó ↔ Mamba thay vào (1 tầng / cả mấy tầng cùng lúc)

  python scripts/exp2_mamba_layer.py selftest
  python scripts/exp2_mamba_layer.py train --layers 1,14,30 --steps 1500
  python scripts/exp2_mamba_layer.py eval  --layers 1,14,30
"""
import argparse, json, math, os, random, sys, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = os.path.join(ROOT, "out", "exp2_mamba_layer")
OUT = OUT_BASE
MODEL_ID = "mlx-community/gemma-4-12B-it-4bit"
F32 = mx.float32
LR_MUL = 5.0  # tham số MỚI (dt, A, conv, D, norm) học nhanh gấp 5 lần (thủ thuật nhân trong forward)


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)


# ───────────────────────── SSD (Mamba-2) dạng chia khúc ─────────────────────────
def ssd_chunked(X, Bm, C, ld, Q=256, s0=None):
    """y_t = Σ_{s≤t} exp(Σ_{r=s+1..t} ld_r) (C_t·B_s) X_s  — X đã nhân dt.
    X (b,L,H,P) · Bm,C (b,L,H,N) · ld (b,L,H) ≤ 0. Trả y (b,L,H,P), trạng thái cuối (b,H,N,P)."""
    b, L, H, P = X.shape
    N = Bm.shape[-1]
    pad = (-L) % Q
    if pad:
        z = lambda a: mx.pad(a, [(0, 0), (0, pad)] + [(0, 0)] * (a.ndim - 2))
        X, Bm, C, ld = z(X), z(Bm), z(C), z(ld)
    c = (L + pad) // Q
    tr = lambda a, d: a.reshape(b, c, Q, H, d).transpose(0, 1, 3, 2, 4)   # (b,c,H,Q,d)
    X, Bm, C = tr(X, P), tr(Bm, N), tr(C, N)
    cum = mx.cumsum(ld.reshape(b, c, Q, H), axis=2).transpose(0, 1, 3, 2)  # (b,c,H,Q)
    diff = cum[..., :, None] - cum[..., None, :]                  # (b,c,H,Q,Q)
    tri = mx.tril(mx.ones((Q, Q), dtype=mx.bool_))
    Lm = mx.where(tri, mx.exp(mx.minimum(diff, 0.0)), 0.0)
    y = ((C @ Bm.swapaxes(-1, -2)) * Lm) @ X                      # (b,c,H,Q,P)
    dec_end = mx.exp(cum[..., -1:] - cum)                         # (b,c,H,Q)
    S = (Bm * dec_end[..., None]).swapaxes(-1, -2) @ X            # (b,c,H,N,P)
    chunk_dec = mx.exp(cum[..., -1])                              # (b,c,H)
    s = mx.zeros((b, H, N, P), dtype=X.dtype) if s0 is None else s0
    prev = []
    for k in range(c):
        prev.append(s)
        s = s * chunk_dec[:, k, :, None, None] + S[:, k]
    Sp = mx.stack(prev, axis=1)                                   # (b,c,H,N,P)
    y = y + (C * mx.exp(cum)[..., None]) @ Sp
    return y.transpose(0, 1, 3, 2, 4).reshape(b, c * Q, H, P)[:, :L], s


def ssd_naive(X, Bm, C, ld):
    b, L, H, P = X.shape
    cum = mx.cumsum(ld, axis=1).transpose(0, 2, 1)                # (b,H,L)
    diff = cum[..., :, None] - cum[..., None, :]
    tri = mx.tril(mx.ones((L, L), dtype=mx.bool_))
    Lm = mx.where(tri, mx.exp(mx.minimum(diff, 0.0)), 0.0)
    T = lambda a: a.transpose(0, 2, 1, 3)
    return (((T(C) @ T(Bm).swapaxes(-1, -2)) * Lm) @ T(X)).transpose(0, 2, 1, 3)


# ───────────────────────── bộ trộn Mamba-2 khởi tạo từ attention ─────────────────────────
def deq(lin):
    if isinstance(lin, nn.QuantizedLinear):
        w = mx.dequantize(lin.weight, lin.scales, lin.get("biases"), lin.group_size, lin.bits, mode=getattr(lin, "mode", "affine"))
    else:
        w = lin.weight
    return w.astype(F32)


def inv_softplus(y):
    return mx.log(mx.expm1(y))


def causal_conv(x, w, buf=None):
    """x (b,L,ch), w (K,ch) — depthwise nhân quả; buf (b,K-1,ch) = phần đuôi của đoạn trước."""
    K = w.shape[0]
    L = x.shape[1]
    xp = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)]) if buf is None else mx.concatenate([buf, x], axis=1)
    return sum(xp[:, k:k + L, :] * w[k] for k in range(K)), xp[:, -(K - 1):]


GLOBAL_MEM = float(os.environ.get('GLOBAL_MEM', 32768))


class Mamba2Mixer(nn.Module):
    def __init__(self, attn=None, hidden=3840, H=16, Hkv=8, D=256, K=4, mem=(8.0, 2048.0), eps=1e-6, chunk=256):
        super().__init__()
        if attn is not None:   # suy cỡ từ attention gốc (tầng global: Hkv=1, D=512, K=V)
            H, Hkv, D = attn.n_heads, attn.n_kv_heads, attn.head_dim
            if not attn.is_sliding:
                mem = (8.0, GLOBAL_MEM)
        self.H, self.Hkv, self.D, self.K, self.eps, self.chunk = H, Hkv, D, K, eps, chunk
        self.q = nn.Linear(hidden, H * D, bias=False)
        self.k = nn.Linear(hidden, Hkv * D, bias=False)
        self.v = nn.Linear(hidden, Hkv * D, bias=False)
        self.o = nn.Linear(H * D, hidden, bias=False)
        self.q_norm = mx.ones((D,))
        self.k_norm = mx.ones((D,))
        if attn is not None:
            self.q.weight, self.k.weight, self.v.weight, self.o.weight = deq(attn.q_proj), deq(attn.k_proj), deq(attn.v_proj if hasattr(attn, 'v_proj') else attn.k_proj), deq(attn.o_proj)
            self.q_norm = attn.q_norm.weight.astype(F32)
            self.k_norm = attn.k_norm.weight.astype(F32)
        # ── phần MỚI (chia LR_MUL, nhân lại trong forward) ──
        delta = mx.zeros((K, 1)); delta[K - 1] = 1.0
        self.conv_c = mx.contiguous(mx.broadcast_to(delta, (K, H * D)) / LR_MUL)
        self.conv_b = mx.contiguous(mx.broadcast_to(delta, (K, Hkv * D)) / LR_MUL)
        self.conv_x = mx.contiguous(mx.broadcast_to(delta, (K, Hkv * D)) / LR_MUL)
        mems = mx.exp(mx.linspace(math.log(mem[0]), math.log(mem[1]), H))   # trí nhớ ~8 → 2048 token, đa thang
        self.dt_bias = inv_softplus(1.0 / mems) / LR_MUL
        self.dt_w = mx.zeros((hidden, H))
        self.A_log = mx.zeros((H,))
        self.Dskip = mx.zeros((H,))
        self.out_norm = mx.ones((H, D)) / LR_MUL

    def _proj(self, h):
        b, L, _ = h.shape
        H, Hkv, D = self.H, self.Hkv, self.D
        C = mx.fast.rms_norm(self.q(h).reshape(b, L, H, D), self.q_norm, self.eps).reshape(b, L, H * D)
        Bm = mx.fast.rms_norm(self.k(h).reshape(b, L, Hkv, D), self.k_norm, self.eps).reshape(b, L, Hkv * D)
        X = mx.fast.rms_norm(self.v(h).reshape(b, L, Hkv, D), None, self.eps).reshape(b, L, Hkv * D)
        dt = nn.softplus(h @ self.dt_w + self.dt_bias * LR_MUL)           # (b,L,H)
        return C, Bm, X, dt

    def _finish(self, y, X, b, L):
        H, D = self.H, self.D
        y = y + (self.Dskip * LR_MUL)[:, None] * X
        y = mx.fast.rms_norm(y, None, self.eps) * (self.out_norm * LR_MUL)
        return self.o(y.reshape(b, L, H * D))

    def __call__(self, h, st=None):
        """cả chuỗi (st=None) hoặc nối tiếp từ trạng thái st (prefill/sinh từng token), st được cập nhật."""
        b, L, _ = h.shape
        H, Hkv, D, r = self.H, self.Hkv, self.D, self.H // self.Hkv
        g = (lambda k: None) if st is None else st.get
        C, Bm, X, dt = self._proj(h)
        C, bc = causal_conv(C, self.conv_c * LR_MUL, g("conv_c"))
        Bm, bb = causal_conv(Bm, self.conv_b * LR_MUL, g("conv_b"))
        X, bx = causal_conv(X, self.conv_x * LR_MUL, g("conv_x"))
        C = C.reshape(b, L, H, D)
        Bm = mx.repeat(Bm.reshape(b, L, Hkv, D), r, axis=2)
        X = mx.repeat(X.reshape(b, L, Hkv, D), r, axis=2)
        ld = -mx.exp(self.A_log * LR_MUL) * dt
        y, sN = ssd_chunked(X * dt[..., None], Bm, C, ld, min(self.chunk, max(L, 1)), g("S"))
        if st is not None:
            st.update(conv_c=bc, conv_b=bb, conv_x=bx, S=sN)
        return self._finish(y, X, b, L)

    def init_state(self, b=1):
        K, H, Hkv, D = self.K, self.H, self.Hkv, self.D
        z = lambda ch: mx.zeros((b, K - 1, ch))
        return {"conv_c": z(H * D), "conv_b": z(Hkv * D), "conv_x": z(Hkv * D), "S": mx.zeros((b, H, D, D))}


from mlx_lm.models.cache import _BaseCache


class SSMCache(_BaseCache):
    def __init__(self):
        self.offset, self._ssm = 0, None

    @property
    def state(self):
        return [] if self._ssm is None else list(self._ssm.values())

    @state.setter
    def state(self, v):
        pass

    def empty(self):
        return self._ssm is None

    def make_mask(self, *args, **kwargs):   # tầng global đầu tiên (5) dựng mask cho mọi tầng full → cần offset đúng
        from mlx_lm.models.cache import create_attention_mask as cam
        return cam(*args, offset=self.offset, **kwargs)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.state)


class Swap(nn.Module):
    """thế chỗ layer.self_attn trong mlx_lm Gemma4 (cache=None: cả chuỗi; SSMCache: nối tiếp trạng thái)."""
    def __init__(self, mixer, zero=False):
        super().__init__()
        self.mixer, self.zero = mixer, zero

    def __call__(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        if self.zero:
            if cache is not None:
                cache.offset += x.shape[1]
            return mx.zeros_like(x), (None, None), offset
        if cache is None:
            y = self.mixer(x.astype(F32))
        else:
            if cache._ssm is None:
                cache._ssm = self.mixer.init_state(x.shape[0])
            y = self.mixer(x.astype(F32), cache._ssm)
            cache.offset += x.shape[1]
        return y.astype(x.dtype), (None, None), offset


# ───────────────────────── dữ liệu ─────────────────────────
def load_model():
    """bản 12B trên HF là 'gemma4_unified' (mlx_lm chưa có) — phần chữ giống hệt gemma4_text: tự dựng, bỏ ảnh/âm."""
    import glob
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from mlx_lm.models import gemma4
    from mlx_lm.utils import load_tokenizer
    path = Path(snapshot_download(MODEL_ID))
    cfg = json.load(open(path / "config.json"))
    t = dict(cfg["text_config"]); t["model_type"] = "gemma4_text"
    model = gemma4.Model(gemma4.ModelArgs(model_type="gemma4", text_config=t, vocab_size=t["vocab_size"]))
    w = {}
    for f in sorted(glob.glob(str(path / "model-*.safetensors"))):
        w.update({k: v for k, v in mx.load(f).items() if k.startswith("language_model.")})
    q = cfg["quantization"]
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"),
                class_predicate=lambda pth, m: f"{pth}.scales" in w)
    model.load_weights(list(w.items()), strict=True)
    mx.eval(model.parameters())
    model.eval(); model.freeze()
    tok = load_tokenizer(path, eos_token_ids=cfg.get("eos_token_id"))
    return model, tok


TRAIN_SRC_LOAI_TRU = ("office", "code", "dich", "gemma4-26b-dialog", "alpaca-soan-dich")   # KHÔNG cho thấy văn phòng/code/dịch


def chat_text(tok, msgs):
    t = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    return t[len("<bos>"):] if t.startswith("<bos>") else t


def train_texts(tok):
    """hội thoại TIẾNG VIỆT chung (sft_vi) — bỏ hẳn nguồn văn phòng / code / dịch."""
    out = []
    with open(os.path.join(ROOT, "data/sft_vi/sft_mix.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if any(x in r.get("source", "") for x in TRAIN_SRC_LOAI_TRU):
                continue
            out.append(chat_text(tok, r["messages"]))
    random.Random(0).shuffle(out)
    return out


def jsonl_chats(tok, path, n, from_end=False):
    lines = open(path).read().splitlines()
    if from_end:
        lines = lines[::-1]
    return [chat_text(tok, json.loads(l)["messages"]) for l in lines[:n]]


def pack(tok, texts, L, n_seq):
    """mỗi chuỗi = <bos> + L-1 token liên tiếp của dòng hội thoại nối nhau."""
    bos = tok.bos_token_id
    buf, seqs = [], []
    for d in texts:
        buf += tok.encode(d, add_special_tokens=False)
        while len(buf) >= L - 1:
            seqs.append([bos] + buf[:L - 1]); buf = buf[L - 1:]
            if len(seqs) >= n_seq:
                return seqs
    return seqs


def train_stream(tok, L, B):
    texts = train_texts(tok)
    log(f"data train: {len(texts)} hội thoại VI (đã bỏ văn phòng/code/dịch)")
    bos = tok.bos_token_id
    buf = []
    while True:
        for d in texts:
            buf += tok.encode(d, add_special_tokens=False)
            while len(buf) >= (L - 1) * B:
                yield mx.array([[bos] + buf[k * (L - 1):(k + 1) * (L - 1)] for k in range(B)])
                buf = buf[(L - 1) * B:]


# ───────────────────────── thầy: lấy đầu vào/đầu ra tầng ─────────────────────────
def lm_inner(model):
    return model.language_model.model


def capture(model, toks, layers):
    lm = lm_inner(model)
    h = lm.embed_tokens(toks) * lm.embed_scale
    masks = lm._make_masks(h, [None] * len(lm.layers))
    out = {}
    for i in range(max(layers) + 1):
        lay = lm.layers[i]
        if i in layers:
            hn = lay.input_layernorm(h)
            o, _, _ = lay.self_attn(hn, masks[i], None)
            out[i] = (hn.astype(F32), o.astype(F32))
            mx.eval(out[i])
        h, _, _ = lay(h, masks[i], None)
        mx.eval(h)   # cắt đồ thị từng tầng — chạy tới tầng 47 không giữ hết activation
    return out


class Bag(nn.Module):
    def __init__(self, mixers):
        super().__init__()
        self.mixers = mixers


def nmse_post(lay, ys, yt):
    w = lay.post_attention_layernorm.weight.astype(F32)
    eps = lay.post_attention_layernorm.eps
    a = mx.fast.rms_norm(ys, w, eps)
    t = mx.fast.rms_norm(yt, w, eps)
    return ((a - t) ** 2).mean() / (t ** 2).mean()


# ───────────────────────── lệnh ─────────────────────────
def cmd_selftest(a):
    mx.random.seed(0)
    b, L, H, N, P = 1, 700, 16, 256, 256
    X = mx.random.normal((b, L, H, P)); Bm = mx.random.normal((b, L, H, N)); C = mx.random.normal((b, L, H, N))
    ld = -mx.random.uniform(0, 0.2, (b, L, H))
    y1, _ = ssd_chunked(X * 0.05, Bm * 0.05, C, ld, Q=256); X = X * 0.05; Bm = Bm * 0.05
    y2 = ssd_naive(X, Bm, C, ld)
    log("ssd chunked vs naive max|Δ| =", mx.abs(y1 - y2).max().item(), " scale", mx.abs(y2).max().item())
    m = Mamba2Mixer(None, hidden=32, H=4, Hkv=2, D=8, chunk=16)
    m.q.weight = mx.random.normal(m.q.weight.shape) * 0.2; m.k.weight = mx.random.normal(m.k.weight.shape) * 0.2
    m.v.weight = mx.random.normal(m.v.weight.shape) * 0.2; m.o.weight = mx.random.normal(m.o.weight.shape) * 0.2
    m.conv_c = mx.random.normal(m.conv_c.shape) * 0.1; m.dt_w = mx.random.normal(m.dt_w.shape) * 0.1
    h = mx.random.normal((1, 40, 32))
    full = m(h)
    st = m.init_state(1)
    rec = mx.concatenate([m(h[:, :7], st)] + [m(h[:, t:t + 1], st) for t in range(7, 40)], axis=1)
    log("mixer full vs step max|Δ| =", mx.abs(full - rec).max().item(), " scale", mx.abs(full).max().item())


def cmd_train(a):
    os.makedirs(OUT, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]
    model, tok = load_model()
    lm = lm_inner(model)
    mix = [Mamba2Mixer(lm.layers[i].self_attn) for i in layers]
    bag = Bag(mix)
    mx.eval(bag.parameters())
    ntr = sum(v.size for _, v in tree_flatten(bag.trainable_parameters()))
    log(f"tầng {layers} · tham số Mamba train = {ntr/1e6:.1f}M · L={a.L} B={a.B} steps={a.steps}")
    sched = optim.join_schedules([optim.linear_schedule(0, a.lr, 50), optim.cosine_decay(a.lr, a.steps - 50, a.lr * 0.1)], [50])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)

    def loss_fn(bag, caps):
        tot, parts = 0.0, []
        for j, i in enumerate(layers):
            hn, yt = caps[i]
            l = nmse_post(lm.layers[i], bag.mixers[j](hn), yt)
            parts.append(l); tot = tot + l
        return tot, parts

    vg = nn.value_and_grad(bag, lambda bg, caps: loss_fn(bg, caps)[0])
    stream = train_stream(tok, a.L, a.B)
    hist, t0, ema = [], time.time(), None
    for step in range(1, a.steps + 1):
        toks = next(stream)
        caps = capture(model, toks, set(layers))
        mx.eval(caps)
        loss, g = vg(bag, caps)
        opt.update(bag, g)
        mx.eval(bag.parameters(), opt.state, loss)
        if step % 10 == 0 or step == 1:
            _, parts = loss_fn(bag, caps)
            p = [round(x.item(), 4) for x in parts]
            hist.append({"step": step, "nmse": p})
            el = time.time() - t0
            log(f"step {step}/{a.steps} nmse/tầng {dict(zip(layers, p))} · {step*a.L*a.B/el:.0f} tok/s · {el/60:.1f} phút")
        if step % 250 == 0 or step == a.steps:
            bag.save_weights(os.path.join(OUT, "mixers.safetensors"))
            json.dump({"layers": layers, "hist": hist, "L": a.L, "B": a.B, "steps": step}, open(os.path.join(OUT, "train_log.json"), "w"))
    log("XONG train")


def cmd_kd(a):
    """P2 thu nhỏ: chạy CẢ model (Mamba đã thay) → KL với xác suất của model gốc ở đầu ra; chỉ cập nhật Mamba."""
    layers = [int(x) for x in a.layers.split(",")]
    model, tok = load_model()
    lm = lm_inner(model)
    orig = {i: lm.layers[i].self_attn for i in layers}
    bag = Bag([Mamba2Mixer(orig[i]) for i in layers])
    bag.load_weights(os.path.join(OUT_BASE, a.init_from, "mixers.safetensors"))
    mx.eval(bag.parameters())
    if a.small_only:
        for m in bag.mixers:            # khoá 4 ma trận lớn (q,k,v,o ~47M/tầng); chỉ chỉnh dt/A/D/conv/chuẩn-hoá
            for sub in (m.q, m.k, m.v, m.o):
                sub.freeze()
    ntr = sum(v.size for _, v in tree_flatten(bag.trainable_parameters()))
    log(f"KD đầu-cuối · tầng {layers} · khởi từ {a.init_from} · L={a.L}×{a.accum} steps={a.steps} lr={a.lr} · train {ntr/1e6:.2f}M tham số · neo P1 λ={a.anchor}")
    wu = max(1, min(20, a.steps // 8))
    sched = optim.join_schedules([optim.linear_schedule(0, a.lr, wu), optim.cosine_decay(a.lr, a.steps - wu, a.lr * 0.1)], [wu])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)

    from mlx_lm.models.gemma4_text import logit_softcap
    fdt = F32 if a.fp32 else mx.bfloat16

    def fwd(toks):
        """forward tự viết: dòng dư (residual) ở fp32 — ở bf16 gradient qua 48 tầng nhiễu tới mức KHÔNG còn hướng giảm."""
        h = (lm.embed_tokens(toks) * lm.embed_scale).astype(fdt)
        m = lm._make_masks(h, [None] * len(lm.layers))
        for i, l in enumerate(lm.layers):
            h, _, _ = l(h, m[i], None)
        return logit_softcap(model.language_model.final_logit_softcapping, lm.embed_tokens.as_linear(lm.norm(h)))

    def kl_loss(bg, toks, lt):
        for j, i in enumerate(layers):
            lm.layers[i].self_attn = Swap(bg.mixers[j])
        ls = fwd(toks)
        pt = nn.log_softmax(lt.astype(F32), axis=-1)
        ps = nn.log_softmax(ls.astype(F32), axis=-1)
        return (mx.exp(pt) * (pt - ps)).sum() / (toks.shape[0] * toks.shape[1])

    from mlx.utils import tree_map
    vg = nn.value_and_grad(bag, kl_loss)
    mt = os.path.join(ROOT, "data/multiturn/train.jsonl")
    val = [mx.array(q)[None] for q in pack(tok, jsonl_chats(tok, mt, 200, from_end=True), a.L * 2, 4)]
    for i in layers:
        lm.layers[i].self_attn = orig[i]
    val_t = [mx.stop_gradient(fwd(v)) for v in val]
    mx.eval(val_t)

    def val_kl():
        r = sum(kl_loss(bag, v, t).item() for v, t in zip(val, val_t)) / len(val)
        for i in layers:
            lm.layers[i].self_attn = orig[i]
        return r

    vk = val_kl()
    log(f"kd VAL KL bước 0 = {vk:.4f}")
    hist_val = [{"step": 0, "val_kl": vk}]
    stream = train_stream(tok, a.L, a.B)
    hist, t0, ema = [], time.time(), None
    for step in range(1, a.steps + 1):
        gacc, lsum = None, 0.0
        for _ in range(a.accum):
            toks = next(stream)
            for i in layers:
                lm.layers[i].self_attn = orig[i]
            lt = mx.stop_gradient(fwd(toks)); mx.eval(lt)
            loss, g = vg(bag, toks, lt)
            gacc = g if gacc is None else tree_map(lambda x, y: x + y, gacc, g)
            mx.eval(gacc, loss)
            lsum += loss.item()
        gacc = tree_map(lambda x: x / a.accum, gacc)
        if a.anchor > 0:                 # neo P1: Mamba vẫn phải giống attention TẠI tầng của nó
            caps = capture(model, toks, set(layers)); mx.eval(caps)
            def anc(bg):
                return sum(nmse_post(lm.layers[i], bg.mixers[j](caps[i][0]), caps[i][1]) for j, i in enumerate(layers))
            av, ag = nn.value_and_grad(bag, anc)(bag)
            gacc = tree_map(lambda x, y: x + a.anchor * y, gacc, ag)
            mx.eval(gacc)
        opt.update(bag, gacc)
        mx.eval(bag.parameters(), opt.state)
        lv = lsum / a.accum
        ema = lv if ema is None else 0.9 * ema + 0.1 * lv
        if True:
            el = time.time() - t0
            hist.append({"step": step, "kl": lv, "kl_ema": ema})
            log(f"kd step {step}/{a.steps} KL {lv:.4f} (ema {ema:.4f}) · {step*a.L*a.B*a.accum/el:.0f} tok/s · {el/60:.1f} phút · mem {mx.get_peak_memory()/1e9:.1f} GB")
        if step % max(1, a.steps // 8) == 0 or step == a.steps:
            vk = val_kl(); hist_val.append({"step": step, "val_kl": vk})
            log(f"kd VAL KL bước {step} = {vk:.4f}")
            bag.save_weights(os.path.join(OUT, f"mixers_s{step}.safetensors"))
            bag.save_weights(os.path.join(OUT, "mixers.safetensors"))
            json.dump({"layers": layers, "hist": hist, "val": hist_val, "L": a.L, "accum": a.accum, "steps": step, "init_from": a.init_from, "fp32": a.fp32}, open(os.path.join(OUT, "train_log.json"), "w"))
    for i in layers:
        lm.layers[i].self_attn = orig[i]
    log("XONG kd")


def kl_stats(lo, ls, toks):
    """lo, ls logits (1,L,V) → KL(gốc‖trò) trung bình, đồng ý top-1, NLL gốc, NLL trò."""
    L = lo.shape[1]
    kl = ag = nll_o = nll_s = 0.0
    tgt = toks[0, 1:]
    for s in range(0, L - 1, 512):
        e = min(s + 512, L - 1)
        po = nn.log_softmax(lo[0, s:e].astype(F32), axis=-1)
        ps = nn.log_softmax(ls[0, s:e].astype(F32), axis=-1)
        kl += (mx.exp(po) * (po - ps)).sum().item()
        ag += (po.argmax(-1) == ps.argmax(-1)).sum().item()
        t = tgt[s:e]
        nll_o += -mx.take_along_axis(po, t[:, None], axis=-1).sum().item()
        nll_s += -mx.take_along_axis(ps, t[:, None], axis=-1).sum().item()
    n = L - 1
    return kl / n, ag / n, nll_o / n, nll_s / n


def cmd_eval(a):
    layers = [int(x) for x in a.layers.split(",")]
    model, tok = load_model()
    lm = lm_inner(model)
    mix = [Mamba2Mixer(lm.layers[i].self_attn) for i in layers]
    bag = Bag(mix)
    if not a.init_only:
        bag.load_weights(os.path.join(OUT, "mixers.safetensors"))
    mx.eval(bag.parameters())
    orig = {i: lm.layers[i].self_attn for i in layers}

    def setv(v):
        for j, i in enumerate(layers):
            if v == "goc":
                lm.layers[i].self_attn = orig[i]
            elif v == "bo_" + str(i):
                lm.layers[i].self_attn = Swap(None, zero=True)
            elif v == "mamba_" + str(i) or v == "mamba_all":
                lm.layers[i].self_attn = Swap(bag.mixers[j])
            elif v == "bo_all":
                lm.layers[i].self_attn = Swap(None, zero=True)
            else:
                lm.layers[i].self_attn = orig[i]

    variants = ([] if a.only_all else [f"bo_{i}" for i in layers] + [f"mamba_{i}" for i in layers]) + ["mamba_all", "bo_all"]
    n = a.n
    mt = os.path.join(ROOT, "data/multiturn/train.jsonl")
    sets = {
        "chat_vi": pack(tok, jsonl_chats(tok, mt, 400, from_end=True), 2048, n),
        "vanphong_vi": pack(tok, jsonl_chats(tok, os.path.join(ROOT, "data/office_v2/val.jsonl"), 300), 2048, n),
        "code_en": pack(tok, jsonl_chats(tok, os.path.join(ROOT, "data/code_sft/val.jsonl"), 400), 2048, n),
        "dai_8k": pack(tok, jsonl_chats(tok, mt, 400), 8192, max(2, n // 4)),
    }
    res = {}
    for name, seqs in sets.items():
        acc = {v: [0, 0, 0, 0] for v in variants}
        base_nll = 0.0
        t0 = time.time()
        for sq in seqs:
            toks = mx.array(sq)[None]
            setv("goc")
            lo = model(toks); mx.eval(lo)
            for v in variants:
                setv(v)
                ls = model(toks); mx.eval(ls)
                kl, ag, no, ns = kl_stats(lo, ls, toks)
                r = acc[v]; r[0] += kl; r[1] += ag; r[2] += no; r[3] += ns
                del ls
            setv("goc")
            del lo
        k = len(seqs)
        res[name] = {v: {"KL": acc[v][0] / k, "top1_dongy": acc[v][1] / k, "ppl_goc": math.exp(acc[v][2] / k), "ppl": math.exp(acc[v][3] / k)} for v in variants}
        log(f"── {name}: {k} chuỗi × {len(seqs[0])} token · {time.time()-t0:.0f}s · ppl gốc {res[name][variants[0]]['ppl_goc']:.3f}")
        for v in variants:
            r = res[name][v]
            log(f"   {v:10s} KL {r['KL']:.4f} · top1 đồng-ý {r['top1_dongy']*100:.1f}% · ppl {r['ppl']:.3f}")
    tag = "init" if a.init_only else "train"
    json.dump(res, open(os.path.join(OUT, f"eval_{tag}.json"), "w"), ensure_ascii=False, indent=1)
    log("XONG eval")


def cmd_gen(a):
    """sinh thử trên vài ca có thật: gốc ↔ các tầng đã thay (trạng thái SSM thay KV ở tầng đó)."""
    from mlx_lm import generate
    layers = [int(x) for x in a.layers.split(",")]
    model, tok = load_model()
    lm = lm_inner(model)
    orig = {i: lm.layers[i].self_attn for i in layers}
    bag = Bag([Mamba2Mixer(orig[i]) for i in layers])
    bag.load_weights(os.path.join(OUT, "mixers.safetensors"))
    cases = {c["id"]: c for c in json.load(open(os.path.join(ROOT, a.cases)))}
    outs = []
    for cid in a.ids.split(","):
        c = cases[cid]
        msgs = [{"role": "user", "content": c["turns"][0]}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        row = {"id": cid, "prompt": c["turns"][0]}
        for v in ("goc", "mamba"):
            cache = model.make_cache()
            for j, i in enumerate(layers):
                if v == "goc":
                    lm.layers[i].self_attn = orig[i]
                else:
                    lm.layers[i].self_attn = Swap(bag.mixers[j]); cache[i] = SSMCache()
            t0 = time.time()
            row[v] = generate(model, tok, prompt, max_tokens=a.max_tokens, verbose=False, prompt_cache=cache)
            row[v + "_s"] = round(time.time() - t0, 1)
            log(cid, v, row[v + "_s"], "s")
        for i in layers:
            lm.layers[i].self_attn = orig[i]
        outs.append(row)
        json.dump(outs, open(os.path.join(OUT, f"gen_{a.tag}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--layers", default="1,14,30")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--init-only", action="store_true")
    ap.add_argument("--cases", default="data/eval_clean/real_vto_28.json")
    ap.add_argument("--ids", default="v01_email_nghi,o01_countifs,t01_weather")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--tag", default="train")
    ap.add_argument("--run", default="chat")
    ap.add_argument("--init-from", default="chat")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--only-all", action="store_true")
    ap.add_argument("--small-only", action="store_true")
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--fp32", type=int, default=1)
    a = ap.parse_args()
    OUT = os.path.join(OUT_BASE, a.run)
    os.makedirs(OUT, exist_ok=True)
    {"selftest": cmd_selftest, "train": cmd_train, "eval": cmd_eval, "gen": cmd_gen, "kd": cmd_kd}[a.cmd](a)
