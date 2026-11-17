"""§TK_059 — ENCODER NGỮ CẢNH NHÂN-QUẢ. Thay masked-LM bằng dự đoán TOKEN KẾ từ
TRÁI-ngữ-cảnh (mask nhân quả) → `h` sắc cho kNN-LM sinh (TK_056).

Lý do (đo ở TK_056): encoder masked-train dùng causal-mask lúc suy là MISMATCH →
top-kNN bị tần suất chi phối ("một"/"là") + nhiễu chéo miền. Train ĐÚNG nhân quả
→ `h[i]` học chính việc dự đoán token[i+1] → kNN truy láng-giềng NỘI DUNG.

KHÔNG ghi đè ctx_encoder.pt (QA dùng forward bidirectional). Lưu riêng
checkpoints/ctx_encoder_causal.pt; kiến trúc (emb/pos/enc/norm) tương thích
`ctx_encoder._Enc` nên nạp lại drop-in, `causal_last` chạy đúng.

    PYTHONPATH=src python scripts/train_ctx_encoder_causal.py --steps 8000 --d 128 --layers 3
"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
T0 = time.time()
SP = "checkpoints"  # bền; scratchpad hay bị xoá


class Encoder(nn.Module):
    def __init__(self, V, d, E0, maxlen, nlayer=3, nhead=4):
        super().__init__()
        self.emb = nn.Embedding(V + 1, d)          # +1 giữ tương thích (MASK cũ, không dùng)
        if E0 is not None:
            with torch.no_grad():
                k = min(d, E0.shape[1])
                self.emb.weight[:V, :k] = torch.tensor(E0[:, :k])
        self.pos = nn.Embedding(maxlen, d)
        L = nn.TransformerEncoderLayer(d, nhead, 4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(L, nlayer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.V = V; self.d = d

    def hidden(self, x, pad, causal=True):
        p = torch.arange(x.shape[1], device=x.device)
        h = self.emb(x) + self.pos(p)[None]
        cm = None
        if causal:
            cm = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), 1)
        return self.norm(self.enc(h, mask=cm, src_key_padding_mask=pad))

    def logits(self, h):
        return h @ self.emb.weight[:self.V].T       # weight-tied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=0, help="số bước lr tăng tuyến tính 0→lr (ổn định net sâu)")
    ap.add_argument("--amp", action="store_true", help="mixed-precision (chỉ CUDA) — nhanh 2-3× + tiết kiệm VRAM")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation: batch hiệu dụng = bs×accum (model lớn/RAM nhỏ)")
    ap.add_argument("--coherence", type=float, default=0.0, help="§TK_084 hệ số coherence loss (0=tắt): phạt h_t lệch centroid chủ đề (k token đầu) → giữ chủ đề trong weights, chống trôi")
    ap.add_argument("--coh-k", type=int, default=8, help="số token đầu định nghĩa chủ đề cho coherence loss")
    ap.add_argument("--init-from", default=None, help="warm-start từ checkpoint .pt (continued pretraining; kiến trúc lấy từ đó)")
    ap.add_argument("--clean-mask", default=f"{SP}/clean_mask.npy")
    ap.add_argument("--data", default=f"{SP}/enc_data.npz")
    ap.add_argument("--out", default="checkpoints/ctx_encoder_causal.pt")
    a = ap.parse_args()
    # ưu tiên CUDA (cloud GPU) > MPS (Mac) > CPU
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    use_amp = a.amp and dev == "cuda"
    dat = np.load(a.data); X = dat["X"]; L = dat["L"]; E0 = dat["E"]
    V = E0.shape[0]; MAXLEN = X.shape[1]
    pool = np.arange(len(X))
    try:
        cm = np.load(a.clean_mask); pool = np.where(cm)[0]
        print(f"[{time.time()-T0:.0f}s] CORPUS SẠCH: {len(pool)}/{len(X)} câu", flush=True)
    except Exception:
        print(f"[{time.time()-T0:.0f}s] không có clean_mask → toàn bộ {len(X)} câu", flush=True)
    rng = np.random.default_rng(0); rng.shuffle(pool)
    n_ho = max(400, len(pool) // 20)
    ho = pool[:n_ho]; tr = pool[n_ho:]
    print(f"[{time.time()-T0:.0f}s] dev={dev} V={V} train={len(tr)} held-out={len(ho)} d={a.d} L={a.layers}", flush=True)
    Xt = torch.tensor(X, dtype=torch.long); Lt = torch.tensor(L, dtype=torch.long)
    # §TK_072 — WARM-START (continued pretraining): nạp trọng số checkpoint cũ thay
    # vì train from-scratch. Kiến trúc (d/layers/heads/maxlen) LẤY TỪ checkpoint để
    # khớp; --d/--layers CLI bị bỏ qua nếu lệch. Giữ kiến thức cũ + thêm data mới.
    if a.init_from:
        obj = torch.load(a.init_from, map_location="cpu")
        c = obj["cfg"]
        if (c["d"], c["layers"], c.get("maxlen", MAXLEN)) != (a.d, a.layers, MAXLEN):
            print(f"[{time.time()-T0:.0f}s] ⚠ ép kiến trúc theo checkpoint: d={c['d']} L={c['layers']} maxlen={c.get('maxlen',MAXLEN)} (data maxlen={MAXLEN})", flush=True)
        a.d = c["d"]; a.layers = c["layers"]; a.heads = c["heads"]
        m = Encoder(V, a.d, E0, MAXLEN, a.layers, a.heads).to(dev)
        m.load_state_dict(obj["state"])
        print(f"[{time.time()-T0:.0f}s] WARM-START từ {a.init_from} (d={a.d} L={a.layers})", flush=True)
    else:
        m = Encoder(V, a.d, E0, MAXLEN, a.layers, a.heads).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    nparam = sum(p.numel() for p in m.parameters())
    print(f"[{time.time()-T0:.0f}s] model {nparam/1e6:.1f}M tham số, init_E=partial({min(a.d,E0.shape[1])}d)", flush=True)

    def batch(ids):
        x = Xt[ids].to(dev); ln = Lt[ids].to(dev)
        pad = torch.arange(MAXLEN, device=dev)[None] >= ln[:, None]
        return x, ln, pad

    @torch.no_grad()
    def ho_bits(cap=4000):
        m.eval(); tot = 0.0; nn_ = 0
        for i in range(0, min(len(ho), cap), a.bs):
            ids = ho[i:i + a.bs]
            x, ln, pad = batch(ids)
            h = m.hidden(x, pad, causal=True); lo = m.logits(h)
            # vị trí i (real) dự đoán token[i+1] (real)
            valid = (~pad)[:, :-1] & (~pad)[:, 1:]
            if valid.sum() == 0: continue
            logp = F.log_softmax(lo[:, :-1], -1)
            tgt = x[:, 1:]
            ll = logp.gather(-1, tgt[..., None])[..., 0]
            tot += float((-ll[valid]).sum()); nn_ += int(valid.sum())
        m.train()
        return tot / max(nn_, 1) / np.log(2)   # bits/token

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[{time.time()-T0:.0f}s] bits/token held-out TRƯỚC train: {ho_bits():.3f} (amp={use_amp})", flush=True)
    m.train()
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        last = 0.0
        # gradient accumulation: cộng dồn `accum` micro-batch → batch hiệu dụng bs×accum
        for _ in range(a.accum):
            ids = tr[rng.integers(0, len(tr), a.bs)]
            x, ln, pad = batch(ids)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                h = m.hidden(x, pad, causal=True); lo = m.logits(h)
                valid = (~pad)[:, :-1] & (~pad)[:, 1:]
                if valid.sum() == 0: continue
                ce = F.cross_entropy(lo[:, :-1][valid], x[:, 1:][valid])
                loss = ce
                if a.coherence > 0:                       # §TK_084 coherence loss
                    k = a.coh_k
                    topic = F.normalize(m.emb(x)[:, :k, :].mean(1), dim=-1).detach()  # (B,d) chủ đề = centroid k token đầu
                    cos_t = (F.normalize(h, dim=-1) * topic[:, None, :]).sum(-1)      # (B,L) cos(h_t, chủ đề)
                    pm = (~pad).clone(); pm[:, :k] = False                            # chỉ phạt SAU vùng thiết lập chủ đề
                    if pm.any():
                        loss = ce + a.coherence * (1.0 - cos_t)[pm].mean()
                loss = loss / a.accum
            scaler.scale(loss).backward()
            last += float(loss) * a.accum
        if a.clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip)
        if a.warmup > 0 and step <= a.warmup:      # lr warmup tuyến tính (net sâu)
            for g in opt.param_groups:
                g["lr"] = a.lr * step / a.warmup
        scaler.step(opt); scaler.update()
        if step % a.eval_every == 0:
            print(f"[{time.time()-T0:.0f}s] step {step} loss {last/a.accum:.3f} | held-out bits {ho_bits():.3f}", flush=True)
            torch.save({"cfg": {"d": a.d, "layers": a.layers, "heads": a.heads, "maxlen": MAXLEN},
                        "state": m.state_dict()}, a.out)   # checkpoint tiến trình
    torch.save({"cfg": {"d": a.d, "layers": a.layers, "heads": a.heads, "maxlen": MAXLEN},
                "state": m.state_dict()}, a.out)
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
