"""§TK_084 — ĐO DRIFT của causal LM trong KHÔNG-GIAN-EMBEDDING (standalone, công
bằng giữa các checkpoint; không cần vocab/server/datastore).

Ý: lấy đoạn viwiki thật, prompt = P token đầu, để LM sinh tiếp G token (greedy +
rep-penalty), rồi đo bằng E CỐ ĐỊNH (enc_data E, thước đo độc lập model):
  anchor : cos(mean_E(sinh), mean_E(prompt))        cao=bám chủ đề prompt
  selfcoh: cos(mean_E(nửa đầu sinh), mean_E(nửa cuối)) cao=mạch-lạc nội tại
  tail   : cos(mean_E(prompt), mean_E(1/3 cuối sinh)) cao=cuối vẫn đúng chủ đề
  distinct: token khác nhau / tổng (thấp=lặp)

    .venv/bin/python scripts/eval_lm_drift.py --ckpt checkpoints/ctx_encoder_causal_d512mix.pt
"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

T0 = time.time()


class Encoder(nn.Module):
    def __init__(self, V, d, maxlen, nlayer, nhead):
        super().__init__()
        self.emb = nn.Embedding(V + 1, d); self.pos = nn.Embedding(maxlen, d)
        L = nn.TransformerEncoderLayer(d, nhead, 4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(L, nlayer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d); self.V = V

    def hidden(self, x):
        p = torch.arange(x.shape[1], device=x.device)
        h = self.emb(x) + self.pos(p)[None]
        cm = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), 1)
        return self.norm(self.enc(h, mask=cm))

    def next_logits(self, ids):            # ids: list[int] -> logits (V,)
        x = torch.tensor([ids], dtype=torch.long, device=self.emb.weight.device)
        h = self.hidden(x)[0, -1]          # h tại vị trí cuối
        return (h @ self.emb.weight[:self.V].T)


@torch.no_grad()
def generate(next_logits, prompt, G, maxlen, rep_pen=3.0, recent=20):
    ids = list(prompt)
    for _ in range(G):
        ctx = ids[-(maxlen):]
        lo = next_logits(ctx).astype(np.float64)
        for t in set(ids[-recent:]):       # rep-penalty (greedy sạch, tránh lặp)
            lo[t] -= rep_pen
        nx = int(lo.argmax())
        ids.append(nx)
    return ids[len(prompt):]               # chỉ phần sinh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="checkpoints/enc_data_viwiki.npz")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--P", type=int, default=16)   # token prompt
    ap.add_argument("--G", type=int, default=48)   # token sinh
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = np.load(a.data); X = d["X"]; Ln = d["L"]; E = d["E"].astype(np.float32)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)   # E cố định, thước đo
    V = E.shape[0]
    obj = torch.load(a.ckpt, map_location="cpu"); c = obj["cfg"]
    if "d_state" in c:                                    # MAMBA
        import sys; sys.path.insert(0, "scripts")
        from train_mamba import Mamba
        ml = int(c.get("ctx", 128))
        al = [int(x) for x in str(c.get("attn_layers", "")).split(",") if x.strip() != ""]
        m = Mamba(V, c["d"], c["layers"], None, c["d_state"], al, c.get("heads", 4)).to(dev)
        m.load_state_dict(obj["state"]); m.eval()
        def next_logits(ctx):
            x = torch.tensor([ctx], dtype=torch.long, device=dev)
            return m(x)[0, -1].float().cpu().numpy()
        arch = "Mamba"
    else:                                                 # TRANSFORMER
        ml = c["maxlen"]
        m = Encoder(V, c["d"], ml, c["layers"], c["heads"]).to(dev)
        m.load_state_dict(obj["state"]); m.eval()
        next_logits = lambda ctx: m.next_logits(ctx).float().cpu().numpy()
        arch = "Transformer"
    print(f"[{time.time()-T0:.0f}s] {arch} {a.ckpt} (d={c['d']} L={c['layers']}) dev={dev}", flush=True)

    def vec(toks):
        toks = [t for t in toks if 0 <= t < V]
        return En[toks].mean(0) if toks else np.zeros(E.shape[1], np.float32)

    def cos(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        return float(u @ v / (nu * nv)) if nu > 1e-9 and nv > 1e-9 else 0.0

    rng = np.random.default_rng(0)
    rows = rng.choice(np.where(Ln >= a.P + 8)[0], a.n, replace=False)
    anc = []; slf = []; tl = []; dst = []
    for r in rows:
        prompt = [int(t) for t in X[r, :a.P]]
        cont = generate(next_logits, prompt, a.G, ml)
        pv = vec(prompt)
        half = len(cont) // 2
        anc.append(cos(vec(cont), pv))
        slf.append(cos(vec(cont[:half]), vec(cont[half:])))
        tl.append(cos(vec(cont[-len(cont)//3:]), pv))
        dst.append(len(set(cont)) / max(len(cont), 1))
    print(f"[{time.time()-T0:.0f}s] n={a.n} P={a.P} G={a.G}", flush=True)
    print(f"  anchor (bám prompt)     = {np.mean(anc):.3f}  (cao tốt)", flush=True)
    print(f"  tail   (cuối vẫn chủ đề)= {np.mean(tl):.3f}  (cao tốt)", flush=True)
    print(f"  selfcoh(mạch-lạc nội)   = {np.mean(slf):.3f}  (cao tốt)", flush=True)
    print(f"  distinct(đa dạng token) = {np.mean(dst):.3f}", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
