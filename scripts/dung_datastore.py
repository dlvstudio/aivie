"""§TK_057 — DỰNG DATASTORE (h, next) cho kNN-LM sinh (TK_056).

Quét câu SẠCH, encoder causal-mask → h[i] (LEFT-context) + next=tokens[i+1].
Lưu sidecar .datastore.npz: keys(N,d) float16 (L2-chuẩn hoá), vals(N,) int32.
Kiến trúc encoder đọc từ CFG trong checkpoint (tương thích cả causal TK_059 d=128).

    python scripts/dung_datastore.py [--sp checkpoints] [--sents 45000]
        [--encoder checkpoints/ctx_encoder.pt] [--ra <out.npz>]
"""
import time, argparse
import numpy as np, torch, torch.nn as nn
ap = argparse.ArgumentParser()
ap.add_argument("--sp", default="checkpoints")
ap.add_argument("--sents", type=int, default=45000)
ap.add_argument("--encoder", default="checkpoints/ctx_encoder.pt")
ap.add_argument("--ra", default="checkpoints/net_ls_concept_conf.datastore.npz")
ap.add_argument("--data", default=None, help="npz X,L,E; mặc định {sp}/enc_data.npz")
ap.add_argument("--clean", default=None, help="clean_mask.npy; NONE=toàn bộ")
a = ap.parse_args(); T0 = time.time()
dat = np.load(a.data or f"{a.sp}/enc_data.npz"); X = dat["X"]; L = dat["L"]; E0 = dat["E"]
V = E0.shape[0]; ML = X.shape[1]
_cm = a.clean if a.clean is not None else f"{a.sp}/clean_mask.npy"
try:
    clean = np.load(_cm) if _cm.upper() != "NONE" else np.ones(len(X), bool)
except Exception: clean = np.ones(len(X), bool)
print(f"[{time.time()-T0:.0f}s] {len(X)} câu, V={V}, sạch={int(clean.sum())}", flush=True)


class Enc(nn.Module):
    def __init__(s, V, d, ml, nl, nh):
        super().__init__(); s.emb = nn.Embedding(V + 1, d); s.pos = nn.Embedding(ml, d)
        l = nn.TransformerEncoderLayer(d, nh, 4 * d, batch_first=True, dropout=0.0)
        s.enc = nn.TransformerEncoder(l, nl, enable_nested_tensor=False); s.norm = nn.LayerNorm(d)

    def causal_h(s, x, lens):
        dv = x.device; p = torch.arange(x.shape[1], device=dv); hh = s.emb(x) + s.pos(p)[None]
        cm = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=dv), 1)
        pad = (torch.arange(x.shape[1], device=dv)[None] >= lens[:, None])
        return s.norm(s.enc(hh, mask=cm, src_key_padding_mask=pad))


dev = "mps" if torch.backends.mps.is_available() else "cpu"
obj = torch.load(a.encoder, map_location="cpu")
if isinstance(obj, dict) and "cfg" in obj:
    c = obj["cfg"]; d = c["d"]; nl = c["layers"]; nh = c["heads"]; state = obj["state"]
    ML = int(c.get("maxlen", ML))   # §TK_081 — pos-embedding theo encoder (128), không theo data (48)
else:
    d = 96; nl = 2; nh = 4; state = obj
m = Enc(V, d, ML, nl, nh).to(dev); m.load_state_dict(state); m.eval()
print(f"[{time.time()-T0:.0f}s] encoder {a.encoder} (d={d} L={nl}) nạp, dev={dev}", flush=True)

ids = np.where(clean)[0]
rng = np.random.default_rng(0); rng.shuffle(ids); ids = ids[:a.sents]
keys = []; vals = []
@torch.no_grad()
def run():
    for i in range(0, len(ids), 128):
        b = ids[i:i + 128]
        x = torch.tensor(X[b], dtype=torch.long).to(dev); lens = torch.tensor(L[b]).to(dev)
        h = m.causal_h(x, lens).half().cpu().numpy()
        for j, si in enumerate(b):
            ln = int(L[si])
            for t in range(ln - 1):
                keys.append(h[j, t]); vals.append(int(X[si, t + 1]))
        if i % 6400 == 0: print(f"[{time.time()-T0:.0f}s]  {i}/{len(ids)} câu · {len(keys)} vị trí", flush=True)
run()
K = np.asarray(keys, np.float16); Vn = np.asarray(vals, np.int32)
Kf = K.astype(np.float32); Kf /= np.linalg.norm(Kf, axis=1, keepdims=True) + 1e-9
np.savez(a.ra, keys=Kf.astype(np.float16), vals=Vn)
print(f"[{time.time()-T0:.0f}s] LƯU {a.ra}: {len(K)} vị trí, {K.nbytes//1e6:.0f}MB", flush=True)
print("### hết", flush=True)
