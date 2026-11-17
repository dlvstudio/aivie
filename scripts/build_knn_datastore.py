"""§TK_097 — DATASTORE kNN-LM cho "RAG expert tri-thức": lưu (hidden ngữ-cảnh → token-kế-tiếp)
từ kho viwiki, dùng hidden của WIKI-EXPERT (mamba_d256_2k) làm khoá. Truy-xuất k láng-giềng
→ phân-phối token GROUNDED trong văn-bản THẬT (không bịa như expert nơ-ron thuần).

    PYTHONPATH=src .venv/bin/python scripts/build_knn_datastore.py
"""
import sys, re, time, argparse
sys.path.insert(0, "scripts")
import numpy as np, torch
from train_mamba import Mamba
T0 = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("--n-seq", type=int, default=1600)
ap.add_argument("--cap", type=int, default=160_000)
ap.add_argument("--out", default="checkpoints/knn_datastore.npz")
ap.add_argument("--backbone", default="checkpoints/mamba_d256_2k.pt", help="§TK_098 wiki-expert mã-hoá khoá (đổi khi d512)")
A = ap.parse_args()
BACKBONE = A.backbone                        # wiki-expert làm bộ mã-hoá khoá
DATA = "checkpoints/enc_data_viwiki.npz"
N_SEQ = A.n_seq; CAP = A.cap
dev = "mps" if torch.backends.mps.is_available() else "cpu"

obj = torch.load(BACKBONE, map_location="cpu"); c = obj["cfg"]; st = obj["state"]
al = sorted({int(m.group(1)) for k in st for m in [re.match(r"blocks\.(\d+)\.attn\.", k)] if m})
V = np.load(DATA)["E"].shape[0]
m = Mamba(V, c["d"], c["layers"], None, c["d_state"], al, c.get("heads", 4)).to(dev)
m.load_state_dict(st); m.eval()
d = np.load(DATA); X = d["X"][:, :128]; L = d["L"]
keep = np.where(L >= 64)[0]; rng = np.random.default_rng(123); rng.shuffle(keep)   # seed 123 ≠ mix (seed 0) → RỜI eval, tránh rò-rỉ
keep = keep[:N_SEQ]
Xt = torch.tensor(X[keep], dtype=torch.long)

keys, vals = [], []
with torch.no_grad():
    for i in range(0, len(Xt), 16):
        ids = Xt[i:i + 16].to(dev)
        _, h = m(ids, return_hidden=True)          # (b,l,d) hidden = khoá ngữ-cảnh tại vị-trí t
        h = h[:, :-1].reshape(-1, c["d"]).cpu().numpy()      # khoá h_t
        nxt = ids[:, 1:].reshape(-1).cpu().numpy()           # value = token t+1
        keys.append(h.astype(np.float16)); vals.append(nxt.astype(np.int32))
        if sum(len(v) for v in vals) >= CAP:
            break
K = np.concatenate(keys)[:CAP]; Vv = np.concatenate(vals)[:CAP]
np.savez(A.out, keys=K, vals=Vv, d=c["d"], backbone=BACKBONE, n_seq=N_SEQ, seed=123)
print(f"[{time.time()-T0:.0f}s] LƯU {A.out}: keys={K.shape} vals={Vv.shape} (từ wiki-expert, {N_SEQ} seq seed123)", flush=True)
print("### hết", flush=True)
