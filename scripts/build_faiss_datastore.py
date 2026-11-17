"""§TK_098 — DATASTORE kNN QUY-MÔ LỚN qua FAISS (ANN) — vượt trần brute-force 2M.

Brute-force cosine trên M4 chết ở ~2M khoá (bộ-nhớ + tốc-độ). FAISS IVFPQ nén vector
(256d→64 byte) + tìm gần-đúng theo cụm → chứa & tra ~8M+ khoá gọn. Khoá = hidden
wiki-expert; value = token-kế-tiếp. Chỉ dùng lúc EVAL (giữ combiner cố-định).

    PYTHONPATH=src .venv/bin/python scripts/build_faiss_datastore.py --n-seq 65000
"""
import sys, re, time, argparse, os
sys.path.insert(0, "scripts")
import numpy as np, torch, faiss
from train_mamba import Mamba
faiss.omp_set_num_threads(1)                          # §TK_098 ép 1 luồng → tránh segfault xung-đột OpenMP với PyTorch
T0 = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("--n-seq", type=int, default=65000)
ap.add_argument("--out", default="checkpoints/knn_faiss_8m")
ap.add_argument("--backbone", default="checkpoints/mamba_d256_2k.pt", help="§TK_099 wiki-expert mã-hoá khoá (đổi khi d512)")
ap.add_argument("--nlist", type=int, default=4096)   # số cụm IVF
ap.add_argument("--mpq", type=int, default=64)       # số sub-vector PQ (256/64=4 chiều/sub)
A = ap.parse_args()
dev = "mps" if torch.backends.mps.is_available() else "cpu"

obj = torch.load(A.backbone, map_location="cpu"); c = obj["cfg"]; st = obj["state"]
al = sorted({int(m.group(1)) for k in st for m in [re.match(r"blocks\.(\d+)\.attn\.", k)] if m})
D = c["d"]; V = np.load("checkpoints/enc_data_viwiki.npz")["E"].shape[0]
m = Mamba(V, D, c["layers"], None, c["d_state"], al, c.get("heads", 4)).to(dev); m.load_state_dict(st); m.eval()
d = np.load("checkpoints/enc_data_viwiki.npz"); X = d["X"][:, :128]; L = d["L"]
keep = np.where(L >= 64)[0]; rng = np.random.default_rng(123); rng.shuffle(keep); keep = keep[:A.n_seq]
Xt = torch.tensor(X[keep], dtype=torch.long)

# thu khoá (float16 để nhẹ RAM) + value — LƯU đĩa trước (resume-able, khỏi thu lại nếu index lỗi)
kpath, vpath = f"{A.out}_keys.npy", f"{A.out}_vals.npy"
if os.path.exists(kpath) and os.path.exists(vpath):
    print(f"[{time.time()-T0:.0f}s] RESUME: nạp khoá đã lưu {kpath}", flush=True)
    Kf16 = np.load(kpath, mmap_mode="r"); Vv = np.load(vpath)
else:
    keys, vals = [], []
    with torch.no_grad():
        for i in range(0, len(Xt), 32):
            ids = Xt[i:i + 32].to(dev)
            _, h = m(ids, return_hidden=True)
            h = torch.nn.functional.normalize(h[:, :-1].reshape(-1, D), dim=-1)   # chuẩn-hoá → cosine=IP
            keys.append(h.cpu().numpy().astype(np.float16))
            vals.append(ids[:, 1:].reshape(-1).cpu().numpy().astype(np.int32))
            if i % 6400 == 0:
                print(f"[{time.time()-T0:.0f}s] thu {sum(len(v) for v in vals)/1e6:.1f}M khoá…", flush=True)
    Kf16 = np.concatenate(keys); Vv = np.concatenate(vals)
    np.save(kpath, Kf16); np.save(vpath, Vv); del keys, vals
    print(f"[{time.time()-T0:.0f}s] LƯU khoá thô {kpath} ({len(Kf16)/1e6:.2f}M)", flush=True)
N = len(Kf16)
print(f"[{time.time()-T0:.0f}s] tổng {N/1e6:.2f}M khoá, dựng IVFPQ nlist={A.nlist} m={A.mpq}", flush=True)

quant = faiss.IndexFlatIP(D)
index = faiss.IndexIVFPQ(quant, D, A.nlist, A.mpq, 8, faiss.METRIC_INNER_PRODUCT)
tr = Kf16[rng.choice(N, min(N, 300_000), replace=False)].astype(np.float32)
index.train(tr); print(f"[{time.time()-T0:.0f}s] train index xong", flush=True)
for i in range(0, N, 500_000):                        # add theo lô (đổi float32 từng lô)
    index.add(Kf16[i:i + 500_000].astype(np.float32))
index.nprobe = 16
faiss.write_index(index, f"{A.out}.index")
np.save(f"{A.out}_vals.npy", Vv)
np.save(f"{A.out}_meta.npy", np.array([D, N], dtype=np.int64))
print(f"[{time.time()-T0:.0f}s] LƯU {A.out}.index ({N/1e6:.2f}M khoá) + vals", flush=True)
print("### hết", flush=True)
