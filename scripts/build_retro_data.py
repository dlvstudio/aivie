"""§TK_074 — DỰNG DATA mini-RETRO từ enc_data_viwiki.npz (numpy thuần, KHÔNG
nn_aivie — tránh xung đột scipy do retro-pytorch cài kèm).

enc_data_viwiki.npz: X(817k,128) mỗi hàng = 128 token LIÊN TIẾP trong 1 bài; E(V,96).
- DB (retrieval): mỗi hàng = 1 đơn-vị-truy = span 128 (chunk[:64] + continuation[64:]).
  key = mean-E của 64 token đầu (retriever đông băng — RETRO dùng BERT, ta mean-E).
- TRAIN seq: hàng KHÁC (rời DB), 128 token = 2 chunk×64.
- Retrieve: mỗi chunk-64 của seq → cosine top-r trong DB (rời DB nên không tự-truy).

Xuất retro_data.npz: seqs(N,128) int32, retrieved(N,2,r,128) int32.

    .venv/bin/python scripts/build_retro_data.py --db 120000 --train 20000 --r 2
"""
import time, argparse
import numpy as np

CH = 64; NL = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="checkpoints/enc_data_viwiki.npz")
    ap.add_argument("--db", type=int, default=120000)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--out", default="checkpoints/retro_data.npz")
    a = ap.parse_args()
    T0 = time.time()

    d = np.load(a.data); X = d["X"].astype(np.int32); E = d["E"].astype(np.float32)
    V = E.shape[0]
    # chỉ giữ hàng đủ dài 128 (đã 128 sẵn), lấy DB rồi TRAIN rời nhau
    n = len(X)
    assert n >= a.db + a.train, f"cần {a.db+a.train} hàng, có {n}"
    DB = X[:a.db]                       # (db,128) span = chunk+continuation
    TR = X[a.db:a.db + a.train]          # (train,128)
    print(f"[{time.time()-T0:.0f}s] X={X.shape} DB={DB.shape} TR={TR.shape} V={V}", flush=True)

    def emb(keys64):                    # (M,64) -> (M,96) chuẩn hoá
        em = E[keys64].mean(1)
        em /= (np.linalg.norm(em, axis=1, keepdims=True) + 1e-9)
        return em.astype(np.float32)

    db_emb = emb(DB[:, :CH])            # (db,96)
    q0 = emb(TR[:, :CH])                # chunk 0 của mỗi seq
    q1 = emb(TR[:, CH:2 * CH])          # chunk 1
    print(f"[{time.time()-T0:.0f}s] nhúng xong; cosine retrieve top-{a.r}…", flush=True)

    def retrieve(Q):                    # (N,96) -> (N,r,128)
        out = np.zeros((len(Q), a.r, NL), np.int32)
        B = 2000
        for i in range(0, len(Q), B):
            sims = Q[i:i + B] @ db_emb.T            # (B,db)
            idx = np.argpartition(sims, -a.r, axis=1)[:, -a.r:]   # top-r (chưa sắp)
            for j in range(idx.shape[0]):
                out[i + j] = DB[idx[j]]
            print(f"[{time.time()-T0:.0f}s]   {min(i+B,len(Q))}/{len(Q)}", flush=True)
        return out

    r0 = retrieve(q0); r1 = retrieve(q1)
    retrieved = np.stack([r0, r1], axis=1)          # (N,2,r,128)
    np.savez(a.out, seqs=TR, retrieved=retrieved)
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}: seqs{TR.shape} retrieved{retrieved.shape}", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
