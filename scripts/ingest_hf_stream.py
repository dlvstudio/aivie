"""§TK_072 — NẠP FineWeb2-HQ (parquet) → enc_data npz cho causal-LM.

FineWeb2-HQ (epfml/FineWeb2-HQ, config `vie_Latn`) = SOTA web tiếng Việt lọc
embedding-classifier (thắng CulturaX/CC-100/mC4). Mỗi shard parquet ~1.17GB.
Tải 1 shard (hf_hub_download) là đủ vài trăm M token cho Mac.

Đọc cột `text` bằng pyarrow (batch, không nạp cả file vào RAM), tokenize bằng vocab
hiện có, chunk TRONG-CÙNG-BÀI maxlen token (mạch lạc) → npz(X,L,E). E tái dùng.

    # 1) tải 1 shard: hf_hub_download('epfml/FineWeb2-HQ','vie_Latn/000_00000.parquet',
    #    repo_type='dataset', local_dir='data/fineweb2hq')
    PYTHONPATH=src python scripts/ingest_hf_stream.py \\
        --parquet data/fineweb2hq/vie_Latn/*.parquet --maxlen 128 \\
        --max-tokens 200000000 --out checkpoints/enc_data_fw2.npz
"""
import sys, time, argparse, glob
sys.path.insert(0, "src")
import gc; gc.disable()
import numpy as np
import pyarrow.parquet as pq
from nn_aivie.core.checkpoint_io import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True, help="file/glob parquet (cột 'text')")
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=200_000_000)
    ap.add_argument("--min-doc", type=int, default=16)
    ap.add_argument("--min-chunk", type=int, default=24)
    ap.add_argument("--ckpt", default="checkpoints/net_ls_concept_conf.pkl")
    ap.add_argument("--emb", default="checkpoints/enc_data.npz")
    ap.add_argument("--out", default="checkpoints/enc_data_fw2.npz")
    a = ap.parse_args()
    T0 = time.time()

    paths = []
    for g in a.parquet:
        paths.extend(sorted(glob.glob(g)))
    if not paths:
        print("Không tìm thấy parquet", file=sys.stderr); return 2

    st, _ = load_checkpoint(a.ckpt); v = st.vocab
    V = int(st.embedding.matrix.shape[0])
    E = np.load(a.emb)["E"].astype(np.float32)
    print(f"[{time.time()-T0:.0f}s] vocab nạp V={V}; {len(paths)} parquet", flush=True)

    ML = a.maxlen
    rows = []; total = 0; ndoc = 0; stop = False
    for path in paths:
        if stop:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            texts = batch.column("text").to_pylist()
            for text in texts:
                if not text:
                    continue
                ids = [t for t in v.encode_prompt(text) if 0 <= t < V]
                if len(ids) < a.min_doc:
                    continue
                ndoc += 1
                for i in range(0, len(ids), ML):
                    c = ids[i:i + ML]
                    if len(c) < a.min_chunk:
                        break
                    rows.append(c); total += len(c)
                if ndoc % 20000 == 0:
                    print(f"[{time.time()-T0:.0f}s]  {ndoc} doc · {len(rows)} đoạn · {total/1e6:.0f}M token", flush=True)
                if total >= a.max_tokens:
                    stop = True; break
            if stop:
                break

    n = len(rows)
    X = np.zeros((n, ML), np.int32); L = np.zeros(n, np.int32)
    for j, r in enumerate(rows):
        X[j, :len(r)] = r; L[j] = len(r)
    np.savez(a.out, X=X, L=L, E=E)
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}: {n} đoạn, {total/1e6:.1f}M token, "
          f"TB {L.mean():.0f} tok/đoạn (doc {ndoc})", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
