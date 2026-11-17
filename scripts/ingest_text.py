"""§TK_071 — NẠP VĂN BẢN THÔ → enc_data npz cho causal-LM (giải data-starvation).

Đo (TK_071): data train hiện tại chỉ 9.8M token (parents) trong khi Chinchilla cần
~660M cho model 33M → THIẾU 67×. `data/text_corpus/viwiki.jsonl` = ~838M token
Wikipedia tiếng Việt đang NẰM SẴN. Script này tokenize văn bản thô (mỗi dòng jsonl
là 1 chuỗi text hoặc {"text":...}) bằng CHÍNH vocab hiện có → đóng gói đoạn maxlen
token → npz(X,L,E). E (embedding) tái dùng từ enc_data cũ (cùng vocab).

KHÁC packing từ `parents`: chunk TRONG-CÙNG-MỘT-BÀI (không bắc cầu chéo tài liệu)
→ mỗi đoạn mạch-lạc-chủ-đề, giảm nguồn trôi-nghĩa.

    PYTHONPATH=src python scripts/ingest_text.py \\
        --input data/text_corpus/viwiki.jsonl --maxlen 128 \\
        --max-tokens 100000000 --out checkpoints/enc_data_viwiki.npz
"""
import sys, time, json, argparse
sys.path.insert(0, "src")
import gc; gc.disable()
import numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="1+ file jsonl văn bản thô")
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=100_000_000, help="trần tổng token (cap RAM/thời gian)")
    ap.add_argument("--min-doc", type=int, default=16, help="bỏ doc ngắn hơn (token)")
    ap.add_argument("--min-chunk", type=int, default=24, help="bỏ chunk đuôi ngắn hơn (token)")
    ap.add_argument("--ckpt", default="checkpoints/net_ls_concept_conf.pkl")
    ap.add_argument("--emb", default="checkpoints/enc_data.npz", help="npz để tái dùng E")
    ap.add_argument("--out", default="checkpoints/enc_data_viwiki.npz")
    a = ap.parse_args()
    T0 = time.time()

    st, _ = load_checkpoint(a.ckpt); v = st.vocab
    V = int(st.embedding.matrix.shape[0])
    E = np.load(a.emb)["E"].astype(np.float32)
    print(f"[{time.time()-T0:.0f}s] vocab nạp, V={V}, E={E.shape}", flush=True)

    ML = a.maxlen
    rows = []  # list[list[int]]
    total = 0; ndoc = 0; nskip = 0
    stop = False
    for path in a.input:
        if stop:
            break
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            text = d if isinstance(d, str) else (d.get("text") or d.get("content") or "")
            if not isinstance(text, str) or not text:
                continue
            ids = [t for t in v.encode_prompt(text) if 0 <= t < V]
            if len(ids) < a.min_doc:
                nskip += 1; continue
            ndoc += 1
            # slice TRONG-CÙNG-BÀI: đoạn liên tiếp maxlen token (mạch lạc)
            for i in range(0, len(ids), ML):
                c = ids[i:i + ML]
                if len(c) < a.min_chunk:
                    break
                rows.append(c); total += len(c)
            if ndoc % 20000 == 0:
                print(f"[{time.time()-T0:.0f}s]  {ndoc} doc · {len(rows)} đoạn · {total/1e6:.0f}M token", flush=True)
            if total >= a.max_tokens:
                stop = True; break

    n = len(rows)
    X = np.zeros((n, ML), np.int32); L = np.zeros(n, np.int32)
    for j, r in enumerate(rows):
        X[j, :len(r)] = r; L[j] = len(r)
    np.savez(a.out, X=X, L=L, E=E)
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}: {n} đoạn, {total/1e6:.1f}M token, "
          f"TB {L.mean():.0f} tok/đoạn (doc dùng {ndoc}, bỏ ngắn {nskip})", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
