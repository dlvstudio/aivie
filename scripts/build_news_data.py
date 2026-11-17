"""§TK_096 — dựng data expert NEWS (domain thứ 3): stream báo VnExpress từ HF,
tokenize BPE bpe_v1, chunk 128-token stream (như wiki/QA). Domain khác hẳn để test routing 3-way.

    PYTHONPATH=src .venv/bin/python scripts/build_news_data.py
"""
import sys, itertools
sys.path.insert(0, "src")
import numpy as np
from datasets import load_dataset
from nn_aivie.core.tokenizer import BPETokenizer

CTX = 128
MAX_TOK = 420_000          # ngang cỡ wiki/QA slice
t = BPETokenizer.load("checkpoints/bpe_v1.json")
EOS = t.token_to_id["<EOS>"]

ds = load_dataset("tmnam20/Vietnamese-News", "vnexpress", split="train", streaming=True)
stream, n = [], 0
for row in ds:
    txt = (row.get("text") or "").strip()
    if len(txt) < 200:
        continue
    stream.extend(list(t.encode(txt)) + [EOS])
    n += 1
    if len(stream) >= MAX_TOK:
        break
print(f"{n} bài báo → {len(stream)} token")

nrow = len(stream) // CTX
X = np.array(stream[:nrow * CTX], dtype=np.int32).reshape(nrow, CTX)
L = np.full(nrow, CTX, dtype=np.int32)
E = np.load("checkpoints/enc_data_viwiki.npz")["E"]
np.savez("checkpoints/enc_news.npz", X=X, L=L, E=E)
print(f"LƯU checkpoints/enc_news.npz: X={X.shape}")
print("--- mẫu chunk[0] ---")
print(repr(t.decode([int(x) for x in X[0]]))[:250])
