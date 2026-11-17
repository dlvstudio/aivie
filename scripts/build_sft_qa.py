"""§TK_095 — đóng gói data SFT hỏi-đáp cho Mamba (BPE bpe_v1, Hỏi:/Đáp: là special).

Stream: "Hỏi: {q}\nĐáp: {a}<EOS>" nối liền → chunk 128-token (không padding).
Copy E từ enc_data_viwiki để loader lấy đúng V=32002 (E0 không dùng khi --init-from).
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np
from nn_aivie.core.tokenizer import BPETokenizer

CTX = 128
t = BPETokenizer.load("checkpoints/bpe_v1.json")
EOS = t.token_to_id["<EOS>"]
print("EOS id =", EOS)

stream = []
n = 0
with open("data/lora_vi_div/train.jsonl") as f:
    for line in f:
        m = json.loads(line)["messages"]
        q = m[0]["content"].strip()
        a = m[1]["content"].strip()
        text = f"Hỏi: {q}\nĐáp: {a}"
        ids = list(t.encode(text)) + [EOS]
        stream.extend(ids)
        n += 1
print(f"{n} cặp Q&A → {len(stream)} token (trung bình {len(stream)/n:.0f} tok/cặp)")

# chunk 128, bỏ dư
nrow = len(stream) // CTX
X = np.array(stream[:nrow * CTX], dtype=np.int32).reshape(nrow, CTX)
L = np.full(nrow, CTX, dtype=np.int32)
E = np.load("checkpoints/enc_data_viwiki.npz")["E"]   # (32002,96) — chỉ để loader lấy V
np.savez("checkpoints/sft_qa.npz", X=X, L=L, E=E)
print(f"LƯU checkpoints/sft_qa.npz: X={X.shape} L={L.shape} E={E.shape}")
# in thử 1 chunk decode
print("--- mẫu chunk[0] decode ---")
print(repr(t.decode([int(x) for x in X[0]]))[:300])
