"""§TK_059 — SO SÁNH encoder masked vs causal qua kNN.

Soi top-kNN cho cùng ngữ cảnh + bits/token kNN-LM held-out. Cổng nghiệm thu:
causal cho top-kNN NỘI DUNG (thay hư từ "một"/"là" + nhiễu chéo miền) và bits thấp hơn.

    PYTHONPATH=src python scripts/eval_causal_knn.py
"""
import sys, numpy as np
sys.path.insert(0, "src")
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm

st, _ = load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); vocab = st.vocab
PAIRS = [
    ("MASKED", "checkpoints/ctx_encoder.pt", "checkpoints/net_ls_concept_conf.datastore.npz"),
    ("CAUSAL", "checkpoints/ctx_encoder_causal.pt", "checkpoints/net_ls_concept_conf.datastore_causal.npz"),
]
CTX = ["sông và", "núi và", "mặt trời và", "cây và", "sắt và", "chó là", "hổ là", "nước sôi ở"]
kl = {}
for name, enc_p, ds_p in PAIRS:
    enc = load_ctx_encoder(enc_p); k = load_knn_lm(enc, ds_p)
    kl[name] = k
    print(f"[{name}] encoder={'OK' if enc else None} knn={'OK' if k else None}", flush=True)

for text in CTX:
    ids = list(vocab.encode_prompt(text))
    print(f"\n=== «{text}»", flush=True)
    for name, *_ in PAIRS:
        k = kl.get(name)
        if k is None: continue
        P = k.next_dist(ids, len(vocab), k=64, T=0.08)
        if P is None: print(f"  {name}: None"); continue
        top = np.argsort(P)[::-1][:10]
        toks = [f"{vocab.detokenize([int(t)]).strip()}·{P[t]:.2f}" for t in top]
        print(f"  {name}: " + "  ".join(toks), flush=True)
print("\n### hết", flush=True)
