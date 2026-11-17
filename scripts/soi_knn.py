import numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
import nn_aivie.hca.knn_lm as K
state,_=load_checkpoint(viz_default_checkpoint()); vocab=state.vocab
enc=load_ctx_encoder(); knn=K.load_knn_lm(enc)
print("sẵn sàng\n",flush=True)
def probe(text):
    ids=list(vocab.encode_prompt(text))
    P=knn.next_dist(ids,len(vocab),k=64,T=0.08)
    if P is None: print(f"«{text}» → None"); return
    top=np.argsort(P)[::-1][:12]
    toks=[(vocab.detokenize([int(t)]),round(float(P[t]),3)) for t in top]
    print(f"«{text}» →",toks,flush=True)
for t in ["sông và","núi và","mặt trời và","cây và","sắt và","chó là","hổ là","nước sôi ở"]:
    probe(t)
print("\n### hết")
