import time
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm
state,_=load_checkpoint(viz_default_checkpoint())
enc=load_ctx_encoder("checkpoints/ctx_encoder_causal_big.pt")
knn=load_knn_lm(enc,"checkpoints/net_ls_concept_conf.datastore_big.npz")
print(f"BIG knn={'OK' if knn else None}",flush=True)
def g(p,mode):
    if mode=="off": set_knn_hook(None)
    else: set_knn_hook(knn,hybrid=True)
    r=state.vocab.detokenize([int(x) for x in generate(state,p,max_len=28,config=HCAConfig())])
    set_knn_hook(None); return r
WEAK=["sắt và","dòng sông chảy","mùa hè trời thường","cây cối cần ánh","trong rừng có nhiều","biển có rất nhiều"]
STRONG=["hổ là loài thú","chó là","nước là chất","vàng là kim loại"]
print("\n### YẾU:",flush=True)
for p in WEAK:
    set_knn_hook(None); _=generate(state,p,max_len=28,config=HCAConfig())
    print(f"=== «{p}»\n  off   : {g(p,'off')}\n  HYBRID: {g(p,'hybrid')}",flush=True)
print("\n### MẠNH (sự thật PHẢI giữ):",flush=True)
for p in STRONG:
    set_knn_hook(None); _=generate(state,p,max_len=28,config=HCAConfig())
    print(f"=== «{p}»\n  off   : {g(p,'off')}\n  HYBRID: {g(p,'hybrid')}",flush=True)
print("\n### hết",flush=True)
