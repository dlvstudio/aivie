import time
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm
state,_=load_checkpoint(viz_default_checkpoint())
enc=load_ctx_encoder("checkpoints/ctx_encoder_causal.pt")
knn=load_knn_lm(enc,"checkpoints/net_ls_concept_conf.datastore_causal.npz")
print(f"knn={'OK' if knn else None}",flush=True)
def g(p,mode):
    if mode=="off": set_knn_hook(None)
    elif mode=="hybrid": set_knn_hook(knn,hybrid=True)
    r=state.vocab.detokenize([int(x) for x in generate(state,p,max_len=26,config=HCAConfig())])
    set_knn_hook(None); return r
WEAK=["sắt và","dòng sông chảy","mùa hè trời thường","cây cối cần ánh","trong rừng có nhiều"]
STRONG=["hổ là loài thú","chó là","nước là chất","người nông dân","vàng là kim loại"]
print("\n### YẾU (mong mượt hơn):",flush=True)
for p in WEAK:
    set_knn_hook(None); _=generate(state,p,max_len=26,config=HCAConfig())
    print(f"\n=== «{p}»",flush=True)
    print(f"  off   : {g(p,'off')}",flush=True)
    print(f"  HYBRID: {g(p,'hybrid')}",flush=True)
print("\n### MẠNH (sự thật PHẢI giữ):",flush=True)
for p in STRONG:
    set_knn_hook(None); _=generate(state,p,max_len=26,config=HCAConfig())
    print(f"\n=== «{p}»",flush=True)
    print(f"  off   : {g(p,'off')}",flush=True)
    print(f"  HYBRID: {g(p,'hybrid')}",flush=True)
print("\n### hết",flush=True)
