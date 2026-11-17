import time
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm
state,_=load_checkpoint(viz_default_checkpoint())
enc_m=load_ctx_encoder("checkpoints/ctx_encoder.pt")
knn_m=load_knn_lm(enc_m,"checkpoints/net_ls_concept_conf.datastore.npz")
enc_c=load_ctx_encoder("checkpoints/ctx_encoder_causal.pt")
knn_c=load_knn_lm(enc_c,"checkpoints/net_ls_concept_conf.datastore_causal.npz")
print(f"masked={'OK' if knn_m else None} causal={'OK' if knn_c else None}",flush=True)
def g(p,knn,lam=0.35,widen=0):
    set_knn_hook(knn,lam=lam,widen_k=widen) if knn else set_knn_hook(None)
    r=state.vocab.detokenize([int(x) for x in generate(state,p,max_len=20,config=HCAConfig())])
    set_knn_hook(None); return r
for p in ["sông và","cây và","mặt trời và","núi và","gió và","sắt và","chó là","hổ và"]:
    set_knn_hook(None); _=generate(state,p,max_len=20,config=HCAConfig())  # warm
    print(f"\n=== «{p}»",flush=True)
    print(f"  off           : {g(p,None)}",flush=True)
    print(f"  masked λ.35   : {g(p,knn_m)}",flush=True)
    print(f"  CAUSAL λ.35   : {g(p,knn_c)}",flush=True)
    print(f"  CAUSAL +nới8  : {g(p,knn_c,widen=8)}",flush=True)
    print(f"  CAUSAL λ.5+nới: {g(p,knn_c,lam=0.5,widen=12)}",flush=True)
print("\n### hết",flush=True)
