import time
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm
T0=time.time()
state,_=load_checkpoint(viz_default_checkpoint())
enc=load_ctx_encoder(); knn=load_knn_lm(enc)
print(f"[{time.time()-T0:.0f}s] sẵn sàng knn={'OK' if knn else None}",flush=True)
def g(p,lam):
    set_knn_hook(knn,lam=lam) if lam>0 else set_knn_hook(None)
    ids=[int(x) for x in generate(state,p,max_len=22,config=HCAConfig())]
    set_knn_hook(None)
    return state.vocab.detokenize(ids)
# warm W_skip 1 lần cho mỗi prompt (gọi nguội trước)
prompts=["sắt và","đồng và","hổ và","mặt trời và","sông và","vàng là kim loại","chó và mèo là"]
for p in prompts:
    _=g(p,0.0)  # warm
    print(f"\n=== «{p}»",flush=True)
    for lam in (0.0,0.35,0.6,0.85):
        t=time.time(); out=g(p,lam); dt=time.time()-t
        tag="đếm  " if lam==0 else f"λ={lam}"
        print(f"  {tag} [{dt:.2f}s]: {out}",flush=True)
print("\n### hết",flush=True)
