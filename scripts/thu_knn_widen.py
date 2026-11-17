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
print(f"[{time.time()-T0:.0f}s] sẵn sàng",flush=True)
def gen(p): return state.vocab.detokenize([int(x) for x in generate(state,p,max_len=22,config=HCAConfig())])
def mode_off(p): set_knn_hook(None); return gen(p)
def mode_reorder(p):
    # ép vocab=None qua monkeypatch lam: gọi blend không vocab
    import nn_aivie.hca.knn_lm as K
    orig=K.KnnLM.blend
    K.KnnLM.blend=lambda self,o,s,l,n,vocab=None,**kw: orig(self,o,s,l,n,vocab=None,**kw)
    set_knn_hook(knn,lam=0.35); r=gen(p); set_knn_hook(None); K.KnnLM.blend=orig; return r
def mode_widen(p): set_knn_hook(knn,lam=0.35); r=gen(p); set_knn_hook(None); return r
prompts=["sắt và","đồng và","mặt trời và","sông và","chó và mèo","biển và","núi và"]
for p in prompts:
    _=mode_off(p)  # warm W_skip
    print(f"\n=== «{p}»",flush=True)
    print(f"  off    : {mode_off(p)}",flush=True)
    print(f"  xáo    : {mode_reorder(p)}",flush=True)
    print(f"  NỚI    : {mode_widen(p)}",flush=True)
print("\n### hết",flush=True)
