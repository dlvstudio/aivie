import time,numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
import nn_aivie.hca.knn_lm as K
state,_=load_checkpoint(viz_default_checkpoint()); vocab=state.vocab
enc=load_ctx_encoder(); knn=K.load_knn_lm(enc)
print("sẵn sàng",flush=True)
orig=K.KnnLM.blend
def g(p,gain):
    def spy(self,o,s,l,n,vocab=None,**kw): return orig(self,o,s,l,n,vocab=vocab,widen_gain=gain,**kw)
    K.KnnLM.blend=spy; set_knn_hook(knn,lam=0.35)
    r=state.vocab.detokenize([int(x) for x in generate(state,p,max_len=18,config=HCAConfig())])
    set_knn_hook(None); K.KnnLM.blend=orig; return r
for p in ["mặt trời và","núi và","cây và","sông và","gió và"]:
    set_knn_hook(None); _=generate(state,p,max_len=18,config=HCAConfig())  # warm
    print(f"\n=== «{p}»",flush=True)
    for gain in (1.0,2.0,3.5,6.0):
        print(f"  gain={gain}: {g(p,gain)}",flush=True)
print("\n### hết",flush=True)
