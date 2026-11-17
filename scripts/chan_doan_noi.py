import time,numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
import nn_aivie.hca.knn_lm as K
state,_=load_checkpoint(viz_default_checkpoint())
enc=load_ctx_encoder(); knn=K.load_knn_lm(enc)
print("sẵn sàng",flush=True)
vocab=state.vocab
# instrument blend: đếm token nới + xem có thắng
STATS={"steps":0,"widen_added":0,"widen_won":0,"examples":[]}
orig=K.KnnLM.blend
def blend_spy(self,out_ids,scores,lam,n,vocab=None,**kw):
    live0=np.isfinite(scores)&(scores>0)
    new=orig(self,out_ids,scores,lam,n,vocab=vocab,**kw)
    STATS["steps"]+=1
    live_new=np.isfinite(new)&(new>0)
    added=np.flatnonzero(live_new & ~live0)
    if len(added):
        STATS["widen_added"]+=len(added)
        win=int(np.argmax(np.where(np.isfinite(new),new,-np.inf)))
        if win in added.tolist():
            STATS["widen_won"]+=1
            if len(STATS["examples"])<12:
                STATS["examples"].append((vocab.detokenize(out_ids[-3:]),vocab.detokenize([win])))
    return new
K.KnnLM.blend=blend_spy
for p in ["mặt trời và","núi và","sông và","biển và","gió và","lửa và","cây và","đá và"]:
    set_knn_hook(None); _=generate(state,p,max_len=18,config=HCAConfig())  # warm
    set_knn_hook(knn,lam=0.35)
    out=state.vocab.detokenize([int(x) for x in generate(state,p,max_len=18,config=HCAConfig())])
    set_knn_hook(None)
    print(f"  «{p}» → {out}",flush=True)
print(f"\nbước blend={STATS['steps']} · token-nới-thêm={STATS['widen_added']} · nới-THẮNG-argmax={STATS['widen_won']}")
for ctx,w in STATS["examples"]: print(f"   nới thắng: …{ctx!r} → +{w!r}")
print("### hết",flush=True)
