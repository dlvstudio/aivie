import time,numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
import nn_aivie.hca.knn_lm as K
state,_=load_checkpoint(viz_default_checkpoint()); vocab=state.vocab
enc=load_ctx_encoder("checkpoints/ctx_encoder_causal.pt")
knn=K.load_knn_lm(enc,"checkpoints/net_ls_concept_conf.datastore_causal.npz")
print(f"causal knn={'OK' if knn else None}",flush=True)
orig=K.KnnLM.blend
STATS={"added":0,"won":0,"ex":[]}
def spy(self,out_ids,scores,lam,n,vocab=None,**kw):
    live0=np.isfinite(scores)&(scores>0)
    new=orig(self,out_ids,scores,lam,n,vocab=vocab,**kw)
    ln=np.isfinite(new)&(new>0); add=np.flatnonzero(ln&~live0)
    if len(add):
        STATS["added"]+=len(add)
        w=int(np.argmax(np.where(np.isfinite(new),new,-np.inf)))
        if w in add.tolist():
            STATS["won"]+=1
            if len(STATS["ex"])<16: STATS["ex"].append((vocab.detokenize(out_ids[-4:]),vocab.detokenize([w])))
    return new
def g(p,lam=0.35,widen=0,gain=1.0,spyon=False):
    if spyon:
        K.KnnLM.blend=lambda s,o,sc,l,nn,vocab=None,**kw: spy(s,o,sc,l,nn,vocab=vocab,widen_k=widen,widen_gain=gain)
    else:
        K.KnnLM.blend=lambda s,o,sc,l,nn,vocab=None,**kw: orig(s,o,sc,l,nn,vocab=vocab,widen_k=widen,widen_gain=gain)
    set_knn_hook(knn,lam=lam)
    r=vocab.detokenize([int(x) for x in generate(state,p,max_len=24,config=HCAConfig())])
    set_knn_hook(None); K.KnnLM.blend=orig; return r
PROMPTS=["sắt là một kim loại","hổ là loài thú","biển có rất nhiều","mùa hè trời thường",
         "cây cối cần ánh","trong rừng có nhiều","dòng sông chảy","người nông dân"]
for p in PROMPTS:
    set_knn_hook(None); _=generate(state,p,max_len=24,config=HCAConfig())  # warm
    print(f"\n=== «{p}»",flush=True)
    print(f"  off        : {g(p,widen=0)}",flush=True)
    print(f"  reorder    : {g(p,widen=0)}",flush=True)
    print(f"  +nới8      : {g(p,widen=8,spyon=True)}",flush=True)
    print(f"  +nới12 g3  : {g(p,widen=12,gain=3.0,spyon=True)}",flush=True)
print(f"\n>>> nới thêm {STATS['added']} token · THẮNG {STATS['won']}",flush=True)
for ctx,w in STATS["ex"]: print(f"   thắng: …{ctx!r} → +{w!r}",flush=True)
print("### hết",flush=True)
