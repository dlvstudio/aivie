import time,sys
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm
T0=time.time()
state,_=load_checkpoint(viz_default_checkpoint())
print(f"[{time.time()-T0:.0f}s] checkpoint: {len(state.parents.parents)} cha, vocab={len(state.vocab._id_to_token) if hasattr(state.vocab,'_id_to_token') else '?'}")
enc=load_ctx_encoder(); knn=load_knn_lm(enc)
print(f"[{time.time()-T0:.0f}s] encoder={'OK' if enc else None} knn={'OK' if knn else None}")
if knn: print(f"   datastore keys={tuple(knn._K.shape)} vals={knn._vals.shape} dev={knn._dev}")
prompts=["sắt và","chó là","nước là","lúa là"]
for p in prompts:
    cfg=HCAConfig()
    set_knn_hook(None)
    t=time.time(); a=state.vocab.detokenize([int(x) for x in generate(state,p,max_len=20,config=cfg)]); ta=time.time()-t
    set_knn_hook(knn,lam=0.35)
    t=time.time(); b=state.vocab.detokenize([int(x) for x in generate(state,p,max_len=20,config=cfg)]); tb=time.time()-t
    set_knn_hook(None)
    print(f"\n=== «{p}»")
    print(f"  KHÔNG kNN [{ta:.2f}s]: {a}")
    print(f"  CÓ   kNN [{tb:.2f}s]: {b}")
print("\n### hết")
