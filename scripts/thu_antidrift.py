import sys
sys.path.insert(0,"src")
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm
state,_=load_checkpoint(viz_default_checkpoint()); vocab=state.vocab
eos=vocab._token_to_id.get("</s>") or vocab._token_to_id.get("<eos>")
enc=load_ctx_encoder("checkpoints/ctx_encoder_causal_big.pt")
knn=load_knn_lm(enc,"checkpoints/net_ls_concept_conf.datastore_big.npz")
print(f"knn={'OK' if knn else None}",flush=True)
def wskip(p): set_knn_hook(None); return [int(x) for x in generate(state,p,max_len=20,config=HCAConfig())]
for p in ["sắt và","cây cối cần ánh","dòng sông chảy","biển có rất nhiều"]:
    ids=wskip(p)
    print(f"\n=== «{p}» (W_skip: {vocab.detokenize(ids)})",flush=True)
    print(f"  g=0.0        : {vocab.detokenize(knn.lm_extend(ids,eos_id=eos,knn_ground=0.0))}",flush=True)
    print(f"  g=0.4        : {vocab.detokenize(knn.lm_extend(ids,eos_id=eos,knn_ground=0.4))}",flush=True)
    print(f"  g=0.4 anc=2.5: {vocab.detokenize(knn.lm_extend(ids,eos_id=eos,knn_ground=0.4,anchor=2.5))}",flush=True)
print("\n### hết",flush=True)
