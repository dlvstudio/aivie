import sys, numpy as np
sys.path.insert(0,"src")
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm
st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); vocab=st.vocab
dat=np.load("checkpoints/enc_data.npz"); X=dat["X"]; L=dat["L"]; V=dat["E"].shape[0]
clean=np.load("checkpoints/clean_mask.npy")
ids=np.where(clean)[0]; rng=np.random.default_rng(0); rng.shuffle(ids); ho=ids[45000:47000]
PAIRS=[("MASKED","checkpoints/ctx_encoder.pt","checkpoints/net_ls_concept_conf.datastore.npz"),
       ("d128","checkpoints/ctx_encoder_causal.pt","checkpoints/net_ls_concept_conf.datastore_causal.npz"),
       ("BIG-d256","checkpoints/ctx_encoder_causal_big.pt","checkpoints/net_ls_concept_conf.datastore_big.npz")]
pos=[]
for si in ho:
    ln=int(L[si])
    for t in range(ln-1): pos.append((int(si),t))
rng.shuffle(pos); pos=pos[:2500]
print(f"held-out {len(pos)} vị trí\n",flush=True)
KL={}
for name,ep,dp in PAIRS:
    enc=load_ctx_encoder(ep); knn=load_knn_lm(enc,dp); KL[name]=knn
    uni=np.bincount(knn._vals,minlength=V).astype(np.float64); uni/=uni.sum()
    tot=0.0;n=0
    for si,t in pos:
        ctx=[int(X[si,k]) for k in range(t+1)]
        P=knn.next_dist(ctx,V,k=64,T=0.08); tgt=int(X[si,t+1]); pu=uni[tgt]
        pk=P[tgt] if P is not None else 0.0; pf=0.5*pk+0.5*pu
        tot+=-np.log2(max(pf,pu*1e-3,1e-12)); n+=1
    print(f"  {name} kNN-LM bits λ=0.5: {tot/n:.3f}",flush=True)
print("\n### top-kNN (BIG):",flush=True)
for text in ["sông và","chó là","nước sôi ở","cây và","mặt trời và"]:
    P=KL["BIG-d256"].next_dist(list(vocab.encode_prompt(text)),V,k=64,T=0.08)
    top=np.argsort(P)[::-1][:8]
    print(f"  «{text}»: "+"  ".join(f"{vocab.detokenize([int(t)]).strip()}·{P[t]:.2f}" for t in top),flush=True)
print("\n### hết",flush=True)
