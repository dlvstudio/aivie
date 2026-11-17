"""§TK_059 — kNN-LM bits/token held-out: causal vs masked. Held-out = câu KHÔNG
trong datastore (cùng shuffle seed 0, [45000:47000])."""
import sys, numpy as np
sys.path.insert(0,"src")
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
from nn_aivie.hca.knn_lm import load_knn_lm
dat=np.load("checkpoints/enc_data.npz"); X=dat["X"]; L=dat["L"]; V=dat["E"].shape[0]
clean=np.load("checkpoints/clean_mask.npy")
ids=np.where(clean)[0]; rng=np.random.default_rng(0); rng.shuffle(ids)
ho=ids[45000:47000]  # ngoài datastore
PAIRS=[("MASKED","checkpoints/ctx_encoder.pt","checkpoints/net_ls_concept_conf.datastore.npz"),
       ("CAUSAL","checkpoints/ctx_encoder_causal.pt","checkpoints/net_ls_concept_conf.datastore_causal.npz")]
# vị trí held-out (câu, t) — dự đoán X[si,t+1] từ trái-ngữ-cảnh tới t
pos=[]
for si in ho:
    ln=int(L[si])
    for t in range(ln-1): pos.append((int(si),t))
rng.shuffle(pos); pos=pos[:2500]
print(f"held-out {len(pos)} vị trí\n",flush=True)
for name,enc_p,ds_p in PAIRS:
    enc=load_ctx_encoder(enc_p); knn=load_knn_lm(enc,ds_p)
    uni=np.bincount(knn._vals,minlength=V).astype(np.float64); uni/=uni.sum()
    for lam in (0.3,0.5):
        tot=0.0;n=0
        for si,t in pos:
            ctx=[int(X[si,k]) for k in range(t+1)]
            P=knn.next_dist(ctx,V,k=64,T=0.08)
            tgt=int(X[si,t+1])
            pu=uni[tgt]
            pk=P[tgt] if P is not None else 0.0
            pf=lam*pk+(1-lam)*pu
            tot+=-np.log2(max(pf,pu*1e-3,1e-12)); n+=1
        print(f"  {name} kNN-LM λ={lam}: {tot/n:.3f} bits",flush=True)
print("\n### hết",flush=True)
