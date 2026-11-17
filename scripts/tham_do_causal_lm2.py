"""Thăm dò 2: causal-LM greedy + repetition penalty + no-repeat-bigram — xem lặp
vòng là bệnh DECODING (sửa được) hay bản thân LM hỏng."""
import sys, numpy as np, torch
sys.path.insert(0,"src")
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.ctx_encoder import load_ctx_encoder, MAXLEN
st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); vocab=st.vocab
enc=load_ctx_encoder("checkpoints/ctx_encoder_causal.pt"); m=enc._m; t=enc._t; V=32002
eos=vocab._token_to_id.get("</s>") or vocab._token_to_id.get("<eos>")
W=m.emb.weight[:V]
@torch.no_grad()
def logits_last(ids):
    e=ids[-MAXLEN:]; x=t.zeros(1,MAXLEN,dtype=t.long); x[0,:len(e)]=t.tensor(e)
    pm=t.arange(MAXLEN)[None]>=len(e)
    h=m.hidden(x,pm,causal=True)[0,len(e)-1]; return (h@W.T)
def gen(prompt, mx=28, rep_pen=1.5, no_bigram=True):
    ids=[int(x) for x in vocab.encode_prompt(prompt)]
    seen_bg=set()
    for _ in range(mx):
        lo=logits_last(ids).clone()
        for tk in set(ids[-12:]): lo[tk]/=rep_pen          # phạt token gần
        if no_bigram and ids:
            prev=ids[-1]
            for (a,b) in seen_bg:
                if a==prev: lo[b]-=1e4                       # cấm lặp bigram
        # cấm lặp token liền kề
        lo[ids[-1]]-=1e4
        nx=int(lo.argmax())
        if eos is not None and nx==eos: break
        if ids: seen_bg.add((ids[-1],nx))
        ids.append(nx)
    return vocab.detokenize(ids)
for p in ["sắt là","chó là","hổ là loài","nước là chất","việt nam là","dòng sông","mặt trời","cây cối cần"]:
    print(f"  «{p}» → {gen(p)}",flush=True)
print("### hết",flush=True)
