"""§TK_056-v2 thăm dò — sinh THẲNG từ logits causal-LM (weight-tied), không W_skip.
Xem mô hình ngôn ngữ transformer tự nó có sinh mạch lạc không → có đáng dựng blend."""
import sys, numpy as np, torch
sys.path.insert(0,"src")
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.ctx_encoder import load_ctx_encoder, MAXLEN
st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); vocab=st.vocab
enc=load_ctx_encoder("checkpoints/ctx_encoder_causal.pt")
m=enc._m; t=enc._t; V=32002
eos=vocab._token_to_id.get("</s>") or vocab._token_to_id.get("<eos>")
W=m.emb.weight[:V]                      # weight-tied logits
@torch.no_grad()
def logits_last(ids):
    e=ids[-MAXLEN:]
    x=t.zeros(1,MAXLEN,dtype=t.long); x[0,:len(e)]=t.tensor(e)
    pm=t.arange(MAXLEN)[None]>=len(e)
    h=m.hidden(x,pm,causal=True)[0,len(e)-1]   # h chuẩn-LayerNorm (khớp train)
    return (h@W.T)
def gen(prompt, mode="greedy", temp=0.8, mx=24, topk=40):
    ids=[int(x) for x in vocab.encode_prompt(prompt)]
    g=torch.Generator().manual_seed(0)
    for _ in range(mx):
        lo=logits_last(ids)
        if mode=="greedy":
            nx=int(lo.argmax())
        else:
            v,i=lo.topk(topk); p=torch.softmax(v/temp,0)
            nx=int(i[torch.multinomial(p,1,generator=g)])
        if eos is not None and nx==eos: break
        ids.append(nx)
    return vocab.detokenize(ids)
PR=["sắt là","chó là","hổ là loài","mặt trời","nước là chất","biển có rất nhiều",
    "cây cối cần","dòng sông","việt nam là","trong rừng"]
for p in PR:
    print(f"\n=== «{p}»",flush=True)
    print(f"  greedy : {gen(p,'greedy')}",flush=True)
    print(f"  T=0.8  : {gen(p,'sample',0.8)}",flush=True)
print("\n### hết",flush=True)
