import sys, torch
sys.path.insert(0,"src")
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.ctx_encoder import load_ctx_encoder, MAXLEN
st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); vocab=st.vocab
eos=vocab._token_to_id.get("</s>") or vocab._token_to_id.get("<eos>")
def mk(path):
    enc=load_ctx_encoder(path); m=enc._m; t=enc._t; V=32002; W=m.emb.weight[:V]
    @torch.no_grad()
    def gen(prompt, mx=30, rep=1.6):
        ids=[int(x) for x in vocab.encode_prompt(prompt)]; bg=set()
        for _ in range(mx):
            e=ids[-MAXLEN:]; x=t.zeros(1,MAXLEN,dtype=t.long); x[0,:len(e)]=t.tensor(e)
            pm=t.arange(MAXLEN)[None]>=len(e)
            h=m.hidden(x,pm,causal=True)[0,len(e)-1]; lo=(h@W.T).clone()
            for tk in set(ids[-12:]): lo[tk]/=rep
            if ids:
                for (a,b) in bg:
                    if a==ids[-1]: lo[b]-=1e4
            lo[ids[-1]]-=1e4
            nx=int(lo.argmax())
            if eos is not None and nx==eos: break
            if ids: bg.add((ids[-1],nx))
            ids.append(nx)
        return vocab.detokenize(ids)
    return gen
gd=mk("checkpoints/ctx_encoder_causal.pt")      # d128
gb=mk("checkpoints/ctx_encoder_causal_big.pt")  # big
for p in ["sắt là","chó là","nước là chất","hổ là loài","dòng sông","việt nam là","cây cối cần","mặt trời"]:
    print(f"\n=== «{p}»",flush=True)
    print(f"  d128: {gd(p)}",flush=True)
    print(f"  BIG : {gb(p)}",flush=True)
print("\n### hết",flush=True)
