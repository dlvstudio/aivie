"""Bàn giao span: W_skip khởi (grounded) → nếu ngắn/cụt, causal-LM BIG nối dài
liên tục (repetition-penalty). W_skip lo ĐÚNG mở đầu, big-LM lo HAY phần đuôi."""
import sys, torch
sys.path.insert(0,"src")
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.paths import viz_default_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate, set_knn_hook
from nn_aivie.hca.ctx_encoder import load_ctx_encoder, MAXLEN
state,_=load_checkpoint(viz_default_checkpoint()); vocab=state.vocab
eos=vocab._token_to_id.get("</s>") or vocab._token_to_id.get("<eos>")
enc=load_ctx_encoder("checkpoints/ctx_encoder_causal_big.pt"); m=enc._m; t=enc._t; V=32002; W=m.emb.weight[:V]
@torch.no_grad()
def extend(ids, mx=24, rep=1.6):
    ids=list(ids); bg=set()
    for _ in range(mx):
        e=ids[-MAXLEN:]; x=t.zeros(1,MAXLEN,dtype=t.long); x[0,:len(e)]=t.tensor(e)
        pm=t.arange(MAXLEN)[None]>=len(e)
        h=m.hidden(x,pm,causal=True)[0,len(e)-1]; lo=(h@W.T).clone()
        for tk in set(ids[-12:]): lo[tk]/=rep
        for (a,b) in bg:
            if a==ids[-1]: lo[b]-=1e4
        lo[ids[-1]]-=1e4
        nx=int(lo.argmax())
        if eos is not None and nx==eos: break
        bg.add((ids[-1],nx)); ids.append(nx)
    return ids
def wskip(p,mx=24):
    set_knn_hook(None); return [int(x) for x in generate(state,p,max_len=mx,config=HCAConfig())]
for p in ["sắt và","dòng sông chảy","mùa hè trời","cây cối cần ánh","trong rừng có nhiều","biển có rất nhiều","chó là","nước là chất"]:
    ids=wskip(p); base=vocab.detokenize(ids)
    plen=len(list(vocab.encode_prompt(p))); genlen=len(ids)-plen
    ho=vocab.detokenize(extend(ids)) if genlen<6 else base+"  [W_skip đủ dài, không nối]"
    print(f"\n=== «{p}» (W_skip sinh {genlen} token)",flush=True)
    print(f"  W_skip: {base}",flush=True)
    print(f"  +NỐI  : {ho}",flush=True)
print("\n### hết",flush=True)
