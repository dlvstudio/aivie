"""TK_105 gỡ lỗi gđ1 (CPU): token mềm có mang thông tin không? So loss bản Anh khi Zamba2 nhận:
   trained (cầu đã học) · shuffled (token mềm của câu KHÁC) · zero (token mềm = 0) · vi_raw (câu Việt thô).
Nếu trained ≈ shuffled → cầu không truyền ý (học thành "đoán mò"). In thêm top-5 token đầu tiên khi sinh."""
import sys, torch, random
sys.path[:0] = ["scripts", "src"]
import torch.nn.functional as F
import tk105_bridge as T

class A: base = "ckpt/hybrid_1p3b_chat_v2.pt"; layer = 23
P = T.Parts.__new__(T.Parts)
import expert_ffn_1p3b as E
from transformers import AutoModelForCausalLM, AutoTokenizer
E.BASE = A.base
P.ours, _ = E.load_trunk("cpu", kernel=False); P.ours.eval()
P.tok_vi = E.tok_(); P.L = A.layer; P.dev = "cpu"
P.ztok = AutoTokenizer.from_pretrained(T.ZAMBA)
P.z = AutoModelForCausalLM.from_pretrained(T.ZAMBA, dtype=torch.bfloat16, device_map="cpu").eval()
P.zemb = P.z.get_input_embeddings()
enc = lambda s: P.ztok(s, add_special_tokens=False).input_ids
P.pre = enc("<|im_start|>user\n"); P.suf = enc("\nTranslate the text above into English.<|im_end|>\n<|im_start|>assistant\n"); P.end = enc("<|im_end|>")
P.emb_rms = float(P.zemb.weight.float().pow(2).mean(-1).sqrt().mean())
print("end ids", P.end, "pre", P.pre, "emb_rms", P.emb_rms, flush=True)

def hidden_vi(texts):
    ids = [P.tok_vi.encode(t)[:96] for t in texts]
    Tm = max(len(x) for x in ids)
    X = torch.zeros(len(ids), Tm, dtype=torch.long)
    for i, x in enumerate(ids): X[i, :len(x)] = torch.tensor(x)
    with torch.no_grad():
        x = P.ours.emb(X)
        for i, (blk, nm) in enumerate(zip(P.ours.blocks, P.ours.norms)):
            if i > P.L: break
            x = x + blk(nm(x))
    return x.float(), [len(x) for x in ids]

br = T.Bridge(emb_rms=P.emb_rms); br.load_state_dict(torch.load("ckpt/tk105/bridge_in.pt", map_location="cpu")); br.eval()
print("scale học được", float(br.scale), flush=True)
ps = T.pairs("test")[:8]
vis, ens = [v for v, _ in ps], [e for _, e in ps]
with torch.no_grad():
    h, lens = hidden_vi(vis)
    soft = br(h)
    print("rms token mềm", float(soft.pow(2).mean(-1).sqrt().mean()), "· rms embedding thật", P.emb_rms, flush=True)
    # độ giống: token mềm gần embedding thật nào nhất?
    W = P.zemb.weight.float(); Wn = W / W.norm(dim=-1, keepdim=True)
    s0 = soft[0, :lens[0]]; sim = (s0 / s0.norm(dim=-1, keepdim=True)) @ Wn.T
    print("token mềm câu 0 gần nhất với:", [P.ztok.decode([int(i)]) for i in sim.argmax(-1)], "· cos", [round(float(v), 2) for v in sim.max(-1).values], flush=True)
    def loss_of(soft_, raw=None):
        emb, mask, Y = P.build(soft_, lens if raw is None else [0]*len(vis), targets=ens, raw_vi=raw)
        lo = P.z(inputs_embeds=emb, attention_mask=mask, use_cache=False).logits
        return float(F.cross_entropy(lo[:, :-1].float().reshape(-1, lo.size(-1)), Y[:, 1:].reshape(-1), ignore_index=-100))
    print("loss trained ", loss_of(soft), flush=True)
    print("loss shuffled", loss_of(soft.roll(1, 0)), flush=True)
    print("loss zero    ", loss_of(torch.zeros_like(soft)), flush=True)
    print("loss vi_raw  ", loss_of(None, raw=vis), flush=True)
    emb, mask, _ = P.build(soft, lens)
    lo = P.z(inputs_embeds=emb, attention_mask=mask, use_cache=False).logits
    for b in range(3):
        top = lo[b, int(mask[b].sum()) - 1].float().softmax(-1).topk(5)
        print("VI:", vis[b][:60], "| top5:", [(P.ztok.decode([int(i)]), round(float(p), 2)) for p, i in zip(top.values, top.indices)], flush=True)
