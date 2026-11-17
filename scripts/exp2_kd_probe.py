"""Thử: gradient KL đầu-cuối (fp32) gộp từ N batch train → đi 1 bước (chuẩn hoá) → KL trên bộ kiểm tra CỐ ĐỊNH có giảm không?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_map, tree_flatten
import exp2_mamba_layer as E
from mlx_lm.models.gemma4_text import logit_softcap
small = "--small" in sys.argv
model, tok = E.load_model(); lm = E.lm_inner(model)
layers = [1, 14, 30]; orig = {i: lm.layers[i].self_attn for i in layers}
bag = E.Bag([E.Mamba2Mixer(orig[i]) for i in layers]); bag.load_weights(os.path.join(E.OUT_BASE, "chat", "mixers.safetensors"))
if small:
    for m in bag.mixers:
        for s_ in (m.q, m.k, m.v, m.o): s_.freeze()
def fwd(t):
    h = (lm.embed_tokens(t) * lm.embed_scale).astype(mx.float32)
    mk = lm._make_masks(h, [None] * 48)
    for i, l in enumerate(lm.layers): h, _, _ = l(h, mk[i], None)
    return logit_softcap(30.0, lm.embed_tokens.as_linear(lm.norm(h)))
def setm(on):
    for j, i in enumerate(layers): lm.layers[i].self_attn = E.Swap(bag.mixers[j]) if on else orig[i]
def kl(bg, t, lt):
    for j, i in enumerate(layers): lm.layers[i].self_attn = E.Swap(bg.mixers[j])
    ls = fwd(t); pt = nn.log_softmax(lt, -1); ps = nn.log_softmax(ls, -1)
    return (mx.exp(pt) * (pt - ps)).sum() / t.shape[1]
mt = os.path.join(E.ROOT, "data/multiturn/train.jsonl")
val = [mx.array(q)[None] for q in E.pack(tok, E.jsonl_chats(tok, mt, 200, from_end=True), 512, 4)]
setm(False); vt = [fwd(v) for v in val]; mx.eval(vt)
def valkl():
    r = sum(kl(bag, v, t).item() for v, t in zip(val, vt)) / len(val); setm(False); return r
st = E.train_stream(tok, int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 512, 1)
vg = nn.value_and_grad(bag, kl)
N = 8; g = None; tr = []
for _ in range(N):
    t = next(st); setm(False); lt = fwd(t); mx.eval(lt)
    l, gi = vg(bag, t, lt); g = gi if g is None else tree_map(lambda a, b: a + b, g, gi); mx.eval(g); tr.append((t, lt))
gn = mx.sqrt(sum((v ** 2).sum() for _, v in tree_flatten(g))).item()
p0 = bag.parameters()
def trkl(): r = sum(kl(bag, t, lt).item() for t, lt in tr) / N; setm(False); return r
E.log(f"{'small' if small else 'full'} · gnorm {gn:.3f} · VAL0 {valkl():.4f} · TRAIN0 {trkl():.4f}")
for eta in (0.01, 0.03, 0.1, 0.3):
    bag.update(tree_map(lambda p, gg: p - (eta / gn) * gg, p0, g))
    E.log(f"bước {eta}: VAL {valkl():.4f} · TRAIN {trkl():.4f}")
bag.update(p0)
