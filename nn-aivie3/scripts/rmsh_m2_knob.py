"""M2 knob — cửa-sổ attn NHỎ (W=6): tăng d_state có đẩy được recall của Mamba-state?
So d_state ∈ {16,64,128} trên recall frame-sớm (chỉ Mamba gánh). T=8, hỏi frame 0..5."""
import os, sys, time, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m2_smoke as S                                               # noqa: E402
from nn_aivie3.rmsh import MultimodalRMSH                               # noqa: E402

dev = S.pick_device()
print(f"=== M2 knob · dev={dev} · W=6 · ngẫu-nhiên {1/S.N_COLORS:.1%} ===")
for ds in (16, 64, 128):
    torch.manual_seed(0)
    m = MultimodalRMSH(vocab=S.VOCAB, d=128, n_layer=6, attn_layers=(2, 4), d_state=ds,
                       heads=4, window=6, d_v=128, n_query=4, img=S.IMG, patch=16,
                       max_frames=S.MAXT).to(dev)
    S.train_recall(m, dev, T=8, steps=400, bs=64, qi_lo=0, qi_hi=6)
    acc = S.eval_recall(m, dev, T=8, qi_lo=0, qi_hi=6)
    print(f"  d_state={ds:>3}: acc frame-sớm (chỉ Mamba) = {acc:.1%}  {'↑' if acc>0.5 else '≈chance'}")
