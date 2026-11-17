"""M2.5 diag2 — cô-lập lỗi định-vị (nhiễu=0). Ba tầng, đâu học đâu kẹt?
(a) linear trên PIXEL thô → quad   (data/nhãn ổn?)
(b) VisionEncoder + mean-pool + linear → quad   (mắt CÓ giữ vị-trí?)
(c) full SalienceNet (resampler)   (đường đầy đủ)"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m25_salience as M                                           # noqa: E402
from nn_aivie3.vision import VisionEncoder                             # noqa: E402
M.N_DIST = 0
dev = M.dev_(); print(f"=== diag2 · dev={dev} · nhiễu=0 · ngẫu-nhiên 25% ===")


def train_eval(net, feat, steps=300, lr=2e-3):
    opt = torch.optim.AdamW(net.parameters(), lr=lr); gen = torch.Generator().manual_seed(1)
    for it in range(steps):
        img, c, quad = M.render(128, gen, dev)
        loss = F.cross_entropy(feat(net, img, c), quad)
        opt.zero_grad(); loss.backward(); opt.step()
    g2 = torch.Generator().manual_seed(999)
    img, c, quad = M.render(512, g2, dev)
    with torch.no_grad():
        acc = feat(net, img, c).argmax(-1).eq(quad).float().mean().item()
    return loss.item(), acc


# (a) linear trên pixel thô
lin = nn.Linear(3 * M.IMG * M.IMG, 4).to(dev)
l, a = train_eval(lin, lambda n, img, c: n(img.flatten(1)))
print(f"  (a) linear-pixel : loss {l:.3f} · acc {a:.1%} {'HỌC' if a > 0.5 else 'KẸT'}")

# (b) encoder + mean-pool grid + linear
class EncPool(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = VisionEncoder(d_v=128, patch=M.CELL, img=M.IMG, heads=4, layers=4)
        s.head = nn.Linear(128, 4)
    def forward(s, img, c): return s.head(s.enc(img).mean(1))
l, a = train_eval(EncPool().to(dev), lambda n, img, c: n(img, c))
print(f"  (b) enc+meanpool : loss {l:.3f} · acc {a:.1%} {'HỌC' if a > 0.5 else 'KẸT'}")

# (b2) encoder KHÔNG mean-pool: flatten toàn token (giữ vị-trí tuyệt-đối)
class EncFlat(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = VisionEncoder(d_v=128, patch=M.CELL, img=M.IMG, heads=4, layers=4)
        s.head = nn.Linear(128 * M.G * M.G, 4)
    def forward(s, img, c): return s.head(s.enc(img).flatten(1))
l, a = train_eval(EncFlat().to(dev), lambda n, img, c: n(img, c))
print(f"  (b2) enc+flatten : loss {l:.3f} · acc {a:.1%} {'HỌC' if a > 0.5 else 'KẸT'}")

# (c) full net
l, a = train_eval(M.SalienceNet(m=4, cond=True).to(dev), lambda n, img, c: n(img, c))
print(f"  (c) full net     : loss {l:.3f} · acc {a:.1%} {'HỌC' if a > 0.5 else 'KẸT'}")
