"""M2.5 crux — bài 'tìm quadrant màu-c giữa nhiễu' có HỌC nổi bởi ca TỐT-NHẤT không?
(best-case = flatten TOÀN 36 grid token + điều-kiện màu c, KHÔNG bottleneck resampler).
Nếu best-case HỌC → task ổn, resampler là mắt-xích yếu (⇒ warm-start).
Nếu best-case KẸT → task/ngân-sách quá khó ⇒ reformulate PoC."""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m25_salience as M                                           # noqa: E402
from nn_aivie3.vision import VisionEncoder                             # noqa: E402
dev = M.dev_()


class Best(nn.Module):
    """enc(0 tầng self-attn nếu el=0) → flatten 36 token + cvec → head. KHÔNG resampler."""
    def __init__(s, el, d=128):
        super().__init__()
        s.enc = VisionEncoder(d_v=d, patch=M.CELL, img=M.IMG, heads=4, layers=el)
        s.tgt = nn.Embedding(M.N_COLORS, d); nn.init.normal_(s.tgt.weight, std=0.2)
        s.head = nn.Sequential(nn.Linear(d * M.G * M.G + d, d), nn.GELU(), nn.Linear(d, 4))
    def forward(s, img, c):
        g = s.enc(img).flatten(1)
        return s.head(torch.cat([g, s.tgt(c)], -1))


def run(nd, el, steps=500, bs=128, lr=1.5e-3):
    M.N_DIST = nd; torch.manual_seed(0)
    net = Best(el).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1); net.train()
    for it in range(steps):
        img, c, quad = M.render(bs, gen, dev)
        loss = F.cross_entropy(net(img, c), quad)
        opt.zero_grad(); loss.backward(); opt.step()
    g2 = torch.Generator().manual_seed(999); net.eval()
    img, c, quad = M.render(512, g2, dev)
    with torch.no_grad():
        acc = net(img, c).argmax(-1).eq(quad).float().mean().item()
    print(f"  best-case nhiễu={nd} enc={el}: loss {loss.item():.3f} · acc {acc:.1%} "
          f"{'HỌC' if acc > 0.5 else 'KẸT'}")


print(f"=== M2.5 crux · dev={dev} · ngẫu-nhiên 25% ===")
run(10, 2)
run(10, 0)
run(3, 2)
