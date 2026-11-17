"""M2.5 smear-test — self-attn encoder có 'bôi' nhận-dạng patch làm resampler mù màu?
Quét enc_layers ∈ {0,1,2,4}, m=4, topdown vs uniform, nhiễu=10."""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m25_salience as M                                           # noqa: E402
M.N_DIST = 10
dev = M.dev_(); print(f"=== M2.5 smear · dev={dev} · nhiễu=10 · m=4 · ngẫu-nhiên 25% ===")
print(f"{'enc_layers':>10} | {'uniform':>8} | {'topdown':>8} | lợi")
print("-" * 44)


def run(enc_layers, cond, steps=500, bs=128, lr=1.5e-3):
    torch.manual_seed(0)
    net = M.SalienceNet(m=4, cond=cond, enc_layers=enc_layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr); gen = torch.Generator().manual_seed(1); net.train()
    for it in range(steps):
        img, c, quad = M.render(bs, gen, dev)
        loss = F.cross_entropy(net(img, c), quad)
        opt.zero_grad(); loss.backward(); opt.step()
    g2 = torch.Generator().manual_seed(999); net.eval()
    img, c, quad = M.render(512, g2, dev)
    with torch.no_grad():
        return net(img, c).argmax(-1).eq(quad).float().mean().item()


for el in (0, 1, 2, 4):
    au = run(el, False); at = run(el, True)
    print(f"{el:>10} | {au:>7.1%} | {at:>7.1%} | {at-au:+.1%}")
