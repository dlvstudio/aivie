"""M2.5 diag — định-vị có HỌC nổi không? Thử dễ dần: nhiễu 0/3/10, nhiều bước hơn.
Nếu nhiễu=0 vẫn kẹt → lỗi wiring; nếu nhiễu=0 học được → bài khó/cần bước."""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m25_salience as M                                           # noqa: E402

dev = M.dev_()
print(f"=== M2.5 diag · dev={dev} · ngẫu-nhiên 25% ===")


def run(n_dist, cond, steps, dev, bs=128, lr=2e-3):
    M.N_DIST = n_dist
    torch.manual_seed(0)
    net = M.SalienceNet(m=4, cond=cond).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr); gen = torch.Generator().manual_seed(1)
    net.train()
    for it in range(steps):
        img, c, quad = M.render(bs, gen, dev)
        loss = F.cross_entropy(net(img, c), quad)
        opt.zero_grad(); loss.backward(); opt.step()
    g2 = torch.Generator().manual_seed(999); net.eval()
    img, c, quad = M.render(512, g2, dev)
    with torch.no_grad():
        acc = net(img, c).argmax(-1).eq(quad).float().mean().item()
    print(f"  nhiễu={n_dist:>2} cond={cond} steps={steps}: loss {loss.item():.3f} · acc {acc:.1%} "
          f"{'HỌC' if acc > 0.5 else 'kẹt'}")
    return acc


run(0, True, 400, dev)
run(0, False, 400, dev)
run(3, True, 600, dev)
run(10, True, 800, dev)
