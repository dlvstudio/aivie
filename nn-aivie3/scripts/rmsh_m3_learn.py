"""M3 learn (nhanh) — token GIỮ-VỊ-TRÍ (grid) có làm vòng quyết-định HỌC bộ-nhớ
chuyển-động không? Lưới 4×4 (16 token/frame), T=4. So với resampler (đã kẹt)."""
import os, sys, time
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nn_aivie3.agent import StreamingActionModel                        # noqa: E402

G, CELL, IMG, T = 4, 8, 32, 4
DIRS = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def make(b, gen):
    d = torch.randint(0, 4, (b,), generator=gen); vel = DIRS[d]
    r = torch.full((b,), G // 2).float() - (vel[:, 0] > 0).float()      # đặt để đi trong khung
    c = torch.full((b,), G // 2).float() - (vel[:, 1] > 0).float()
    fr = torch.zeros(b, T, 3, IMG, IMG)
    for t in range(T):
        rr = (r + vel[:, 0] * t).clamp(0, G - 1).long(); cc = (c + vel[:, 1] * t).clamp(0, G - 1).long()
        for i in range(b):
            fr[i, t, :, rr[i] * CELL:rr[i] * CELL + CELL, cc[i] * CELL:cc[i] * CELL + CELL] = 1.0
    fr = fr + 0.03 * torch.randn(b, T, 3, IMG, IMG, generator=gen)
    return fr.to(dev), d.to(dev)


def run(tokens, steps=250, bs=48, lr=1.5e-3):
    torch.manual_seed(0)
    m = StreamingActionModel(n_actions=4, d=128, n_layer=6, attn_layers=(2, 4), d_state=16,
                             heads=4, window=128, n_query=4, img=IMG, patch=CELL, tokens=tokens).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr); gen = torch.Generator().manual_seed(1)
    m.train(); t0 = time.time()
    for it in range(steps):
        fr, d = make(bs, gen)
        logit = m(fr)[:, 1:]
        loss = F.cross_entropy(logit.reshape(-1, 4), d[:, None].expand(-1, T - 1).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); g2 = torch.Generator().manual_seed(999)
    fr, d = make(256, g2)
    with torch.no_grad():
        acc = m(fr)[:, -1].argmax(-1).eq(d).float().mean().item()
    print(f"  tokens={tokens:9s} (k={m.k}/frame): acc hướng {acc:.1%} · {time.time()-t0:.0f}s "
          f"{'HỌC' if acc > 0.6 else 'kẹt'}")
    return acc


print(f"=== M3 learn · dev={dev} · lưới {G}x{G} · T={T} · ngẫu-nhiên 25% ===")
run("grid")
run("resampler")
