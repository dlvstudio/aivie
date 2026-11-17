"""nn-aivie3 · M3.5 SMOKE — GIẢI đánh-đổi token↔vị-trí bằng gate salience trong runtime.

Hợp-nhất M2.5 (chú-ý học-được) vào M3 (vòng realtime): train dense-gated (học được, giữ
vị-trí) rồi SERVE chỉ giữ top-k patch salient/frame. Nếu quyết-định vẫn ĐÚNG với k NHỎ ⇒
"ít token, đúng token" — vừa realtime (rẻ) vừa giữ vị-trí (học được). Bài: đoán hướng
chuyển-động (cần bộ-nhớ + định-vị chấm).

CHẠY:  python nn-aivie3/scripts/rmsh_m35_smoke.py
"""
import os, sys, time
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nn_aivie3.agent import SalientActionModel                          # noqa: E402

G, CELL, IMG, T = 4, 8, 32, 4
DIRS = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def make(b, gen):
    d = torch.randint(0, 4, (b,), generator=gen); vel = DIRS[d]
    r = torch.full((b,), G // 2).float() - (vel[:, 0] > 0).float()
    c = torch.full((b,), G // 2).float() - (vel[:, 1] > 0).float()
    fr = torch.zeros(b, T, 3, IMG, IMG)
    for t in range(T):
        rr = (r + vel[:, 0] * t).clamp(0, G - 1).long(); cc = (c + vel[:, 1] * t).clamp(0, G - 1).long()
        for i in range(b):
            fr[i, t, :, rr[i] * CELL:rr[i] * CELL + CELL, cc[i] * CELL:cc[i] * CELL + CELL] = 1.0
    fr = fr + 0.03 * torch.randn(b, T, 3, IMG, IMG, generator=gen)
    return fr.to(dev), d.to(dev)


def main():
    print(f"=== nn-aivie3 M3.5 smoke · dev={dev} · lưới {G}x{G} ({(IMG//CELL)**2} patch/frame) · ngẫu-nhiên 25% ===\n")
    torch.manual_seed(0)
    m = SalientActionModel(n_actions=4, d=128, img=IMG, patch=CELL, window=128,
                           gate_mode="softmax").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3); gen = torch.Generator().manual_seed(1)
    m.train(); t0 = time.time(); STEPS, KT = 350, 3
    print(f"  train k-anneal {m.np}→{KT} (ép dồn dần vào patch salient):")
    for it in range(STEPS):
        k = max(KT, round(m.np - (m.np - KT) * it / (STEPS * 0.7)))   # 16→3 trong 70% đầu
        fr, d = make(48, gen)
        logit, g = m(fr, topk=k)
        loss = F.cross_entropy(logit[:, 1:].reshape(-1, 4), d[:, None].expand(-1, T - 1).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == STEPS - 1:
            print(f"    it {it:>4} k={k:>2} loss {loss.item():.3f} · {time.time()-t0:.0f}s", flush=True)

    # eval: giữ top-k patch salient/frame (k giảm dần)
    m.eval(); g2 = torch.Generator().manual_seed(999); fr, d = make(256, g2)
    print(f"\n  giữ top-k patch salient/frame (trên {m.np} patch) — acc hướng ở frame cuối:")
    for k in (m.np, 8, 4, 2, 1):
        with torch.no_grad():
            logit, g = m(fr, topk=k)
            acc = logit[:, -1].argmax(-1).eq(d).float().mean().item()
        tag = "full" if k == m.np else f"top-{k}"
        print(f"    {tag:>6} ({k:>2} token/frame): acc {acc:.1%}  {'✓' if acc > 0.6 else ''}")
    print("\n→ M3.5: nếu top-2/4 giữ acc cao ⇒ gate salience cho phép ÍT token mà GIỮ vị-trí")
    print("  ⇒ realtime (rẻ) + học-được — giải đánh-đổi M2↔M3. Kế: nối kênh chữ/đối-thoại.")


if __name__ == "__main__":
    main()
