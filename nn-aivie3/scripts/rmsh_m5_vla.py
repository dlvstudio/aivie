"""nn-aivie3 · M5 — ROBOT VLA (vision→action) VÒNG KÍN: qua 'mắt' hiểu thế-giới → hành-động.

Agent (đỏ) + đích (lục) trên lưới. Chính-sách reactive: nhìn khung → xuất HƯỚNG đi về đích.
Train imitation (nhãn = hướng trội tới đích). Rồi CHẠY VÒNG KÍN qua runtime streaming
(perceive O(1) → decide → di-chuyển → lặp), đo tỉ-lệ CHẠM ĐÍCH — đúng 'gắn robot, mắt=cam'.

CHẠY:  python nn-aivie3/scripts/rmsh_m5_vla.py
"""
import os, sys, time
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nn_aivie3.agent import StreamingActionModel                        # noqa: E402

G, CELL, IMG = 4, 8, 32
DIRS = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])                 # lên/xuống/trái/phải
RED, GRN = torch.tensor([1., 0, 0]), torch.tensor([0, 1., 0])
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def draw(ar, ac, tr, tc, b):
    """(b,) toạ-độ agent/đích → khung (b,3,IMG,IMG): agent đỏ, đích lục."""
    fr = torch.zeros(b, 3, IMG, IMG)
    for i in range(b):
        fr[i, :, tr[i] * CELL:tr[i] * CELL + CELL, tc[i] * CELL:tc[i] * CELL + CELL] = GRN[:, None, None]
        fr[i, :, ar[i] * CELL:ar[i] * CELL + CELL, ac[i] * CELL:ac[i] * CELL + CELL] = RED[:, None, None]
    return fr


def expert_dir(ar, ac, tr, tc):
    """hướng trội tới đích (nhãn imitation)."""
    dr, dc = (tr - ar).float(), (tc - ac).float()
    vert = dr.abs() >= dc.abs()
    return torch.where(vert, torch.where(dr < 0, 0, 1), torch.where(dc < 0, 2, 3))


def rand_pos(b, gen):
    p = torch.stack([torch.randint(0, G, (b,), generator=gen) for _ in range(4)], 0)
    return p[0], p[1], p[2], p[3]                                       # ar,ac,tr,tc


def main():
    print(f"=== nn-aivie3 M5 · robot VLA vòng-kín · dev={dev} · lưới {G}x{G} ===\n")
    torch.manual_seed(0)
    m = StreamingActionModel(n_actions=4, d=128, img=IMG, patch=CELL, window=64, tokens="grid").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3); gen = torch.Generator().manual_seed(1)

    # train imitation (khung đơn T=1: nhìn → hướng tới đích)
    m.train(); t0 = time.time()
    for it in range(300):
        ar, ac, tr, tc = rand_pos(64, gen)
        same = (ar == tr) & (ac == tc)                                 # bỏ ca trùng ô
        ar = torch.where(same, (ar + 1) % G, ar)
        fr = draw(ar, ac, tr, tc, 64)[:, None].to(dev)                 # (64,1,3,H,W)
        lbl = expert_dir(ar, ac, tr, tc).to(dev)
        logit = m(fr)[:, 0]                                            # (b,A) hành-động khung hiện-tại
        loss = F.cross_entropy(logit, lbl)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == 299:
            print(f"    it {it:>4} loss {loss.item():.3f} · {time.time()-t0:.0f}s", flush=True)

    # [1] imitation acc
    m.eval(); g2 = torch.Generator().manual_seed(999)
    ar, ac, tr, tc = rand_pos(256, g2)
    same = (ar == tr) & (ac == tc); ar = torch.where(same, (ar + 1) % G, ar)
    with torch.no_grad():
        fr = draw(ar, ac, tr, tc, 256)[:, None].to(dev)
        acc = m(fr)[:, 0].argmax(-1).eq(expert_dir(ar, ac, tr, tc).to(dev)).float().mean().item()
    print(f"\n[1] imitation acc (nhìn→hướng đúng): {acc:.1%} → {'ĐẠT' if acc > 0.6 else 'YẾU'}")

    # [2] VÒNG KÍN: rollout qua perceive/decide, đo chạm đích trong K bước
    E, K = 128, 10
    g3 = torch.Generator().manual_seed(7); ar, ac, tr, tc = rand_pos(E, g3)
    same = (ar == tr) & (ac == tc); ar = torch.where(same, (ar + 1) % G, ar)
    reached = torch.zeros(E, dtype=torch.bool); steps = torch.full((E,), K)
    with torch.no_grad():
        st = m.new_stream(dev, batch=E)
        for k in range(K):
            fr = draw(ar, ac, tr, tc, E).to(dev)
            h = m.perceive(fr, st); a = m.decide(h).cpu()             # perceive O(1) → hành-động
            mv = DIRS[a]
            hit = (ar == tr) & (ac == tc)
            newly = hit & ~reached
            steps = torch.where(newly, torch.full_like(steps, k), steps)
            reached = reached | hit
            ar = (ar + torch.where(hit, torch.zeros_like(mv[:, 0]), mv[:, 0])).clamp(0, G - 1)
            ac = (ac + torch.where(hit, torch.zeros_like(mv[:, 1]), mv[:, 1])).clamp(0, G - 1)
        reached = reached | ((ar == tr) & (ac == tc))
    rate = reached.float().mean().item()
    avg = steps[reached].float().mean().item() if reached.any() else float('nan')
    print(f"[2] VÒNG KÍN chạm đích trong {K} bước: {rate:.1%} (trung-bình {avg:.1f} bước) "
          f"→ {'ĐẠT' if rate > 0.7 else 'YẾU'}")
    print("\n→ M5: 'mắt→hành-động' vòng-kín qua runtime streaming O(1) — nền cho robot (gắn cam=mắt).")
    print("  Nâng-cấp: action liên-tục (khớp/bánh), RL theo phần-thưởng, sim thật (§2.5/§9 L3).")


if __name__ == "__main__":
    main()
