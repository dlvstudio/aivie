"""nn-aivie3 · CURIOSITY — động-cơ TỰ-THÂN: phần-thưởng nội-sinh = độ-bất-định world-model.

Hiện-thực ghi-chú 2016 của anh ("tò mò → khám phá → thỏa-mãn"): agent THÍCH đi tới chỗ nó
CHƯA đoán được (world-model còn bất-định) → tự khám-phá. So với agent đi NGẪU-NHIÊN:
  • CURIOUS: chọn hành-động mà world-model KÉM TỰ-TIN nhất về ô-kế (đi vào cái LẠ).
  • RANDOM: đi bừa.
Cả hai học world-model online như nhau (replay + SGD); khác duy-nhất là THU-THẬP dữ-liệu.

Đo theo thời-gian: (a) PHỦ bản-đồ (số ô thăm được), (b) world-model ĐÚNG bao nhiêu (học nhanh?).
Kỳ-vọng: tò mò phủ nhanh hơn + world-model giỏi sớm hơn — "động-cơ tự-thân" biến agent từ
'được dạy' → 'TỰ khám-phá'.

CHẠY:  python nn-aivie3/scripts/rmsh_curiosity.py
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmsh_dual as D
dev = "mps" if torch.backends.mps.is_available() else "cpu"
G = 8; GG = G * G
DIRS = D.DIRS


def free_(cell, walls): return 0 <= cell[0] < G and 0 <= cell[1] < G and cell not in walls
def idx(c): return c[0]*G + c[1]
def nxt_(c, a, walls):
    d = DIRS[a]; n = (c[0]+d[0], c[1]+d[1]); return n if free_(n, walls) else c


def gen_maze(gen, wp=0.15):
    while True:
        walls = {(r, c) for r in range(G) for c in range(G) if torch.rand(1, generator=gen).item() < wp}
        frees = [(r, c) for r in range(G) for c in range(G) if (r, c) not in walls]
        if len(frees) > 0.6*GG: return walls, frees


class WM(nn.Module):
    def __init__(s, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(GG+4, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, GG))
    def forward(s, c, a):
        return s.net(torch.cat([F.one_hot(c, GG).float(), F.one_hot(a, 4).float()], -1))


def wm_accuracy(wm, walls, frees):
    with torch.no_grad():
        cs, as_, tg = [], [], []
        for c in frees:
            for a in range(4):
                cs.append(idx(c)); as_.append(a); tg.append(idx(nxt_(c, a, walls)))
        p = wm(torch.tensor(cs, device=dev), torch.tensor(as_, device=dev)).argmax(-1)
        return (p == torch.tensor(tg, device=dev)).float().mean().item()


def explore(curious, walls, frees, start, T, gen):
    torch.manual_seed(0)
    wm = WM().to(dev); opt = torch.optim.AdamW(wm.parameters(), lr=3e-3)
    buf = []; cur = start; visited = {start}; cov, acc = [], []
    for step in range(T):
        ci = idx(cur)
        if curious:                                                   # chọn hành-động world-model KÉM TỰ-TIN nhất
            with torch.no_grad():
                p = wm(torch.full((4,), ci, device=dev), torch.arange(4, device=dev)).softmax(-1)
                unc = 1 - p.max(-1).values                            # bất-định = 1 - tự-tin
            a = int(torch.multinomial((unc/0.1).softmax(0), 1))       # ưu-tiên cái LẠ (có nhiễu)
        else:
            a = int(torch.randint(4, (1,), generator=gen))
        nc = nxt_(cur, a, walls); buf.append((ci, a, idx(nc))); cur = nc; visited.add(cur)
        # học world-model online (giống nhau cho cả 2)
        if len(buf) >= 32:
            for _ in range(2):
                ix = torch.randint(len(buf), (64,))
                c = torch.tensor([buf[i][0] for i in ix], device=dev)
                aa = torch.tensor([buf[i][1] for i in ix], device=dev)
                t = torch.tensor([buf[i][2] for i in ix], device=dev)
                loss = F.cross_entropy(wm(c, aa), t)
                opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 100 == 0:
            cov.append(len(visited)/len(frees)); acc.append(wm_accuracy(wm, walls, frees))
    return cov, acc


def main():
    print(f"=== nn-aivie3 · CURIOSITY (động-cơ tự-thân) · lưới {G}x{G} · dev={dev} ===\n")
    g = torch.Generator().manual_seed(3)
    walls, frees = gen_maze(g); start = frees[0]; T = 1500
    print(f"mê-cung {len(frees)} ô đi-được · {T} bước khám-phá\n")
    t0 = time.time()
    cov_c, acc_c = explore(True, walls, frees, start, T, torch.Generator().manual_seed(1))
    cov_r, acc_r = explore(False, walls, frees, start, T, torch.Generator().manual_seed(1))
    print(f"    (chạy {time.time()-t0:.0f}s)\n")
    print(f"  {'bước':>5} | {'PHỦ bản-đồ':^19} | {'world-model ĐÚNG':^19}")
    print(f"  {'':>5} | {'tò mò':>8} {'ngẫu-nhiên':>9} | {'tò mò':>8} {'ngẫu-nhiên':>9}")
    print("  " + "-"*52)
    for i, s in enumerate(range(100, T+1, 100)):
        print(f"  {s:>5} | {cov_c[i]:>7.0%} {cov_r[i]:>9.0%} | {acc_c[i]:>7.0%} {acc_r[i]:>9.0%}")
    # tóm-tắt: bước để world-model đạt 90%
    def first_ge(acc, th):
        for i, v in enumerate(acc):
            if v >= th: return (i+1)*100
        return None
    tc, tr = first_ge(acc_c, 0.9), first_ge(acc_r, 0.9)
    print(f"\n  ⇒ phủ bản-đồ cuối: tò mò {cov_c[-1]:.0%} vs ngẫu-nhiên {cov_r[-1]:.0%}")
    print(f"    world-model đạt 90%: tò mò @{tc} bước vs ngẫu-nhiên @{tr} bước")
    win = cov_c[-1] > cov_r[-1] + 0.05 or (tc and (not tr or tc < tr))
    print("    " + ("THÀNH ✓ — TÒ MÒ (phần-thưởng=bất-định world-model) khám-phá & học NHANH hơn đi bừa. "
                    "Động-cơ tự-thân: agent TỰ tìm cái mới thay vì chờ được dạy."
                    if win else "chưa tách rõ — chỉnh nhiệt-độ/tốc-độ học."))


if __name__ == "__main__":
    main()
