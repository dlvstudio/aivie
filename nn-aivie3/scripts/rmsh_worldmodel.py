"""nn-aivie3 · WORLD-MODEL (PoC "thủ-môn") — mạng TỰ DỰNG giả-lập để DỰ-ĐOÁN tương-lai.

Bóng bay ngang qua sân theo vận-tốc vx; NHƯNG bị CHE-KHUẤT ở đoạn cuối (không thấy). Muốn
bắt được phải ĐOÁN điểm rơi từ đường bóng đã thấy — "tưởng-tượng" nốt quãng bị che (forward
model / thủ-môn đọc đường bóng). So:
  • PHẢN-ỨNG (reactive): đứng ở vị-trí THẤY CUỐI-CÙNG → bóng còn bay khi bị che → TRƯỢT.
  • DỰ-ĐOÁN (world-model): học vx → ngoại-suy điểm rơi qua đoạn che → BẮT.

Chứng-minh: dự-đoán ≫ phản-ứng, đúng "thủ-môn căn đường bóng". Bài tuyến-tính, tránh tường
spatial-reasoning → CHẠY được ở cỡ đồ-chơi.

CHẠY:  python nn-aivie3/scripts/rmsh_worldmodel.py
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from nn_aivie3.model import HybridBackbone

W, T, VIS = 16, 12, 7                                                  # sân rộng W · bay T bước · thấy VIS bước đầu
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def gen(b, gen_):
    c0 = torch.randint(0, W, (b,), generator=gen_).float()
    vx = torch.randint(-2, 3, (b,), generator=gen_).float()           # vận-tốc ngang -2..2
    cols = []
    cur = c0.clone()
    for _ in range(T):
        cols.append(cur.clamp(0, W-1)); cur = (cur + vx)
    cols = torch.stack(cols, 1)                                       # (b,T) cột bóng mỗi bước
    landing = cols[:, -1]                                             # điểm RƠI (bước cuối)
    last_visible = cols[:, VIS-1]                                     # thấy-cuối (reactive dựa vào đây)
    # đầu-vào chuỗi (b,T,3): [hàng=bước/T, cột/W nếu THẤY, cờ-thấy]
    seq = torch.zeros(b, T, 3)
    for s in range(T):
        seq[:, s, 0] = s / (T-1)
        if s < VIS:
            seq[:, s, 1] = cols[:, s] / (W-1); seq[:, s, 2] = 1.0
    return seq.to(dev), landing.to(dev), last_visible.to(dev)


class WorldModelNet(nn.Module):
    """nhìn đường bóng (có che) → ngoại-suy điểm rơi (1 số)."""
    def __init__(s, d=128, n_layer=5, attn_layers=(1, 3), d_state=16, heads=4, window=32):
        super().__init__()
        s.inp = nn.Linear(3, d); s.ln = nn.LayerNorm(d)
        s.bb = HybridBackbone(d, n_layer, attn_layers, d_state, heads, window)
        s.head = nn.Linear(d, 1)
    def forward(s, seq):
        return s.head(s.bb(s.ln(s.inp(seq)))[:, -1]).squeeze(-1)      # (b,) cột-rơi dự-đoán


def main():
    print(f"=== nn-aivie3 · world-model 'thủ-môn' · dev={dev} · sân {W} · che {VIS}..{T} ===\n")
    torch.manual_seed(0)
    m = WorldModelNet().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3); g = torch.Generator().manual_seed(1)
    m.train(); t0 = time.time()
    for it in range(500):
        seq, land, _ = gen(96, g)
        loss = F.mse_loss(m(seq), land)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == 499:
            print(f"    luyện it {it:>3} MSE {loss.item():.3f} · {time.time()-t0:.0f}s", flush=True)

    m.eval(); g2 = torch.Generator().manual_seed(999); N = 2000
    seq, land, last_vis = gen(N, g2)
    with torch.no_grad():
        pred = m(seq)
    err_pred = (pred - land).abs().mean().item()                     # sai-số dự-đoán (ô)
    err_react = (last_vis - land).abs().mean().item()                # sai-số phản-ứng (thấy-cuối)
    catch_pred = ((pred.round() - land).abs() <= 1).float().mean().item()
    catch_react = ((last_vis - land).abs() <= 1).float().mean().item()

    print(f"\n  sai-số điểm rơi (ô, càng nhỏ càng giỏi):")
    print(f"    PHẢN-ỨNG (đứng chỗ thấy-cuối)     : {err_react:.2f} ô")
    print(f"    DỰ-ĐOÁN  (world-model ngoại-suy)  : {err_pred:.2f} ô")
    print(f"\n  tỉ-lệ BẮT được (|lệch|≤1 ô):")
    print(f"    PHẢN-ỨNG : {catch_react:.0%}")
    print(f"    DỰ-ĐOÁN  : {catch_pred:.0%}")
    ok = catch_pred > catch_react + 0.2
    print(f"\n  ⇒ " + ("THÀNH ✓ — mạng TỰ DỰNG giả-lập đường bóng, đoán điểm rơi XUYÊN che-khuất → "
                       "bắt hơn hẳn phản-ứng. Đúng 'thủ-môn căn đường bóng'."
                       if ok else "chưa tách rõ — chỉnh vx/che/steps."))
    print("    Đây là world-model: hành-động theo TƯƠNG-LAI dự-đoán, không theo hiện-tại. Nối:")
    print("    (1) tưởng-tượng nhiều bước = kế-hoạch Hệ-2 (vá dual-process); (2) sai-số dự-báo = tín-hiệu bất-ngờ/chú-ý.")


if __name__ == "__main__":
    main()
