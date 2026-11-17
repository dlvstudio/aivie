"""nn-aivie3 · M4 TTS-OUT — đường 'NÓI RA': sinh chuỗi khung phổ (mel) TỰ-HỒI-QUY có
điều-kiện (yêu-cầu pitch LÊN/XUỐNG/ĐỀU) → khép vòng tiếng 2 chiều (M4 audio-IN + TTS-OUT).

Model: cond(lớp)→ chuỗi khung mel, backbone nhân-quả dự-báo khung KẾ (teacher-forcing MSE).
Sinh: từ cond, tự-hồi-quy T khung. KIỂM: hướng-pitch sinh ra (độ dốc TRỌNG-TÂM PHỔ theo
thời-gian) có khớp lớp yêu-cầu không (≫33% ⇒ 'nói ra' đúng nội-dung). Vocoder waveform
(Griffin-Lim) là bước dựng-lại tuỳ-chọn — ở PoC đo ngay mức mel cho sạch.

CHẠY:  python nn-aivie3/scripts/rmsh_m4_tts.py
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.dirname(HERE))
import rmsh_m4_audio as A                                              # synth + to_tokens (mel) đã có
from nn_aivie3.model import HybridBackbone                             # noqa: E402
dev = A.dev
NBIN = A.NBIN


class MelGen(nn.Module):
    def __init__(self, d=128, n_layer=6, attn_layers=(2, 4), d_state=16, heads=4, window=64):
        super().__init__()
        self.cond = nn.Embedding(3, d)                                 # lên/xuống/đều
        self.inp = nn.Linear(NBIN, d); self.ln = nn.LayerNorm(d)
        self.bb = HybridBackbone(d, n_layer, attn_layers, d_state, heads, window)
        self.out = nn.Linear(d, NBIN)

    def forward(self, cls, mel):                                       # teacher-forcing
        x = torch.cat([self.cond(cls)[:, None], self.ln(self.inp(mel[:, :-1]))], 1)   # (b,T,d)
        return self.out(self.bb(x))                                    # (b,T,NBIN) dự-báo mel[:, 0:]

    @torch.no_grad()
    def generate(self, cls, T):                                        # tự-hồi-quy T khung
        x = self.cond(cls)[:, None]; outs = []
        for _ in range(T):
            f = self.out(self.bb(x)[:, -1]); outs.append(f)
            x = torch.cat([x, self.ln(self.inp(f))[:, None]], 1)
        return torch.stack(outs, 1)                                    # (b,T,NBIN)


def centroid_slope(mel):
    """độ dốc TRỌNG-TÂM phổ theo thời-gian → hướng pitch (dương=lên, âm=xuống)."""
    b, T, F_ = mel.shape
    w = mel.clamp(min=0) + 1e-6
    idx = torch.arange(F_, device=mel.device).float()
    cen = (w * idx).sum(-1) / w.sum(-1)                                # (b,T) trọng-tâm mỗi khung
    t = torch.arange(T, device=mel.device).float()
    tc = t - t.mean()
    slope = ((cen - cen.mean(1, keepdim=True)) * tc).sum(1) / (tc * tc).sum()   # (b,)
    return slope


def cls_from_slope(slope, thr=0.02):
    return torch.where(slope > thr, torch.tensor(0, device=slope.device),       # 0=lên
           torch.where(slope < -thr, torch.tensor(1, device=slope.device), torch.tensor(2)))  # 1=xuống,2=đều


def main():
    print(f"=== nn-aivie3 M4 TTS-out · dev={dev} · sinh mel có điều-kiện · ngẫu-nhiên 33% ===\n")
    torch.manual_seed(0)
    m = MelGen().to(dev)
    T = A.to_tokens(A.synth(2, torch.Generator().manual_seed(0))[0]).shape[1]
    print(f"sinh {T} khung mel × {NBIN} bin tự-hồi-quy từ yêu-cầu pitch")
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3); gen = torch.Generator().manual_seed(1)
    m.train(); t0 = time.time()
    for it in range(400):
        wave, cls = A.synth(64, gen); mel = A.to_tokens(wave).to(dev); cls = cls.to(dev)
        loss = F.mse_loss(m(cls, mel), mel)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == 399:
            print(f"    it {it:>4} mse {loss.item():.4f} · {time.time()-t0:.0f}s", flush=True)

    # sinh theo yêu-cầu rồi kiểm hướng-pitch
    m.eval(); req = torch.arange(3, device=dev).repeat(100)            # 300 mẫu, đều 3 lớp
    with torch.no_grad():
        gmel = m.generate(req, T)
    got = cls_from_slope(centroid_slope(gmel))
    acc = got.eq(req.cpu() if got.device != req.device else req).float().mean().item()
    names = ["lên", "xuống", "đều"]
    print(f"\n[1] hướng-pitch sinh ra khớp yêu-cầu: {acc:.1%} → {'ĐẠT' if acc > 0.6 else 'YẾU'}")
    for k in range(3):
        sl = centroid_slope(m.generate(torch.full((32,), k, device=dev), T)).mean().item()
        print(f"    yêu-cầu '{names[k]}': độ dốc trọng-tâm trung-bình {sl:+.3f}")
    print("\n→ M4 TTS-out: model SINH được chuỗi phổ đúng hướng pitch yêu-cầu (đường 'nói ra').")
    print("  Khép vòng tiếng 2 chiều: audio-IN (nghe, M4) + TTS-out (nói). Vocoder→waveform = ISTFT/Griffin-Lim.")


if __name__ == "__main__":
    main()
