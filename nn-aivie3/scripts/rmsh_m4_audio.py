"""nn-aivie3 · M4 — KÊNH TIẾNG (audio-IN): âm-thanh cắm vào CÙNG runtime streaming.

Pipeline tiếng THẬT: waveform → STFT (log-magnitude, pool freq) → token/khung-thời-gian →
HybridBackbone StreamState (giống hình ở M1/M3) → quyết-định. Bài: chirp LÊN/XUỐNG/ĐỀU
(tần-số biến theo thời-gian) — buộc nghe phổ THEO THỜI-GIAN (bộ-nhớ streaming).

Chứng-minh: (1) audio-in HỌC được (acc≫33%) ⇒ tiếng là 'một luồng token' như hình;
(2) PARITY forward(train)==streaming(serve). TTS-out (sinh tiếng) = thiết-kế ở doc §2.3.

CHẠY:  python nn-aivie3/scripts/rmsh_m4_audio.py
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nn_aivie3.model import HybridBackbone                              # noqa: E402

SR, DUR = 8000, 0.5
N = int(SR * DUR)
NFFT, HOP, NBIN = 256, 128, 32
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def synth(b, gen):
    """b waveform chirp: lớp 0=lên, 1=xuống, 2=đều. Trả (b,N) + nhãn."""
    cls = torch.randint(0, 3, (b,), generator=gen)
    t = torch.arange(N).float() / SR
    f0 = torch.where(cls == 1, 800.0, torch.where(cls == 0, 200.0, 500.0))
    f1 = torch.where(cls == 1, 200.0, torch.where(cls == 0, 800.0, 500.0))
    fmid = (f0[:, None] + (f1 - f0)[:, None] * (t[None] / DUR))          # tần-số tức-thời (b,N)
    phase = 2 * torch.pi * torch.cumsum(fmid, 1) / SR
    wave = torch.sin(phase) + 0.02 * torch.randn(b, N, generator=gen)
    return wave, cls


def to_tokens(wave):
    """waveform → (b, T_khung, NBIN) log-magnitude STFT pool freq (đặc-trưng phổ theo thời-gian)."""
    win = torch.hann_window(NFFT)
    spec = torch.stft(wave, NFFT, HOP, window=win, return_complex=True)  # (b, F=129, T)
    mag = spec.abs().transpose(1, 2)                                     # (b, T, 129)
    mag = mag[..., :128].reshape(mag.shape[0], mag.shape[1], NBIN, 4).mean(-1)  # pool 128→32
    return torch.log1p(mag)                                              # (b, T, 32)


class AudioStreamModel(nn.Module):
    def __init__(self, n_cls=3, d=128, n_layer=6, attn_layers=(2, 4), d_state=16, heads=4, window=64):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(NBIN, d)                                   # mel-bin → token
        self.ln = nn.LayerNorm(d)
        self.backbone = HybridBackbone(d, n_layer, attn_layers, d_state, heads, window)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_cls))

    def forward(self, feats):                                           # (b,T,NBIN) → (b,n_cls)
        x = self.ln(self.inp(feats))
        return self.head(self.backbone(x)[:, -1])                       # đọc sau khung cuối

    @torch.no_grad()
    def stream(self, feats):                                            # serve: từng khung qua StreamState
        st = self.backbone.init_stream(feats.shape[0], feats.device)
        x = self.ln(self.inp(feats)); h = None
        for t in range(feats.shape[1]):
            h = self.backbone.stream_step(x[:, t], st)
        return self.head(h)


def main():
    print(f"=== nn-aivie3 M4 audio-in · dev={dev} · chirp lên/xuống/đều · ngẫu-nhiên 33% ===\n")
    torch.manual_seed(0)
    m = AudioStreamModel().to(dev)
    T = to_tokens(synth(2, torch.Generator().manual_seed(0))[0]).shape[1]
    print(f"pipeline: waveform {N} mẫu → STFT → {T} khung × {NBIN} bin → token → StreamState")
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3); gen = torch.Generator().manual_seed(1)
    m.train(); t0 = time.time()
    for it in range(300):
        wave, cls = synth(64, gen); feats = to_tokens(wave).to(dev); cls = cls.to(dev)
        loss = F.cross_entropy(m(feats), cls)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == 299:
            print(f"    it {it:>4} loss {loss.item():.3f} · {time.time()-t0:.0f}s", flush=True)

    m.eval(); g2 = torch.Generator().manual_seed(999)
    wave, cls = synth(300, g2); feats = to_tokens(wave).to(dev); cls = cls.to(dev)
    with torch.no_grad():
        tr = m(feats); acc = tr.argmax(-1).eq(cls).float().mean().item()
        sv = m.stream(feats[:4])
    err = (tr[:4] - sv).abs().max().item()
    print(f"\n[1] acc chirp (held-out): {acc:.1%} → {'ĐẠT' if acc > 0.6 else 'YẾU'} ⇒ audio-in {'HỌC ĐƯỢC' if acc>0.6 else 'chưa'}")
    print(f"[2] parity forward==stream: max|Δ|={err:.2e} → {'ĐẠT' if err < 1e-3 else 'SAI'}")
    print("\n→ M4: TIẾNG là 'một luồng token' cắm vào cùng StreamState như hình (audio-IN).")
    print("  TTS-out (sinh token tiếng streaming) = thiết-kế §2.3; nối ASR→chữ hoặc token-tiếng bản-địa.")


if __name__ == "__main__":
    main()
