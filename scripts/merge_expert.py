"""Ghép expert: lấy checkpoint --base, thay các expert --names bằng bản trong --donor (cùng thân).

Dùng khi một vòng học làm tụt một expert (vd vòng thầy P3.3' kéo expert văn phòng: Excel 3,44 → 2,31).
  python scripts/merge_expert.py --base checkpoints/hf_hybrid/aivie_2p6b_p3_r1_bf16.pt \
      --donor checkpoints/hf_hybrid/aivie_2p6b_p3_text_bf16.pt --names office \
      --out checkpoints/hf_hybrid/aivie_2p6b_p3_r1vp_bf16.pt
"""
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--donor", required=True)
ap.add_argument("--names", default="office", help="tên expert cần thay, cách nhau dấu phẩy")
ap.add_argument("--out", required=True)
a = ap.parse_args()

base = torch.load(a.base, map_location="cpu", weights_only=False)
donor = torch.load(a.donor, map_location="cpu", mmap=True, weights_only=False)
for k in ("d", "layers", "attn_every"):                    # phải cùng thân
    assert base["cfg"].get(k) == donor["cfg"].get(k), f"khác thân: {k}"
for n in a.names.split(","):
    assert n in donor["experts"], f"donor không có expert {n}"
    old = base["experts"].get(n)
    base["experts"][n] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in donor["experts"][n].items()}
    base["experts"][n]["state"] = {kk: vv.clone() for kk, vv in donor["experts"][n]["state"].items()}
    print(f"[merge] {n}: {'thay' if old else 'thêm'} từ {a.donor}")
base.setdefault("merged_from", []).append({"donor": a.donor, "names": a.names})
torch.save(base, a.out)
print(f"[merge] → {a.out} · expert {list(base['experts'])}")
