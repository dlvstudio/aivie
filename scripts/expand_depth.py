"""P0 (nuôi lớn thân 1.3B → ~2,6B) · CHÈN KHỐI ĐỒNG NHẤT (kiểu LLaMA-Pro / SOLAR): 48 → 96 khối.

    khối cũ i  → vị trí mới 2i (giữ nguyên trọng số)
    khối mới   → vị trí 2i+1, chép trọng số từ khối cũ CÙNG LOẠI gần nhất, rồi ĐẶT LỚP RA = 0
                 (Mamba: out_proj = 0 · attention: wo = 0) → x + khối(x) = x → lúc đầu model Y HỆT bản cũ.
    loại khối mới: attention nếu i ≡ 3 (mod 8) (chèn GIỮA 2 attention cũ) còn lại Mamba
    → 96 khối: 84 Mamba + 12 attention, attention cách nhau 7/9 khối (~mỗi 8 khối như cũ).

Checkpoint ra: cfg có layers=96, attn_layers (danh sách), rope_base, new_blocks (để train "chỉ khối mới"),
expanded_from. KHÔNG mang "opt" (optimizer cũ không khớp số tham số).

    python scripts/expand_depth.py --src <ckpt 1.3B> --out <ckpt 2.6B> [--check]   # --check: so logit cũ/mới
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import torch


def plan(old_layers, old_attn, every=8):
    """→ (danh sách (nguồn_cũ, là_mới) cho 2·old_layers vị trí, attn mới, chỉ số khối mới)."""
    olds = set(old_attn)
    layout, new_attn, new_blocks = [], [], []
    for i in range(old_layers):
        layout.append((i, False))
        if i in olds:
            new_attn.append(2 * i)
        k = 2 * i + 1
        want_attn = (i % every) == (every // 2 - 1)                     # i ≡ 3 (mod 8)
        if want_attn:
            src = min(olds, key=lambda j: (abs(j - i), j))              # attention cũ gần nhất
            new_attn.append(k)
        else:
            src = i if i not in olds else i - 1                         # Mamba cũ ngay trước
        layout.append((src, True))
        new_blocks.append(k)
    return layout, sorted(new_attn), new_blocks


def expand(obj, every=8, dtype=None):
    import train_hybrid_scale as ths
    c = dict(obj["cfg"])
    st = obj["state"]
    L = c["layers"]
    old_attn = ths.attn_layers_of(c)
    layout, new_attn, new_blocks = plan(L, old_attn, every)
    out = {k: v for k, v in st.items() if not k.startswith(("blocks.", "norms."))}
    for k_new, (src, is_new) in enumerate(layout):
        for pre in ("blocks", "norms"):
            p_old = f"{pre}.{src}."
            for k, v in st.items():
                if k.startswith(p_old):
                    nk = f"{pre}.{k_new}." + k[len(p_old):]
                    v = v.clone() if dtype is None else v.to(dtype)
                    if is_new and pre == "blocks" and k[len(p_old):] in ("out_proj.weight", "out_proj.bias", "wo.weight"):
                        v.zero_()                                       # lớp ra = 0 → khối đồng nhất
                    out[nk] = v
    c2 = {**c, "layers": 2 * L, "attn_layers": new_attn, "attn_every": 0, "rope_base": c.get("rope_base", 10000.0),
          "new_blocks": new_blocks, "expanded_from": f"{L} khối · attn {old_attn}"}
    for k in ("tokens", "steps", "init_from", "out", "hf_repo", "hf_name"):         # tham số train cũ → bỏ
        c2.pop(k, None)
    return {"cfg": c2, "state": out, "step": 0, "note": f"nuôi lớn từ {L} khối (P0): khối mới lớp ra = 0"}


@torch.no_grad()
def logits(obj, ids, dtype):
    import train_hybrid_scale as ths
    ths.HAVE_KERNEL = False
    with torch.device("meta"):                                         # không cấp phát fp32 → gắn thẳng trọng số
        m = ths.build_from_cfg(obj["cfg"]).eval()
    m.load_state_dict({k: v.to(dtype) for k, v in obj["state"].items()}, assign=True)
    lo = m(ids).float()
    n = sum(p.numel() for p in m.parameters())
    del m
    return lo, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--check", action="store_true", help="so logit bản cũ/mới trên 48 token ngẫu nhiên (CPU, bf16)")
    ap.add_argument("--bf16", action="store_true", help="chép trọng số ở bf16 (kiểm trên máy ít RAM; bản train dùng fp32)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    t = time.time()
    obj = torch.load(a.src, map_location="cpu", weights_only=False, mmap=True)
    obj = {"cfg": obj["cfg"], "state": {k: v for k, v in obj["state"].items()}}
    new = expand(obj, a.every, torch.bfloat16 if a.bf16 else None)
    c = new["cfg"]
    print(f"[expand] {obj['cfg']['layers']} → {c['layers']} khối · attention {c['attn_layers']} · "
          f"{len(c['new_blocks'])} khối mới · {time.time()-t:.0f}s", flush=True)
    if a.check:
        torch.manual_seed(0)
        ids = torch.randint(5, 32000, (1, 48))
        lo_old, n_old = logits(obj, ids, torch.bfloat16)
        lo_new, n_new = logits(new, ids, torch.bfloat16)
        d = (lo_old - lo_new).abs().max().item()
        print(f"[check] tham số thân {n_old/1e9:.3f}B → {n_new/1e9:.3f}B · sai lệch logit lớn nhất {d:.2e} · "
              f"argmax khớp {int((lo_old.argmax(-1) == lo_new.argmax(-1)).sum())}/48", flush=True)
    if not a.no_save:
        torch.save(new, a.out)
        print(f"### {a.out} · {os.path.getsize(a.out)/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
