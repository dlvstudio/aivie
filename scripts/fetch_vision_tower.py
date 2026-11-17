"""VLM V0 · tải RIÊNG tháp ảnh (vision_model.*) của SigLIP2 so400m — không tải cả file 4,5GB (gồm tháp chữ không dùng).

safetensors = 8 byte độ dài + JSON đầu (tên → dtype, shape, vị trí byte) + dữ liệu thô → đọc đầu file, rồi tải từng đoạn
byte của các tensor vision_model.* bằng HTTP Range, đổi sang bf16, ghi ra một file ~0,8GB.

    .venv/bin/python scripts/fetch_vision_tower.py            # → checkpoints/hf_hybrid/vision/siglip2_so400m_384_bf16.safetensors
"""
from __future__ import annotations
import argparse, json, os, struct, sys, urllib.request
import numpy as np
import torch
from safetensors.torch import save_file

DT = {"F32": np.float32, "F16": np.float16, "BF16": None}


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--out", default="checkpoints/hf_hybrid/vision/siglip2_so400m_384_bf16.safetensors")
    a = ap.parse_args()
    base = f"https://huggingface.co/{a.repo}/resolve/main/"
    url = base + "model.safetensors"
    n = struct.unpack("<Q", rng(url, 0, 8))[0]
    head = json.loads(rng(url, 8, 8 + n))
    off0 = 8 + n
    keys = sorted(k for k in head if k.startswith("vision_model."))
    lo = min(head[k]["data_offsets"][0] for k in keys); hi = max(head[k]["data_offsets"][1] for k in keys)
    print(f"{len(keys)} tensor tháp ảnh · {(hi - lo) / 1e9:.2f}GB (cả file {sum(v['data_offsets'][1] - v['data_offsets'][0] for k, v in head.items() if k != '__metadata__') / 1e9:.2f}GB)", flush=True)
    blob = bytearray()                                    # các tensor vision_model.* nằm liền một dải [lo, hi)
    CH = 64 << 20
    while lo + len(blob) < hi:
        blob += rng(url, off0 + lo + len(blob), off0 + min(hi, lo + len(blob) + CH))
        sys.stdout.write(f"\rtải {len(blob) / 1e9:.2f}/{(hi - lo) / 1e9:.2f}GB"); sys.stdout.flush()
    out = {}
    for k in keys:
        info = head[k]; s, e = info["data_offsets"]
        assert info["dtype"] in ("F32", "F16"), info["dtype"]
        arr = np.frombuffer(bytes(blob[s - lo:e - lo]), dtype=DT[info["dtype"]]).reshape(info["shape"])
        out[k[len("vision_model."):]] = torch.from_numpy(arr.copy()).to(torch.bfloat16)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cfg = json.loads(urllib.request.urlopen(base + "config.json").read())
    pre = json.loads(urllib.request.urlopen(base + "preprocessor_config.json").read())
    save_file(out, a.out, metadata={"repo": a.repo, "vision_config": json.dumps(cfg["vision_config"]),
                                    "preprocessor": json.dumps(pre)})
    print(f"\n### xong → {a.out} · {sum(v.numel() for v in out.values()) / 1e6:.0f}M tham số · "
          f"{os.path.getsize(a.out) / 1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
