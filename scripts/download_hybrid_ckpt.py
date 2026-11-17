"""Tải ckpt hybrid 1.3B từ HF (pointer + A/B slot) về Mac.

    PYTHONPATH=src .venv/bin/python scripts/download_hybrid_ckpt.py [--keep-old]
    ... download_hybrid_ckpt.py --repo donaldluong/nn-aivie-hybrid-1p3b-sft --prefix hybrid_1p3b_sft
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time

_ap = argparse.ArgumentParser()
_ap.add_argument("--repo", default="donaldluong/nn-aivie-hybrid-1p3b-ckpt")
_ap.add_argument("--prefix", default="hybrid_1p3b")
_ap.add_argument("--keep-old", action="store_true")
ARGS = _ap.parse_args()
POINTER = ARGS.repo
PREFIX = ARGS.prefix
OUT_DIR = "checkpoints/hf_hybrid"


def _read_latest():
    from huggingface_hub import hf_hub_download
    for r in (POINTER, f"{POINTER}-a", f"{POINTER}-b"):
        try:
            p = hf_hub_download(r, "latest.json", repo_type="model", force_download=True)
            meta = json.load(open(p))
            slot = meta.get("slot_repo") or r
            print(f"latest từ {r}: step={meta.get('step')} slot={slot}", flush=True)
            return meta, slot
        except Exception as e:
            print(f"skip {r}: {type(e).__name__}", flush=True)
    sys.exit("không đọc được latest.json trên pointer/slot")


def _download_once() -> str:
    from huggingface_hub import hf_hub_download
    meta, slot = _read_latest()
    step = int(meta.get("step") or 0)
    name = meta.get("file") or "hybrid_1p3b.pt"
    dest = os.path.join(OUT_DIR, f"{PREFIX}_step{step}.pt")
    if os.path.isfile(dest) and os.path.getsize(dest) == int(meta.get("bytes") or 0):
        print(f"đã có {dest}", flush=True)
        return dest
    print(f"DOWNLOAD {slot}/{name} → {dest}", flush=True)
    path = hf_hub_download(repo_id=slot, filename=name, repo_type="model", local_dir=OUT_DIR)
    if path != dest and os.path.isfile(path):
        if os.path.exists(dest):
            os.remove(dest)
        os.rename(path, dest)
    want = int(meta.get("bytes") or 0)
    if want and os.path.getsize(dest) != want:
        raise RuntimeError(f"size mismatch {os.path.getsize(dest)} != {want}")
    with open(os.path.join(OUT_DIR, "latest_local.json" if PREFIX == "hybrid_1p3b" else f"latest_local_{PREFIX}.json"), "w") as f:
        json.dump({**meta, "local_path": dest}, f, indent=2)
    print(f"OK {dest} {os.path.getsize(dest)} bytes", flush=True)
    return dest


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    dest = None
    for attempt in range(1, 5):
        try:
            dest = _download_once()
            break
        except SystemExit:
            raise
        except Exception as e:
            # slot có thể bị xoá giữa chừng khi watcher đẩy bản mới → đọc lại latest.json
            print(f"attempt {attempt} fail: {type(e).__name__}: {e}", flush=True)
            time.sleep(30)
    if not dest:
        sys.exit("download fail sau 4 lần")
    if not ARGS.keep_old:
        for old in glob.glob(os.path.join(OUT_DIR, f"{PREFIX}_step*.pt")):
            if os.path.abspath(old) != os.path.abspath(dest):
                os.remove(old)
                print(f"xoá bản cũ {old}", flush=True)
    print(dest)


if __name__ == "__main__":
    main()
