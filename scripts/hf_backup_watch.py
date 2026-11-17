"""Sidecar HF backup — A/B slot, không xóa bản cũ trước khi bản mới xong.

Mỗi LƯU (@ --every, mặc-định 1000 ≈ 40 phút) đẩy HF. Offline giữa upload
thì slot cũ vẫn còn. Destroy chỉ mất tối đa 1 mốc save.

    python -u scripts/hf_backup_watch.py --every 1000
"""
from __future__ import annotations

import os, re, subprocess, sys, time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", "/workspace/.hf")

SAVE_RE = re.compile(r"LƯU .+ @step (\d+)")


def hf_latest_step(pointer: str) -> int:
    best = 0
    try:
        from huggingface_hub import hf_hub_download
        import json
    except Exception:
        return 0
    for r in (pointer, f"{pointer}-a", f"{pointer}-b"):
        try:
            p = hf_hub_download(r, "latest.json", repo_type="model")
            best = max(best, int(json.load(open(p)).get("step") or 0))
        except Exception:
            continue
    return best


def push(path: str, repo: str, step: int, name: str) -> None:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_backup_ckpt.py")
    env = os.environ.copy()
    env["HF_HOME"] = env.get("HF_HOME") or "/workspace/.hf"
    env["HF_HUB_DISABLE_XET"] = "1"
    print(f"WATCH push step {step}", flush=True)
    r = subprocess.run(
        [sys.executable, "-u", script, "--path", path, "--repo", repo,
         "--step", str(step), "--name", name],
        env=env,
    )
    print(f"WATCH push exit {r.returncode} step {step}", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/workspace/nn-aivie/train.log")
    ap.add_argument("--path", default="/workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt")
    ap.add_argument("--repo", default="donaldluong/nn-aivie-hybrid-1p3b-ckpt")
    ap.add_argument("--name", default="hybrid_1p3b.pt")
    ap.add_argument("--every", type=int, default=1000,
                    help="đẩy mỗi N step (khớp --save-every). 1000 ≈ 40 phút nếu box Destroy.")
    ap.add_argument("--settle", type=float, default=8.0, help="giây sau LƯU trước khi copy")
    a = ap.parse_args()

    done: set[int] = set()
    print(f"WATCH start log={a.log} every={a.every} (mất tối đa ~{a.every} step nếu offline)", flush=True)
    with open(a.log, "r", errors="replace") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(2)
                continue
            m = SAVE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if a.every <= 0 or step % a.every != 0 or step in done:
                continue
            print(f"WATCH LƯU @{step}, settle {a.settle}s", flush=True)
            time.sleep(a.settle)
            got = hf_latest_step(a.repo)
            if got >= step:
                print(f"WATCH skip {step}: HF đã có {got}", flush=True)
                done.add(step)
                continue
            if not os.path.isfile(a.path):
                print(f"WATCH skip {step}: không thấy {a.path}", flush=True)
                continue
            push(a.path, a.repo, step, a.name)
            done.add(step)


if __name__ == "__main__":
    main()
