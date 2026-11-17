"""Đẩy checkpoint lên Hugging Face (python API — đừng dùng `hf` CLI, xem TK_103).

Bài-học TK_103: ckpt 15k–90k chỉ nằm trên ổ Vast → Destroy = mất. Rolling `--out`
sống qua Stop nhưng KHÔNG sống qua Destroy. Mỗi mốc `--hf-every` phải có bản trên HF.

Private Hub: KHÔNG xóa bản đang dùng trước khi bản mới xong. Upload sang
slot `{repo}-a` / `{repo}-b`, ghi latest.json (kèm slot_repo), *mới* xóa slot cũ.
Box offline giữa chừng → slot cũ vẫn còn. Cần quota ~2×15GB; thiếu thì GIỮ bản cũ
(không xóa) và log 403 — [nâng storage](https://huggingface.co/settings/billing).

Token WRITE: env `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, hoặc file `$HF_HOME/token`.
KHÔNG đưa token vào argv, log, git, hay chat.
"""
from __future__ import annotations

import json, os, shutil, sys, threading, time
from io import BytesIO

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_lock = threading.Lock()
_busy_step: int | None = None
_threads: list[threading.Thread] = []

_LARGE = 1_000_000_000  # ≥1GB → reset repo, không overwrite LFS


def hf_token() -> str:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    p = os.path.join(home, "token")
    if os.path.isfile(p):
        t = open(p).read().strip()
        if t:
            return t
    return ""


def warn_if_no_hf_repo(repo: str, every: int) -> None:
    if repo:
        tok = hf_token()
        if not tok:
            print("⚠️ --hf-repo nhưng KHÔNG thấy HF_TOKEN WRITE. Backup sẽ fail. "
                  "Nạp token vào $HF_HOME/token (chmod 600) — ĐỪNG dán token vào chat/git.",
                  flush=True)
        else:
            print(f"HF backup: {repo} mỗi {every} step · token OK (không in token)", flush=True)
        return
    print("⚠️ KHÔNG --hf-repo: ckpt chỉ sống trên ổ box. Vast Destroy = mất hết. "
          "Bài-học TK_103 → thêm: --hf-repo donaldluong/nn-aivie-hybrid-1p3b-ckpt --hf-every 5000",
          flush=True)


def _is_quota(err: Exception) -> bool:
    s = str(err).lower()
    return "storage limit" in s or "403 forbidden" in s


def _slot_repos(pointer: str) -> tuple[str, str]:
    return f"{pointer}-a", f"{pointer}-b"


def _load_latest(repo: str) -> dict | None:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, "latest.json", repo_type="model")
        return json.load(open(p))
    except Exception:
        return None


def _inactive_slot(pointer: str, active: str | None) -> str:
    a, b = _slot_repos(pointer)
    return b if active == a else a


def _write_latest(api, repo: str, meta: bytes, msg: str) -> None:
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=True)
    api.upload_file(
        path_or_fileobj=BytesIO(meta), path_in_repo="latest.json",
        repo_id=repo, repo_type="model", commit_message=msg,
    )


def _drop_old_slot(api, pointer: str, old: str, dest: str, path_in_repo: str) -> None:
    """Chỉ gọi SAU KHI dest đã có file đủ byte. Không đụng dest."""
    if not old or old == dest:
        return
    if old == pointer:
        try:
            api.delete_file(path_in_repo, repo_id=pointer, repo_type="model",
                            commit_message="drop legacy blob; pointer-only")
            print(f"HF xóa blob cũ trên pointer {pointer}", flush=True)
        except Exception as e:
            print(f"HF xóa blob pointer: {type(e).__name__}: {e}", flush=True)
        return
    try:
        api.delete_repo(repo_id=old, repo_type="model")
        print(f"HF xóa slot cũ {old} (bản mới đã nằm ở {dest})", flush=True)
    except Exception as e:
        print(f"HF xóa slot cũ: {type(e).__name__}: {e}", flush=True)


def _upload(local: str, repo: str, step: int, path_in_repo: str, token: str) -> None:
    """repo = pointer. Upload slot mới trước — không xóa bản đang dùng."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    t0 = time.time()
    size = os.path.getsize(local)
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    pointer = repo
    prev = _load_latest(pointer)
    if not prev:
        for r in (pointer, *_slot_repos(pointer)):
            prev = _load_latest(r)
            if prev:
                prev.setdefault("slot_repo", r)
                break
    active = (prev or {}).get("slot_repo") or (pointer if prev else None)
    dest = _inactive_slot(pointer, active)
    meta_obj = {
        "step": step,
        "file": path_in_repo,
        "bytes": size,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slot_repo": dest,
        "pointer_repo": pointer,
    }
    meta = json.dumps(meta_obj, indent=2).encode()
    print(f"HF A/B: giữ {active or '(chưa có)'} → upload {dest} @step {step}", flush=True)

    sealed = False
    try:
        api.create_repo(repo_id=dest, repo_type="model", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=local, path_in_repo=path_in_repo,
            repo_id=dest, repo_type="model",
            commit_message=f"ckpt step {step} ({size / 1e9:.1f}GB)",
        )
        infos = api.get_paths_info(dest, [path_in_repo], repo_type="model")
        got = getattr(infos[0], "size", None) if infos else None
        if got != size:
            raise RuntimeError(f"HF size lệch dest={got} local={size} — KHÔNG xóa slot cũ")
        _write_latest(api, dest, meta, f"latest.json step {step}")
        sealed = True
        _write_latest(api, pointer, meta, f"pointer step {step} -> {dest}")
    except Exception as e:
        print(f"HF slot mới fail — GIỮ bản cũ {active}: {type(e).__name__}: {e}", flush=True)
        if not sealed and dest != active:
            try:
                api.delete_repo(repo_id=dest, repo_type="model")
                print(f"HF dọn slot dở {dest}", flush=True)
            except Exception:
                pass
        raise

    _drop_old_slot(api, pointer, active, dest, path_in_repo)
    print(f"HF backup OK {dest}/{path_in_repo} @step {step} · {size/1e9:.1f}GB · "
          f"{time.time()-t0:.0f}s", flush=True)


def _worker(src: str, snap: str, repo: str, step: int, path_in_repo: str) -> None:
    global _busy_step
    try:
        shutil.copy2(src, snap)
        tok = hf_token()
        if not tok:
            print(f"HF backup FAIL @step {step}: không có HF_TOKEN WRITE", flush=True)
            return
        _upload(snap, repo, step, path_in_repo, tok)
    except Exception as e:
        print(f"HF backup FAIL @step {step}: {type(e).__name__}: {e}", flush=True)
    finally:
        try:
            if os.path.isfile(snap):
                os.remove(snap)
        except OSError:
            pass
        with _lock:
            if _busy_step == step:
                _busy_step = None


def start_hf_backup(path: str, repo: str, step: int, path_in_repo: str = "hybrid_1p3b.pt",
                    wait: bool = False) -> None:
    """Đẩy `path` lên HF. Nền mặc-định; `wait=True` cho mốc cuối (đừng để daemon chết)."""
    global _busy_step
    if not repo:
        return
    if not os.path.isfile(path):
        print(f"HF backup skip @step {step}: không thấy {path}", flush=True)
        return
    with _lock:
        if _busy_step is not None:
            print(f"HF backup skip @step {step}: đang upload step {_busy_step} "
                  "(mốc này bỏ, mốc sau/cuối vẫn đẩy)", flush=True)
            return
        _busy_step = step
    snap = f"{path}.hf_step{step}.pt"
    t = threading.Thread(
        target=_worker, args=(path, snap, repo, step, path_in_repo),
        name=f"hf-backup-{step}", daemon=False,
    )
    t.start()
    _threads.append(t)
    if wait:
        t.join()


def maybe_hf_backup(path: str, repo: str, step: int, every: int,
                    path_in_repo: str = "hybrid_1p3b.pt") -> None:
    if not repo or every <= 0 or step % every != 0:
        return
    print(f"HF backup queue @step {step} → {repo}/{path_in_repo}", flush=True)
    start_hf_backup(path, repo, step, path_in_repo, wait=False)


def wait_hf_backups() -> None:
    for t in list(_threads):
        t.join()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Push one checkpoint to HF (WRITE token via env/file).")
    ap.add_argument("--path", required=True)
    ap.add_argument("--repo", default="donaldluong/nn-aivie-hybrid-1p3b-ckpt")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--name", default="hybrid_1p3b.pt", help="path_in_repo")
    a = ap.parse_args()
    if not hf_token():
        sys.exit("không có HF_TOKEN WRITE — nạp vào env hoặc $HF_HOME/token")
    if a.step <= 0:
        try:
            import torch
            a.step = int(torch.load(a.path, map_location="cpu", weights_only=False).get("step", 0))
        except Exception:
            pass
    start_hf_backup(a.path, a.repo, a.step, a.name, wait=True)
    wait_hf_backups()


if __name__ == "__main__":
    main()
