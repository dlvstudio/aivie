# TK_101 — Runbook: build pre-training data on the M4 and push it to the GPU box

> Written: 2026-09-05 · Follows [TK_100_GPU_SCALEUP_PLAN](TK_100_GPU_SCALEUP_PLAN.md) §15
> Status: **operational RUNBOOK** (not a plan) — used when building data on the M4 machine (Mac, CPU/MPS)
> instead of (or in addition to) building directly on the GPU box.
> *English translation of the Vietnamese original in the repository root.*

## 0. Why this path exists

Building directly on the GPU box (data-centre, fast network) looked like it would be faster, but real measurements on the FPT H100
container showed that **the bottleneck is reading/decoding the streaming parquet from HuggingFace in the
main process (single-threaded)**, not CPU tokenisation (already parallel over 127 workers). Throughput slid
from ~700K tok/s (initial burst) to **~195–300K tok/s** after a few minutes → a real ETA for 18B tokens of
**~17–25h**, not the ~7h first estimated. If the M4 already has an equivalent or larger corpus (built
earlier, not limited by data-centre bandwidth at build time), **re-using it + uploading** is much cheaper
than rebuilding from scratch on the box.

## 1. Build (or confirm you already have) the data on the M4

```bash
cd nn-aivie && export PYTHONPATH=$(pwd)/src
python scripts/build_pretrain_pipeline.py \
  --sources fineweb2 wiki --no-dedup \
  --limit 100000000 --max-tokens 18e9 --shard-tokens 400000000 \
  --out data/pretrain_vi
```

**Trap we hit:** the default `--limit` is only **5000** docs per source (a value meant for smoke tests) — if you forget
to set it high, the pipeline stops at a few tens of M tokens even though `--max-tokens` has not been reached. Always set `--limit` far
above the number of docs you need so that `--max-tokens` is the real stopping condition.

Output: directory `data/pretrain_vi/` containing `shard_00000.npy … shard_NNNNN.npy` (uint16, each shard
~`--shard-tokens` tokens) + `manifest.json`.

## 2. Verify before uploading (mandatory)

```bash
python -c "import json; m=json.load(open('data/pretrain_vi/manifest.json')); print(m['V'], m['EOS'], m['dtype'], m['total_tokens'], m['num_shards'])"
```

- `V` must match **the very same file** `checkpoints/bpe_v1.json` used on the GPU box (vocab size
  32002 at the time of writing). **Different tokenizer → wrong token ids, training produces garbage** — if in doubt,
  compare checksums: `shasum -a 256 checkpoints/bpe_v1.json` must be identical on both machines.
- `dtype` must be `uint16` (V ≤ 65535).
- `total_tokens` — check against the Chinchilla target (~18–26B tokens, see TK_100 §15).

## 3. Upload to the GPU box

> ⚠️ **UPDATE (Claude-Mac, 2026-09-05): rsync-over-SSH DOES NOT work on FPT serverless.**
> Verified from the Mac: **every port on the box times out** — `tcp-endpoint.serverless.fptcloud.com`
> only opens **443 (HTTPS)**; the forwarded SSH port (e.g. 37457) and the direct IP (109.237.64.x) are
> **not routed** (high-port portquiz from the Mac works → the problem is on FPT's side, not the home firewall).
> → **Drop §3a (rsync/SSH). Use §3b (HuggingFace Hub over HTTPS).**

### 3a. rsync over SSH — ❌ DOES NOT work on FPT (kept for reference if switching to a provider with real SSH)
```bash
rsync -avz --progress --partial -e "ssh -p <PORT>" \
  data/pretrain_vi/ root@<IP>:/workspace/data/pretrain_vi_m4/
```
Only usable where the TCP endpoint/SSH actually routes (Vast/RunPod/VM with a public IP). FPT serverless: no.

### 3b. HuggingFace Hub (HTTPS) — ✅ THE WORKING PATH
Both the Mac and the box can reach `huggingface.co:443`. Push the data (compressed as `pretrain_vi.tar.zst`, ~24GB)
to a **private dataset**, and the box downloads it (fast data-centre network).

**On the Mac** (once: `huggingface-cli login` and enter the token; the upload can resume):
```bash
PYTHONPATH=src .venv/bin/huggingface-cli upload donaldluong/nn-aivie-pretrain-vi \
  data/pretrain_vi.tar.zst pretrain_vi.tar.zst --repo-type dataset --private
```
(~2.3h @ 23Mbps; if it drops, run it again — it resumes. Off-clock, no GPU cost.)

**✅ STATUS (2026-09-06): UPLOAD COMPLETE** — file `pretrain_vi.tar.zst` (24.13GB, 18.8B tokens)
is on `donaldluong/nn-aivie-pretrain-vi` (private, verified). The box only needs to download it.

**On the box** (use `hf`, not the deprecated `huggingface-cli`):
```bash
hf auth login          # paste donaldluong's read-scope HF token  (or: export HF_TOKEN=<token>)
hf download donaldluong/nn-aivie-pretrain-vi pretrain_vi.tar.zst --repo-type dataset --local-dir /workspace/data
tar -I zstd -xf /workspace/data/pretrain_vi.tar.zst -C /workspace/data   # → data/pretrain_vi/
# verify: python -c "import json;m=json.load(open('/workspace/data/pretrain_vi/manifest.json'));print(m['total_tokens'],m['num_shards'],m['V'])"
#   → must be: 18800060992 47 32002
```
Then pass `--data /workspace/data/pretrain_vi` to train_hybrid_scale.py.

**Comparison:** rebuilding on the box ~17–25h + ~$60 of GPU (measured, network-bound) · HF Hub ~2.3h upload
(off-box, $0) + fast download on the box → **HF Hub clearly wins**.

## 4. On the box: confirm everything arrived + point training at the data

```bash
python -c "import json; m=json.load(open('/workspace/data/pretrain_vi_m4/manifest.json')); print(m)"
du -sh /workspace/data/pretrain_vi_m4
```

Match `total_tokens`, `num_shards`, `V`, `EOS` against the values measured on the M4 (step 2). If they match, use
`--data /workspace/data/pretrain_vi_m4` directly for `scripts/train_hybrid_scale.py` (no need to rename
or merge into `pretrain_vi/` unless you want to add other sources).

## 5. Troubleshooting

| Symptom | Usual cause | Fix |
|:---|:---|:---|
| `ssh: connect to host … port 22: Connection timed out` | Cloud NAT uses a port other than 22 | Check the provider console for the real public port |
| rsync drops midway | Unstable M4↔DC network | Re-run **the same command** (it has `--partial`) to resume |
| `V` differs between M4 and box | The two machines use different copies of `bpe_v1.json` | Copy exactly one tokenizer file shared by both build and training |
| Total size after rsync differs from the M4 | Transfer not finished or partially failed | Compare `du -sh` + shard counts on both sides; rsync again if anything is missing |
