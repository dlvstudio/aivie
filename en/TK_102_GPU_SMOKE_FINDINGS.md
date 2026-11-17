# TK_102 — GPU box: two real bugs surfaced by the smoke test + real throughput benchmark

> Written: 2026-09-06 (Claude on the GPU box, after receiving the 18.8B-token data from the M4 via HF Hub)
> Follows [TK_100_GPU_SCALEUP_PLAN](TK_100_GPU_SCALEUP_PLAN.md) §15, [TK_101_M4_DATA_RUNBOOK](TK_101_M4_DATA_RUNBOOK.md)
> Purpose: report back to the M4 for further analysis — these two bugs directly affect the reliability of
> `train_hybrid_scale.py`; they are not environment issues.
> *English translation of the Vietnamese original in the repository root.*

## 1. Bug: `causal-conv1d` version does not match the `mamba_ssm 2.2.2` API

**Symptom:** `from mamba_ssm import Mamba` imports fine, but the **real forward pass** (10-step smoke test
on real data) crashes: `TypeError: causal_conv1d_fwd(): incompatible function arguments`.

**Cause:** `causal-conv1d>=1.5.0` (including `1.7.0` — the installed version, an old wheel in
`/workspace/wheels`) changed the C++ signature of `causal_conv1d_fwd`: it now takes a pre-allocated `out`
tensor and returns `None` (writes in place). `mamba_ssm 2.2.2` (`selective_scan_interface.py`) was written for
the **older** API: it calls it with 6–7 arguments and the function **returns a Tensor**. The mismatch → error on the very first forward.

**Fix applied:**
- Removed the old `causal_conv1d-1.7.0` wheel and rebuilt **`causal-conv1d==1.4.0`** — but the PyPI sdist for this
  version is **missing the `csrc/*.cpp` files** (`ninja: error: ... missing and no known rule to make it`) →
  it has to be built from the **original GitHub** `git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0`.
- The new wheel is saved as `/workspace/wheels/causal_conv1d-1.4.0-cp312-cp312-linux_x86_64.whl`
  (`reinit.sh` needs no change — its `*.whl` glob picks up the right file).
- `scripts/setup_container.sh` was fixed so a NEW box does not hit this again: pins
  `causal-conv1d==1.4.0` from git source + `mamba-ssm==2.2.2` (previously unpinned) + also adds
  the `MambaLMHeadModel` sed patch (previously only in `reinit.sh`; the original box-setup script lacked it).

**Question for the M4:** if we later want the newer causal-conv1d ≥1.5 optimisations (e.g.
`initial_states`/`return_final_states` for continuous-batching inference), we will need to *upgrade*
`mamba_ssm` to a version supporting the matching new API — do not hand-patch `selective_scan_interface.py`,
because the backward pass (`causal_conv1d_bwd`) changed its signature in the same way.

## 2. Bug: Python late-binding closure in `--grad-ckpt` (train_hybrid_scale.py)

**Symptom:** Enabling `--grad-ckpt` (with or without `--compile`) → crash on the first step:
`torch.utils.checkpoint.CheckpointError: Recomputed values ... have different metadata`, with completely
different shapes between saved/recomputed (e.g. saved `[2048,16384]` vs recomputed `[2048,2048]`).

**Cause (identified — not a Mamba kernel bug):**

```python
# HybridLM.forward(), before the fix:
for blk, nm in zip(self.blocks, self.norms):
    if self.grad_ckpt and self.training:
        x = x + torch.utils.checkpoint.checkpoint(lambda t: blk(nm(t)), x, use_reentrant=False)
```

`use_reentrant=False` **defers the recompute until `backward()`** (unlike reentrant mode, which recomputes
inside the forward). But `lambda t: blk(nm(t))` captures `blk`/`nm` **by reference** — a Python loop
does not create a new scope per iteration, so by the time backward runs (AFTER all 48 layers have run
forward), `blk`/`nm` in the closure point at the **last layer** of the loop. Every recompute
(whichever layer the checkpoint belongs to) calls the last layer by mistake → shape mismatch and completely wrong gradients
(without the `check_recomputed_tensors_match` safety check it would **silently train wrong** instead of crashing —
even more dangerous).

**Fix:** default-argument binding so each lambda keeps the right reference at creation time:
```python
x = x + torch.utils.checkpoint.checkpoint(
    lambda t, blk=blk, nm=nm: blk(nm(t)), x, use_reentrant=False)
```
Verified: bits fall along the same trajectory as the no-grad-ckpt baseline (15.7→13.3 after 10 steps).

## 3. Real throughput benchmark (H100, 1.277B model, ctx=2048, real 18.8B-token data)

| Config | tok/s (steady state) | MFU | Peak VRAM | Notes |
|---|---|---|---|---|
| baseline, bs=8×ga8 | ~24–26K | ~19–21% | 64.4GB/80GB (79%) | fastest |
| `--grad-ckpt`, bs=8 | ~16.4K | — | lower (not re-measured after the fix) | ~35% slower, unnecessary (VRAM to spare) |
| `--compile`, bs=8 | ~21.4K | — | — | **slower than baseline** — the Mamba/causal-conv1d CUDA kernels cannot be traced, torch.compile keeps breaking the graph around them (`Graph break due to unsupported builtin causal_conv1d_cuda...`), tracing overhead > fusion benefit |
| both, bs=8 | ~17.5K | — | — | |
| baseline, bs=10×ga8 | ~25–26.5K | ~19–21% | 74.7GB/80GB (92%) | slightly faster than bs=8, thinner VRAM margin for long unattended runs |
| baseline, bs=12×ga8 | — | — | **OOM** (needs 79GB+, tried to allocate another 2.93GiB at 77GB used) | over the ceiling |

**Conclusion:** `--grad-ckpt`/`--compile` are **not needed** at 1.3B/ctx=2048/bs=8 — VRAM is spare
(64GB/80GB), and both flags only slow things down because the custom CUDA kernels (Mamba) do not play well
with `torch.compile`, while `--grad-ckpt` trades speed for VRAM we are not short of. **Recommendation: use
the baseline (no flags), bs=8** — safer on VRAM than bs=10 for multi-day unattended runs, and the
tok/s difference is negligible (~24–26K vs ~25–26.5K).

## 4. Real ETA/cost vs the plan

Plan §15 assumed ~1 week / ~$500 for 26B tokens. Measured on the H100 (baseline, no flags):

- **18.8B tokens ÷ ~25K tok/s ≈ 209 hours ≈ 8.7 days**
- **8.7 days × ~78,592 VND/hour (~$3.02) ≈ ~$630** — about 25% over the original budget.

MFU ~19–21% is reasonable for a hybrid Mamba+Attention architecture (attention in only 1/8 of the layers, batch not
large) — not a misconfiguration; the original assumption in TK_100 §15 was simply more optimistic than what was
measured. Awaiting a decision: run the full 18.8B tokens (accept ~$630/8.7 days), or adjust the target.

## 5. Files changed (committed alongside)

- `scripts/train_hybrid_scale.py` — closure-bug fix (section 2)
- `scripts/setup_container.sh` — correct causal-conv1d/mamba-ssm version pins + sed patch (section 1)
- New wheel: `/workspace/wheels/causal_conv1d-1.4.0-*.whl` (not in git, per the rule of
  NEVER pushing binaries — it only exists on the box's `/workspace`)
