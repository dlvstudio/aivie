# Training process, hardware and cost

This page summarises how the aivie models were trained: the stages, what was done in each, which GPUs were used and roughly what it cost. Details and numbers for each stage are in the linked reports (Vietnamese originals in the root, English translations in [`en/`](en/)).

**About the costs.** Hourly prices and run durations below were recorded at the time. Where a box's total bill was not logged, the cost is an **estimate** (price × wall-clock time, rounded) and marked with `~`. Apart from the FPT smoke test and the first ~11K pre-training steps, every box was rented on demand from Vast.ai. Teacher/judge models ran on the same rented GPUs or on the local Mac, so there was no API spending.

## Hardware used

| Hardware | Where | Used for |
|---|---|---|
| Apple Mac M4, 24 GB | local | Hebbian graph LM (TK_001–062), data building and tokenizer, smoke tests, MLX experiments, local evaluation and chat UI, Gemma 4 / Qwen reference runs (Ollama), Gemma 4 → Mamba experiments |
| 1× H100 80 GB | FPT AI Factory, 78,592 VND/h (~$3.02/h) | GPU smoke test and the start of the 1.3B pre-training |
| 1× H100 SXM 80 GB | Vast.ai, $1.83–2.28/h | 1.3B pre-training, SFT, curriculum research, experts, VLM, Zamba2 bridge, P3, TK_108 C1–C5 |
| 4× H100 SXM 80 GB | Vast.ai | P2: continued pre-training of the grown ~2.5B trunk at 8K context |
| 2× A100 SXM4 80 GB (500 W) | Vast.ai, $1.67/h | Large code expert cx3, C6 (whole trunk +10B tokens), SFT A/B, VLM on the new trunk, final teacher loop c6l1 |

Measured throughput, for comparison: 1.3B on 1× H100 ~26K tok/s (MFU ~20%); 2.5B code expert on 1× H100 ~9.0K tok/s vs 2× A100 9.1K tok/s at $1.67/h vs 4× A100 17.9K tok/s at $6.4/h. Per token, the 2× A100 box was the cheapest option we found. Lesson: check `nvidia-smi -q -d POWER` right after renting. One Vast H100 host was power-capped at 500/700 W and only gave 15K tok/s.

## Stages

| # | Dates (2026) | Stage | What was done | GPU | Time | Cost |
|---|---|---|---|---|---|---|
| 0 | → Aug | Hebbian graph LM | Non-backprop graph LM, child-like curriculum, small LM-led generation, MoE combiner, Mamba on the M4 | Mac M4 | months | $0 |
| 1 | early Sep | Data + tokenizer ([TK_101](en/TK_101_M4_DATA_RUNBOOK.md)) | Vietnamese + code pre-training corpus, 18.8B tokens, BPE 32K. Built on the Mac and moved via the HF Hub: ~2.3 h, versus ~17–25 h and ~$60 if built on the box | Mac M4 | ~1 day | $0 |
| 2 | early Sep | GPU smoke test ([TK_102](en/TK_102_GPU_SMOKE_FINDINGS.md)) | Found two real bugs; throughput benchmark | 1× H100 (FPT) | hours | ~$10 |
| 3 | Sep 9 → 29 | **1.3B pre-training** ([TK_103](en/TK_103_VAST_TRAIN_HANDOFF.md)) | HybridLM 1.277B (42 Mamba + 6 attention), 18.8B tokens, 143K steps, WSD. First ~11K steps on FPT, then Vast. One Vast box was power-capped (15K tok/s); another was lost with no HF backup, so the run restarted from step 11K. The final full-power box ran Sep 21 → 29 at 26.2K tok/s (~$24 per billion tokens) | 1× H100 | ~20 days total | ~$1,000 |
| 4 | Sep 29 | SFT + evaluation ([TK_103](en/TK_103_VAST_TRAIN_HANDOFF.md)) | Chat SFT, 2 epochs (~3 h); VMLU, RAG, long context, clean eval set eval_v1; Zamba2 pilot (~1.5 h) | 1× H100 (Vast) | ~5 h | ~$12 |
| 5 | Sep 29 → Oct 6 | Curriculum, experts, assistant, bridge, vision ([TK_104](en/TK_104_CURRICULUM_RESEARCH.md), [TK_105](en/TK_105_CAU_NOI_ZAMBA2.md), [TK_106](en/TK_106_VLM_DOC_ANH.md)) | 19 curriculum runs (9–10 in parallel); gated experts (code 1.1B tokens); teacher loop with Gemma-4-31B FP8 on vLLM; DailyDialog translation (~10 GPU-h); 1.3B assistant v2 → v5; Zamba2-7B bridge; VLM v1 → v4 | 1× H100 (Vast) | ~1 week | ~$350 |
| 6 | Oct 6 → 7 | P1: data for growing ([TK_107](en/TK_107_NUOI_LON_3B.md)) | 16.7B new tokens (86 shards): edu-scored Vietnamese web, code, math; published as HF datasets | 1× H100 box (≥500 GB disk, ≥32 CPU) | ~1 day | ~$50 |
| 7 | Oct 7 → 11 | **P2: grow to ~2.5B + 8K context** ([TK_107](en/TK_107_NUOI_LON_3B.md)) | Identity-block insertion 48 → 96 blocks (84 Mamba + 12 attention), RoPE 1e6, 8K context. Three phases: A1 4B tokens (~54K tok/s), A2 6B (~42K), B 10B (~35K, grad-ckpt). Val bits 3.47 → 2.31 | 4× H100 SXM (Vast) | ~4.8 days | ~$1,000 |
| 8 | Oct 12 → 18 | P3 + TK_108 C1–C5 ([TK_107](en/TK_107_NUOI_LON_3B.md), [TK_108](en/TK_108_CODE_TOOL_USE.md)) | SFT chat, 3 gated experts at 12 points, VLM, teacher loop; code + tool use: sandboxed agent, verified data, teacher Gemma-4-31B (beat Qwen3-Coder 41/50 vs 28/50 as teacher), DPO, GRPO | 1× H100 (Vast) | ~6 days | ~$300 |
| 9 | Oct 19 → 22 | Large code expert cx3 → cx3e ([TK_108](en/TK_108_CODE_TOOL_USE.md), [cx3b](en/TK_108_KIEN_TRUC_CX3B.md)) | cx3 pre-training 1.5B tokens (23K steps, 47 h, 9.1K tok/s), SFT, tool-use top-up → cx3b (tool 55 · HumanEval 13.4 · MBPP 23.3); C7 real-world task set (2 vLLM Gemma teachers); cx3c/d/e | 2× A100 (Vast) | ~4 days | ~$160 |
| 10 | Oct 23 → Nov 2 | **C6: whole trunk +10B tokens** ([TK_108](en/TK_108_CODE_TOOL_USE.md)) | Continued pre-training of the whole 2.49B trunk from the 20B-token stable base. Code 45% · Vietnamese 30% · edu 12% · math 8%, ~38K steps, WSD | 2× A100 (Vast) | ~10 days | ~$400 |
| 11 | Nov 2 → 5 | SFT A ‖ B, vision, teacher loop | SFT A (8.2K steps) vs B (13.2K steps): B won. Vision c6a → c6b on B. Teacher loop c6l1 → **[nn-aivie-2p5b-c6l1](https://huggingface.co/dlvstudio/nn-aivie-2p5b-c6l1)** (latest). Box destroyed Nov 5 after backup to HF | 2× A100 (Vast) | ~3 days | ~$120 |
| 12 | Nov 6 → | Real-use evaluation, Gemma 4 → Mamba ([c6l1 eval](en/TK_108_DANH_GIA_C6L1_THUC_TE.md), [Exp. 2](en/TK_108_THI_NGHIEM_2_MAMBA_GEMMA12B.md)) | 22 real Vietnamese coding tasks vs Gemma 4 / Qwen-Coder; replacing Gemma 4 12B attention layers with Mamba-2 (MLX) | Mac M4 | — | $0 |

**Approximate total GPU rental: ~$3,400** (≈ $2,000 for the 1.3B line and the 2.5B growth, ≈ $1,000 for the 4× H100 P2 run, ≈ $700 on the 2× A100 box).

## Token budget of the latest model

| Stage | Tokens |
|---|---|
| 1.3B pre-training | 18.8B |
| P2 growth to ~2.5B / 8K (A1 + A2 + B) | 20B |
| C6 whole-trunk continued pre-training | 10B |
| Code expert cx3 pre-training | 1.5B |
| SFT / experts / VLM / teacher loops | small (≤ 1B) |

## Teacher and judge models

All of them are open-weight and were self-hosted, so there was no API cost and their terms of use were respected:

- **Gemma-4-31B (FP8, vLLM)** on the rented GPUs. Used for data generation, translation, grading, edu scoring and the teacher loops.
- **Gemma 4 / Qwen2.5-Coder** via Ollama on the M4, for reference evaluations.
- **SigLIP2** as the vision tower.

## What we would do differently

- Back up checkpoints to the HF Hub at every milestone from day one. One lost box cost about a week of H100 time.
- Benchmark tokens per dollar before a long run. 2× A100 beat 1× H100 and 4× A100 for the 2.5B model.
- Check power caps and disk layout right away (`/workspace` on Vast is on the same overlay disk; destroying the box wipes it).
- Build data on the cheapest machine that has a good uplink, not on the GPU box.
