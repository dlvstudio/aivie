# aivie — a Vietnamese language-model research log

Source code, training/eval logs and the full research journal of **aivie**, a one-person project exploring how to build a useful Vietnamese assistant model, from biologically-inspired Hebbian networks to hybrid Mamba + attention transformers trained on rented GPUs.

> Vietnamese original of this README (longer, with the early Hebbian design rules): [`README.vi.md`](README.vi.md).
> Most research notes (`TK_xxx_*.md`, `out/*.md`) are written in **Vietnamese**. English translations of the main reports (TK_100 → TK_108) are in [`en/`](#english-translations-of-the-main-reports).

This repo contains **no training data and no model weights** — see [Data & weights](#data--weights). Local paths are replaced by `~` and GPU-box IPs by `<IP>`.

## Timeline

| Phase | Notes | What it is |
|---|---|---|
| Hebbian graph LM | `TK_001` … `TK_062`, `RUNBOOK_*.md`, `src/` | A non-backprop, Hebbian "graph" language model learning Vietnamese the way a child would (grade-by-grade curriculum, `lessons/`). Ended with an honest finding: the counting core drifts; letting a small LM lead and the graph check worked better. |
| GPU scale-up | `TK_100` … `TK_103` | Hybrid Mamba + attention model (1.3B → 2.6B) trained from scratch on Vietnamese + code on rented H100/A100s. |
| Curriculum & assistant | `TK_104` | Learning by grade/stage, gated experts, teacher-in-the-loop (chat → grade → fix) assistant. |
| Bridge to a large model | `TK_105` | "Zamba2 thinks, our 1.3B speaks" — coupling with Zamba2-7B. |
| Vision | `TK_106` | VLM reading screenshots/documents with a single checkpoint. |
| Growing to 3B / 8K | `TK_107` | Continued pre-training to the 20B-token milestone. |
| Code & tool use | `TK_108_*` | Large code expert, tool calling, GRPO/DPO, sandboxed code execution, real-use evaluations vs Gemma 4 / Qwen. |
| Gemma 4 → hybrid Mamba | `TK_108_THI_NGHIEM_2_MAMBA_GEMMA12B.md`, `scripts/exp2_*.py` | Converting Gemma 4 12B attention layers to Mamba-2 on a Mac M4 (MLX). Layer-wise distillation works for single layers; long-range retrieval needs ≥4 of 8 global attention layers kept, evenly spread. |
| nn-aivie3 | `nn-aivie3/` | RMSH — realtime multimodal streaming hybrid; small demos (self-driving city, robot soccer, CNN vision with face memory). |

## English translations of the main reports

The Vietnamese originals stay in the repository root; each English file in `en/` has the same name.

| Report | Topic |
|---|---|
| [TK_100](en/TK_100_GPU_SCALEUP_PLAN.md) | GPU scale-up plan: CPT vs from-scratch, BTX experts, budget tiers, data sources, the 1-GPU 1.3B hybrid roadmap |
| [TK_101](en/TK_101_M4_DATA_RUNBOOK.md) | Runbook: building pre-training data on the M4 and pushing it to the GPU box |
| [TK_102](en/TK_102_GPU_SMOKE_FINDINGS.md) | GPU smoke test: two real bugs + real throughput benchmark |
| [TK_103](en/TK_103_VAST_TRAIN_HANDOFF.md) | Training the 1.3B hybrid on a Vast H100: config, traps, VMLU, RAG, long context, Zamba2 pivots, clean evaluation |
| [TK_104](en/TK_104_CURRICULUM_RESEARCH.md) | Curriculum research (learning by grade, revision, depth), BTX experts, building the 1.3B assistant v2 → v5 |
| [TK_105](en/TK_105_CAU_NOI_ZAMBA2.md) | Hidden-state bridge 1.3B ↔ Zamba2 ("Zamba2 thinks, the 1.3B speaks") |
| [TK_106](en/TK_106_VLM_DOC_ANH.md) | A single-checkpoint VLM reading screenshots and documents |
| [TK_107](en/TK_107_NUOI_LON_3B.md) | Growing the trunk to ~2.6B with 8K context; the 20B-token milestone |
| [TK_108 code & tool use](en/TK_108_CODE_TOOL_USE.md) | Code + tool use: sandboxed agent, verified data, DPO/GRPO, large code expert cx3 → cx3e |
| [TK_108 cx3b architecture](en/TK_108_KIEN_TRUC_CX3B.md) | Architecture of the cx3b checkpoint |
| [TK_108 c6l1 evaluation](en/TK_108_DANH_GIA_C6L1_THUC_TE.md) | Real-world evaluation of the latest SFT checkpoint (c6l1) |
| [TK_108 Gemma 4 evaluation](en/TK_108_DANH_GIA_GEMMA4_THUC_TE.md) · [Vietnamese/office/tools](en/TK_108_DANH_GIA_GEMMA4_VI_TOOL_OFFICE.md) | Gemma 4 26B-A4B on the M4 as a reference |
| [TK_108 Qwen-Coder evaluation](en/TK_108_DANH_GIA_QWEN_CODER_THUC_TE.md) | Qwen2.5-Coder on the M4 as a reference |
| [TK_108 Gemma → Mamba distillation plan](en/TK_108_KE_HOACH_CHUNG_CAT_GEMMA_MAMBA.md) | Option 2: distilling Gemma 4 into a hybrid Mamba model · P0 checks |
| [TK_108 Experiment 2](en/TK_108_THI_NGHIEM_2_MAMBA_GEMMA12B.md) | Replacing Gemma 4 12B attention layers with Mamba-2 on the M4 |

## Repository layout

```
src/           core library (Hebbian network, hybrid models, tokenizer)
scripts/       training, evaluation, data building and GPU-box run scripts
tests/         unit tests
curriculum/    curriculum generators
lessons/       lesson definitions for the child-like curriculum
config/        configs
viz/           visualisation tools
docs/          design documents
out/           logs, metrics (JSON) and written reports (*.md)
nn-aivie3/     the RMSH line of work and browser demos
TK_*.md        research journal, one file per work package
```

## Data & weights

See [`DATA.md`](DATA.md) for the public source datasets and the project's Hugging Face repositories.

## License

[MIT](LICENSE) for the code and documents in this repository. Datasets and third-party models referenced here keep their own licenses.
