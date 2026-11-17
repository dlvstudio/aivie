# aivie — a Vietnamese language-model research log

Source code, training/eval logs and the full research journal of **aivie**, a one-person project exploring how to build a useful Vietnamese assistant model, from biologically-inspired Hebbian networks to hybrid Mamba + attention transformers trained on rented GPUs.

> Vietnamese original of this README (longer, with the early Hebbian design rules): [`README.vi.md`](README.vi.md).
> Most research notes (`TK_xxx_*.md`, `out/*.md`) are written in **Vietnamese**.

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
