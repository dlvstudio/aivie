# Data & weights

This GitHub repository contains **no training/eval data and no model weights** (they are too large for git). Weights and the project's datasets live on Hugging Face (below); the scripts in `scripts/` can also rebuild the data from the original sources.

## Project repositories on Hugging Face (public)

| Repo | Type | Contents |
|---|---|---|
| **[donaldluong/nn-aivie-2p5b-c6l1](https://huggingface.co/donaldluong/nn-aivie-2p5b-c6l1)** | model | **Latest checkpoint.** `c6l1_vlm_bf16.pt` = 2.49B hybrid trunk after C6 + SFT-B + image reading + one teacher loop (self-contained); also `c6b_vlm_bf16.pt`, text-only `c6_sft_b_bf16.pt`, vision parts — see [en/TK_108_DANH_GIA_C6L1_THUC_TE.md](en/TK_108_DANH_GIA_C6L1_THUC_TE.md) |
| [donaldluong/nn-aivie-pretrain-vi](https://huggingface.co/datasets/donaldluong/nn-aivie-pretrain-vi) | dataset | Vietnamese pre-training corpus, 18.8B tokens (packed) |
| [donaldluong/nn-aivie-cpt-data](https://huggingface.co/datasets/donaldluong/nn-aivie-cpt-data) | dataset | Tokenized continued-pre-training shards (Vietnamese, English, maths, code) |
| [donaldluong/nn-aivie-vlm-data](https://huggingface.co/datasets/donaldluong/nn-aivie-vlm-data) | dataset | Vietnamese image-reading data (screenshots, OCR, charts, photos) — gated: access on request |

Older checkpoints of the same architecture (20B-token stable base, C6 base, SFT-A, cx3 expert pre-training, P3 expert backup and snapshots) and the P3 SFT / teacher-loop data are kept private for now.

Checkpoints are PyTorch `.pt` files (`{"cfg", "state", …}`) for the project's own `HybridLM` code (`scripts/train_hybrid_scale.py`, `scripts/hybrid_infer.py`, UI `scripts/serve_hybrid_ui.py`), not `transformers` models. They are research artifacts: see the evaluation reports before relying on them.

## Public source datasets used

- Vietnamese / multilingual: [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [bkai-foundation-models/vi-alpaca](https://huggingface.co/datasets/bkai-foundation-models/vi-alpaca), [5CD-AI/Viet-Doc-VQA](https://huggingface.co/datasets/5CD-AI/Viet-Doc-VQA), [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), [hungnm/vietnamese-medical-qa](https://huggingface.co/datasets/hungnm/vietnamese-medical-qa)
- Code / math / reasoning: [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2), [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts), [open-thoughts/OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M)
- General English: [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
- Base models used for distillation / comparison: Gemma 4 (Google), Qwen, Zamba2 (Zyphra) — see their own licenses.

Check each dataset's license before use.
