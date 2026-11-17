# Data & weights

This repository contains **no training/eval data and no model weights**: crawled text can contain personal information, some datasets do not allow redistribution, and teacher-model–generated data is not published here. The scripts in `scripts/` download and rebuild the data from the original sources.

## Project repositories on Hugging Face (public)

| Repo | Type | Contents |
|---|---|---|
| [donaldluong/nn-aivie-hybrid-2p6b-stable](https://huggingface.co/donaldluong/nn-aivie-hybrid-2p6b-stable) | model | 2.6B hybrid Mamba + attention, stable continued-pre-training checkpoint |
| [donaldluong/nn-aivie-2p6b-p3-snap](https://huggingface.co/donaldluong/nn-aivie-2p6b-p3-snap) | model | P3 snapshots of the 2.6B model (C4 / code-expert variants) |
| [donaldluong/nn-aivie-cx3-ckpt](https://huggingface.co/donaldluong/nn-aivie-cx3-ckpt) | model | cx3 large code expert (pre-training checkpoint + logs) |
| [donaldluong/nn-aivie-pretrain-vi](https://huggingface.co/datasets/donaldluong/nn-aivie-pretrain-vi) | dataset | Vietnamese pre-training corpus (packed) |
| [donaldluong/nn-aivie-cpt-data](https://huggingface.co/datasets/donaldluong/nn-aivie-cpt-data) | dataset | Tokenized continued-pre-training shards (Vietnamese + code) |

Other `donaldluong/nn-aivie-*` repos referenced by the `box_*.sh` scripts are private.

## Public source datasets used

- Vietnamese / multilingual: [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [bkai-foundation-models/vi-alpaca](https://huggingface.co/datasets/bkai-foundation-models/vi-alpaca), [5CD-AI/Viet-Doc-VQA](https://huggingface.co/datasets/5CD-AI/Viet-Doc-VQA), [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), [hungnm/vietnamese-medical-qa](https://huggingface.co/datasets/hungnm/vietnamese-medical-qa)
- Code / math / reasoning: [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2), [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts), [open-thoughts/OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M)
- General English: [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
- Base models used for distillation / comparison: Gemma 4 (Google), Qwen, Zamba2 (Zyphra) — see their own licenses.

Check each dataset's license before use.
