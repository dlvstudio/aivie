# Dữ liệu

Repo này **không chứa dữ liệu train/eval**. Lý do: dữ liệu crawl có thể chứa thông tin cá nhân, một số bộ có giấy phép không cho phân phối lại, và dữ liệu sinh từ mô hình thầy không công khai. Các script trong `scripts/` tải và dựng lại dữ liệu từ nguồn gốc.

## Nguồn công khai đã dùng (Hugging Face)
- Tiếng Việt / đa ngữ: [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [bkai-foundation-models/vi-alpaca](https://huggingface.co/datasets/bkai-foundation-models/vi-alpaca), [5CD-AI/Viet-Doc-VQA](https://huggingface.co/datasets/5CD-AI/Viet-Doc-VQA), [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), [hungnm/vietnamese-medical-qa](https://huggingface.co/datasets/hungnm/vietnamese-medical-qa)
- Code / toán / suy luận: [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2), [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts), [open-thoughts/OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M)
- Tiếng Anh tổng quát: [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)

Xem giấy phép của từng bộ trước khi dùng.

## Repo Hugging Face của dự án
Checkpoint và dữ liệu đã xử lý nằm trong các repo `donaldluong/nn-aivie-*` trên Hugging Face (phần lớn là private; tên repo xuất hiện trong các script `box_*.sh`).
