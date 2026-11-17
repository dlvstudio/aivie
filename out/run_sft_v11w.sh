#!/usr/bin/env bash
set -euo pipefail
cd ~/Documents/learn/nn-aivie
source .venv/bin/activate
export PYTHONPATH=src

# SFT trên base v11w (10d: qa_heldout answer-only 4.821, TỐT NHẤT). Test "base mạnh hơn → QA sinh
# khá hơn không" (so SFT-trên-v10 ra rác). ⚠️ replay = config/train_v11_weighted.yaml (corpus
# PRETRAIN của v11w — không phải v10 default) để guard chống-quên đo đúng nền v11w.
python -u scripts/sft_neural_lm.py --ckpt out/lm_v2_v11w_qa.pt --save out/lm_v2_v11w_sft.pt \
  --manifest config/train_v11_weighted.yaml --log-file out/sft_v11w.log

# đọc 3 verdict cuối log: (1) bits-đáp giảm ≥0.5? (2) guard essays Δ (quên?) (3) 12 mẫu đọc tay
