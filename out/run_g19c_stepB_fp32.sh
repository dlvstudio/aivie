#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/learn/nn-aivie
source .venv/bin/activate
export PYTHONPATH=src

# G19c fp32 b16 · 30000 step = 245M token (khớp budget G7) — so mốc 5.594 apples-to-apples.
# ~8.6h ở 1.03 s/step (đo smoke b16); fp32 vì bf16/fp16 = 0 lợi trên MPS, batch>16 swap.
python -u scripts/train_neural_lm.py --arch v2 --opt muon --batch 16 \
  --manifest config/train_v10.yaml --d-model 512 --layers 8 --heads 8 --ctx 512 \
  --steps 30000 --eval-every 500 --save out/lm_v2_v10_fp32.pt --log-file out/lm_v2_v10_fp32.log