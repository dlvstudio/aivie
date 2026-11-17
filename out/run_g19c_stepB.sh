#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/learn/nn-aivie
source .venv/bin/activate
export PYTHONPATH=src

python -u scripts/train_neural_lm.py --arch v2 --opt muon --bf16 --batch 48 \
  --manifest config/train_v10.yaml --d-model 512 --layers 8 --heads 8 --ctx 512 \
  --steps 15000 --eval-every 500 --save out/lm_v2_v10.pt --log-file out/lm_v2_v10.log