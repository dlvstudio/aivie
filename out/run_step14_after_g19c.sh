#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/learn/nn-aivie
source .venv/bin/activate
export PYTHONPATH=src

# Wait until G19c main training process exits.
while pgrep -f 'train_neural_lm.py --arch v2 --opt muon --batch 16 --manifest config/train_v10.yaml --d-model 512 --layers 8 --heads 8 --ctx 512 --steps 30000 --eval-every 500 --save out/lm_v2_v10_fp32.pt --log-file out/lm_v2_v10_fp32.log' >/dev/null; do
  sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] G19c done. Starting Step 14 SFT..."

if [ ! -f out/lm_v2_v10_fp32.pt ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: missing checkpoint out/lm_v2_v10_fp32.pt"
  exit 1
fi

~/Documents/learn/nn-aivie/.venv/bin/python -u scripts/sft_neural_lm.py \
  --ckpt out/lm_v2_v10_fp32.pt \
  --save out/lm_v2_sft.pt \
  --log-file out/sft_neural_lm.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 14 SFT finished."