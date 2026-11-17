#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/learn/nn-aivie
source .venv/bin/activate
export PYTHONPATH=src

echo "[$(date)] [10c] install datasets"
.venv/bin/python -m pip install -U datasets

echo "[$(date)] [10c] ingest viwiki"
.venv/bin/python -u scripts/ingest_viwiki.py --max-tokens 160000000

echo "[$(date)] [10c] train v2 on v11 corpus (30k)"
.venv/bin/python -u scripts/train_neural_lm.py --arch v2 --opt muon --manifest config/train_v11.yaml \
  --d-model 512 --layers 8 --heads 8 --ctx 512 --batch 16 \
  --steps 30000 --eval-every 1000 --save out/lm_v2_v11.pt --log-file out/lm_v2_v11.log

echo "[$(date)] [10c] done"