#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/learn/nn-aivie
source .venv/bin/activate
export PYTHONPATH=src

echo "[$(date)] Starting G19 arm 1"
python -u scripts/train_neural_lm.py --arch v2 --opt muon \
  --d-model 512 --layers 8 --heads 8 --ctx 512 --batch 16 \
  --steps 10000 --eval-every 500 --save out/lm_v2_42m.pt --log-file out/lm_v2_42m.log

echo "[$(date)] Starting G19 arm 2"
python -u scripts/train_neural_lm.py --arch v2 --opt muon --mem-slots 16384 \
  --d-model 512 --layers 8 --heads 8 --ctx 512 --batch 16 \
  --steps 10000 --eval-every 500 --save out/lm_v2_mem.pt --log-file out/lm_v2_mem.log

echo "[$(date)] Training complete, committing logs"
git add -f out/lm_v2_42m.log out/lm_v2_mem.log
git commit --no-gpg-sign -m "TK_042: G19 — train v2 42M + mem arm"
git push origin main

echo "[$(date)] G19 complete"