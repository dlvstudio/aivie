#!/bin/bash
# Resume train + HF watcher sau Stop/reboot. Destroy vẫn mất ổ → sống nhờ HF.
# Gọi từ /workspace/onstart.sh
set -u
cd /workspace/nn-aivie || exit 1
export HF_HOME=/workspace/.hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/workspace/nn-aivie/scripts
LOG=/workspace/nn-aivie/ensure.log
echo "== ensure $(date -u +%FT%TZ) ==" >> "$LOG"

INIT=/workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt
if [ ! -f "$INIT" ]; then
  INIT=/workspace/nn-aivie/ckpt/hybrid_1p3b.pt
fi

if ! pgrep -f 'scripts/train_hybrid_scale.py' >/dev/null 2>&1; then
  echo "ensure: start train from $INIT" >> "$LOG"
  cat > /workspace/nn-aivie/run_train.sh <<EOF
#!/bin/bash
set -euo pipefail
cd /workspace/nn-aivie
export HF_HOME=/workspace/.hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=scripts
echo "== train start \$(date -u +%FT%TZ) init=$INIT ==" | tee -a /workspace/nn-aivie/train.log
python -u scripts/train_hybrid_scale.py \\
  --data /workspace/nn-aivie/pretrain_vi \\
  --ctx 2048 --d 2048 --layers 48 --attn-every 8 --d-state 16 --heads 16 \\
  --bs 8 --grad-accum 8 --lr 3e-4 --min-lr 3e-5 --warmup 200 \\
  --tokens 18800060992 --eval-every 500 --save-every 1000 \\
  --hf-repo donaldluong/nn-aivie-hybrid-1p3b-ckpt --hf-every 1000 \\
  --init-from $INIT \\
  --out /workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt \\
  2>&1 | tee -a /workspace/nn-aivie/train.log
EOF
  chmod +x /workspace/nn-aivie/run_train.sh
  tmux has-session -t train 2>/dev/null && tmux kill-session -t train
  tmux new-session -d -s train /workspace/nn-aivie/run_train.sh
else
  echo "ensure: train already running" >> "$LOG"
fi

if ! pgrep -f hf_backup_watch.py >/dev/null 2>&1; then
  echo "ensure: start hf_backup_watch every=1000" >> "$LOG"
  tmux has-session -t hfwatch 2>/dev/null && tmux kill-session -t hfwatch
  tmux new-session -d -s hfwatch "export HF_HOME=/workspace/.hf HF_HUB_DISABLE_XET=1 PYTHONPATH=/workspace/nn-aivie/scripts; python -u /workspace/nn-aivie/scripts/hf_backup_watch.py --every 1000 --log /workspace/nn-aivie/train.log --path /workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt 2>&1 | tee -a /workspace/nn-aivie/hf_watch.log"
else
  echo "ensure: hfwatch already running" >> "$LOG"
fi
