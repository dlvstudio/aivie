#!/bin/bash
# Chạy (lại) SFT sau Stop/reboot. SFT không resume giữa chừng → chạy lại từ đầu nếu chưa xong.
# Gọi từ /workspace/onstart.sh. Xong thì có ckpt/sft.done → không chạy nữa.
set -u
cd /workspace/nn-aivie || exit 1
LOG=/workspace/nn-aivie/ensure.log
echo "== ensure_sft $(date -u +%FT%TZ) ==" >> "$LOG"

if [ -f /workspace/nn-aivie/ckpt/sft.done ]; then
  echo "ensure_sft: đã xong" >> "$LOG"; exit 0
fi
if pgrep -f 'scripts/train_hybrid_sft.py' >/dev/null 2>&1; then
  echo "ensure_sft: đang chạy" >> "$LOG"; exit 0
fi

cat > /workspace/nn-aivie/run_sft.sh <<'EOF'
#!/bin/bash
set -euo pipefail
cd /workspace/nn-aivie
export HF_HOME=/workspace/.hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=scripts:src
echo "== sft start $(date -u +%FT%TZ) ==" | tee -a sft.log
python -u scripts/train_hybrid_sft.py \
  --init-from /workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt \
  --data data/sft_vi/sft_mix.jsonl --tokenizer checkpoints/bpe_v1.json \
  --max-len 2048 --bs 8 --grad-accum 4 --lr 2e-5 --min-lr 2e-6 --warmup 100 --epochs 2 \
  --eval-every 200 --save-every 500 \
  --hf-repo donaldluong/nn-aivie-hybrid-1p3b-sft --hf-every 500 --hf-name hybrid_1p3b_sft.pt \
  --out /workspace/nn-aivie/ckpt/hybrid_1p3b_sft.pt \
  2>&1 | tee -a sft.log
touch /workspace/nn-aivie/ckpt/sft.done
EOF
chmod +x /workspace/nn-aivie/run_sft.sh
tmux has-session -t sft 2>/dev/null && tmux kill-session -t sft
tmux new-session -d -s sft /workspace/nn-aivie/run_sft.sh
echo "ensure_sft: start" >> "$LOG"
