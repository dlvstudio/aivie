#!/usr/bin/env bash
# Post-v12: probes on network_v12_sent.pkl (run under nohup+caffeinate)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=".venv/bin/python"
LOG=out/post_v12_pipeline.log
CKPT=checkpoints/network_v12_sent.pkl

echo "===== post-v12 probes $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$LOG"

run() {
  echo "" | tee -a "$LOG"
  echo ">>> $*" | tee -a "$LOG"
  "$PY" -u "$@"
  echo "exit=$?" | tee -a "$LOG"
}

run scripts/probe_sentence_parents.py --checkpoint "$CKPT" --log-file out/probe_sentence_parents_v12.log
run scripts/probe_multi_prompt_sent.py --checkpoint "$CKPT" --log-file out/probe_multi_prompt_v12.log --max-len 36
run scripts/probe_sent_intra_ablation.py --checkpoint "$CKPT" --log-file out/probe_sent_intra_ablation_v12.log
run scripts/probe_sharp_h.py --checkpoint "$CKPT" --log-file out/probe_sharp_h_v12.log

echo "ALL PROBES DONE $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
