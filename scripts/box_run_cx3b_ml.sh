#!/bin/bash
# TK_108: MultiPL-E cho cx3b (box_run_cx3b.sh không đo) — 4 máy chủ = 2 / GPU, ctx 4096, 7 ngôn ngữ × 30 bài → HF.
# Dùng: (setsid nohup bash scripts/box_run_cx3b_ml.sh > /workspace/cx3b_ml_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie AIVIE_SANDBOX_USER=sbx PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1
EV=/workspace/evalenv/bin/python; CK=ckpt/aivie_p3_cx3b_bf16.pt
PIDS=""
for p in 8881 8882 8883 8884; do
  CUDA_VISIBLE_DEVICES=$(( (p - 8881) / 2 )) python -W ignore -u scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag --max-ctx 4096 > /workspace/cx3b_mlui_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in 8881 8882 8883 8884; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI cx3b ML: UI $p"; kill $PIDS; exit 1; }; done
done
log "cx3b MultiPL-E bắt đầu (4 máy chủ)"
ml() { $EV -u scripts/code_eval_ml.py run --url http://localhost:$2 --tag $1 --langs $3 --n 30 --workers 1 > /workspace/cx3b_ml_$1.log 2>&1; }
MJ=""
for x in "cx3b-1 8881 js,ts" "cx3b-2 8882 java,cs" "cx3b-3 8883 php,go" "cx3b-4 8884 sh"; do ( ml $x ) & MJ="$MJ $!"; done
wait $MJ
log "cx3b MultiPL-E (cx3 6,7 · cx2 ~5,7) · $($EV scripts/code_eval_ml.py report --tags cx3b-1+cx3b-2+cx3b-3+cx3b-4 | cut -c1-220)"
kill $PIDS
tar czf /workspace/cx3b_ml.tgz out/code_eval/*cx3b-* /workspace/cx3b_ml_*.log 2>/dev/null
hf upload donaldluong/nn-aivie-2p6b-p3 /workspace/cx3b_ml.tgz backup/cx3/cx3b_ml.tgz > /dev/null 2>&1
log "cx3b ML XONG"
