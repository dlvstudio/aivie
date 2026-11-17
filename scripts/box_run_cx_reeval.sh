#!/bin/bash
# TK_108 ĐO LẠI pilot cx + cx2 (18/10): 4 máy chủ × expert lớn = 81/80 GB → câu dài (tool-use ~1.400 token) tràn GPU,
# máy chủ nuốt lỗi trả rỗng → tool-use cx 10% / cx2 5% là SỐ HỎNG (lượt "không gọi công cụ" = chuỗi rỗng).
# Đo lại bằng 2 máy chủ / lần (~40 GB, dư chỗ cho câu dài): tool-use khung v3 + khung cũ, HumanEval, MBPP. Tag hậu tố "r".
# Dùng: (setsid nohup bash scripts/box_run_cx_reeval.sh > /workspace/cx_reeval.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
cp scripts/code_agent.py scripts/serve_hybrid_ui.py scripts/expert_ffn_1p3b.py /workspace/c1/scripts/
CE=/workspace/c1/scripts/code_eval.py
log "Đo lại cx2 + cx (số tool-use cũ hỏng do tràn GPU 4 máy chủ) · 2 máy chủ / lần"
for M in cx2 cx; do
  CK=ckpt/aivie_p3_${M}_bf16.pt; T=${M}r
  PIDS=""
  for p in 8895 8896; do
    $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag > /workspace/${T}_ui_$p.log 2>&1 &
    PIDS="$PIDS $!"
  done
  for p in 8895 8896; do
    W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI đo lại $M: UI $p"; kill $PIDS; exit 1; }; done
  done
  ( $EV -u $CE run --suite agent --url http://localhost:8895 --tag $T > /workspace/${T}_agent.log 2>&1 \
      && log "$T agent (khung v3) · $(grep '###' /workspace/${T}_agent.log | tail -1 | cut -c1-200)" || log "LỖI $T agent"
    $EV -u $CE run --suite humaneval --url http://localhost:8895 --tag $T > /workspace/${T}_humaneval.log 2>&1 \
      && log "$T humaneval · $(grep '###' /workspace/${T}_humaneval.log | tail -1 | cut -c1-200)" || log "LỖI $T humaneval" ) &
  J1=$!
  ( AIVIE_TOOL_GUARD=0 $EV -u $CE run --suite agent --url http://localhost:8896 --tag ${T}_khungcu > /workspace/${T}_agent0.log 2>&1 \
      && log "$T agent (khung cũ; c4c 35) · $(grep '###' /workspace/${T}_agent0.log | tail -1 | cut -c1-200)" || log "LỖI $T agent khung cũ"
    $EV -u $CE run --suite mbpp --url http://localhost:8896 --tag $T > /workspace/${T}_mbpp.log 2>&1 \
      && log "$T mbpp · $(grep '###' /workspace/${T}_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI $T mbpp" ) &
  J2=$!
  wait $J1 $J2
  log "$T lỗi máy chủ: $(cat /workspace/${T}_ui_*.log | grep -c '\[lỗi /api/chat\]') lần"
  kill $PIDS; sleep 15
done
log "Đo lại cx2 + cx XONG"
