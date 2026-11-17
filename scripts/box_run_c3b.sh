#!/bin/bash
# C3 phần 2 (sau khi tách khỏi box_run_c3.sh): chờ dựng bài xong + C4 sớm đo xong (GPU rảnh) → thử nhỏ chọn thầy → sinh quỹ đạo
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 AIVIE_SANDBOX_USER=sbx
EV=/workspace/evalenv/bin/python; VLLM=/workspace/vllm_env/bin/vllm
QWEN=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
while pgrep -f "c3_teacher.py build" > /dev/null; do sleep 60; done
log "C3 dựng bài · $(wc -l < data/c3/tasks.jsonl) bài"
until grep -q "C4 sớm đo XONG\|LỖI C4 sớm đo" $S; do sleep 60; done
vllm_on() {
  pkill -f "vllm serve"; sleep 20
  VLLM_USE_DEEP_GEMM=0 CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve $1 --served-model-name $2 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --max-num-seqs 128 --port 8000 > /workspace/c3_vllm_$2.log 2>&1 < /dev/null &
  W=0; until curl -s localhost:8000/v1/models | grep -q "$2"; do sleep 15; W=$((W+1)); [ $W -gt 160 ] && return 1; done
  return 0      # until trả mã của lệnh cuối trong thân vòng ([ … ] sai = 1) → phải trả 0 rõ ràng
}
score() { grep '###' /workspace/c3_pilot_$1.log 2>/dev/null | sed -n 's/.*thành công \([0-9]*\).*/\1/p'; }
vllm_on RedHatAI/gemma-4-31B-it-FP8-dynamic gemma4-31b || { log "LỖI C3: vLLM Gemma"; exit 1; }
$EV -u scripts/c3_teacher.py run --model gemma4-31b --tag pilot_gemma --n 50 --offset 0 --workers 50 > /workspace/c3_pilot_gemma.log 2>&1
G=$(score gemma); Q=""
if vllm_on $QWEN qwen3-coder; then
  $EV -u scripts/c3_teacher.py run --model qwen3-coder --tag pilot_qwen --n 50 --offset 0 --workers 50 > /workspace/c3_pilot_qwen.log 2>&1
  Q=$(score qwen)
else
  log "C3: Qwen không lên được (c3_vllm_qwen3-coder.log) → dùng Gemma"
fi
G=${G:-0}; Q=${Q:-0}
if [ "$Q" -gt "$G" ]; then WIN=qwen3-coder; REPO=$QWEN; else WIN=gemma4-31b; REPO=RedHatAI/gemma-4-31B-it-FP8-dynamic; fi
log "C3 thử nhỏ 50 bài · Gemma thành công $G · Qwen3-Coder $Q → thầy: $WIN"
curl -s localhost:8000/v1/models | grep -q "$WIN" || vllm_on $REPO $WIN || { log "LỖI C3: vLLM $WIN"; exit 1; }
END=$(date -u -d "tomorrow 00:30" +%s); [ $(date -u +%H) -lt 12 ] && END=$(date -u -d "today 00:30" +%s)
H=$(python3 -c "print(max(1.0, ($END - $(date +%s)) / 3600 - 0.3))")
[ -n "$C3_HOURS" ] && H=$C3_HOURS                     # chạy ban ngày: đặt thẳng số giờ
log "C3 sinh quỹ đạo: thầy $WIN · ngân sách ${H} giờ"
$EV -u scripts/c3_teacher.py run --model $WIN --tag main --offset 100 --workers 64 --hours $H > /workspace/c3_main.log 2>&1 \
  || log "LỖI C3 sinh quỹ đạo (c3_main.log)"
pkill -f "vllm serve"
for f in tasks.jsonl traj_pilot_gemma.jsonl traj_pilot_qwen.jsonl traj_main.jsonl; do
  [ -s data/c3/$f ] && hf upload donaldluong/nn-aivie-2p6b-p3 data/c3/$f data/c3/$f > /dev/null 2>&1
done
log "C3 XONG · $(grep '###' /workspace/c3_main.log | sed 's/### c3 //') · đã đẩy HF nn-aivie-2p6b-p3/data/c3/"
