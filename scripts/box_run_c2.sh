#!/bin/bash
# TK_108 C2 trên box: thầy Gemma-4-31B (vLLM, cả GPU) sinh dữ liệu code — mọi mẫu kiểm bằng chạy thật (c2_teacher.py).
#   thử nhỏ 40 mẫu mỗi kiểu → chạy đủ: vi 30K · debug 20K · predict 10K · sql 8K → gói + đẩy HF. Trạng thái → /workspace/p3_status.log
# Dùng: (setsid nohup bash scripts/box_run_c2.sh > /workspace/c2_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1
EV=/workspace/evalenv/bin/python                       # không cần torch; sqlite3 + code_agent + gen_office
[ -s data/c2/oci_verified.jsonl ] || { log "LỖI C2: thiếu data/c2/oci_verified.jsonl"; exit 1; }
if ! curl -s localhost:8000/v1/models | grep -q gemma4-31b; then
  CUDA_VISIBLE_DEVICES=0 setsid nohup /workspace/vllm_env/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 128 --port 8000 > /workspace/c2_vllm.log 2>&1 < /dev/null &
  W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI C2: vLLM không lên"; exit 1; }; done
fi
log "C2 bắt đầu: thầy Gemma-4-31B sinh dữ liệu code (vi 30K · debug 20K · predict 10K · sql 8K), kiểm bằng chạy thật"
mkdir -p /workspace/c2_smoke
for k in vi debug predict sql; do                      # thử nhỏ: tỉ lệ giữ + tốc độ, ghi riêng để không lẫn dữ liệu thật
  $EV -u - <<PY > /workspace/c2_smoke_$k.log 2>&1
import sys; sys.argv = ["c2_teacher.py", "$k", "--n", "40", "--workers", "40"]
sys.path.insert(0, "scripts"); import c2_teacher as T
T.OUT = "/workspace/c2_smoke"; T.main()
PY
done
log "C2 thử nhỏ · $(grep -h '###' /workspace/c2_smoke_*.log | sed 's/### c2 //' | sed 's/ →.*//' | tr '\n' '|')"
for kn in vi:30000 debug:20000 predict:10000 sql:8000; do
  k=${kn%%:*}; n=${kn#*:}
  $EV -u scripts/c2_teacher.py $k --n $n --workers 96 > /workspace/c2_$k.log 2>&1 || log "LỖI C2 $k (c2_$k.log)"
  log "C2 $k xong · $(grep '###' /workspace/c2_$k.log | sed 's/### c2 //' | cut -c1-200)"
done
pkill -f "vllm serve"
wc -l data/c2/{vi,debug,predict,sql}.jsonl > /workspace/c2_counts.txt
for f in vi debug predict sql; do
  hf upload donaldluong/nn-aivie-2p6b-p3 data/c2/$f.jsonl data/c2/$f.jsonl > /dev/null 2>&1
done
log "C2 XONG · $(tr '\n' ' ' < /workspace/c2_counts.txt) · đã đẩy HF nn-aivie-2p6b-p3/data/c2/"
