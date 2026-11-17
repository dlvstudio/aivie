#!/bin/bash
# đo lại C4 sớm (lần đầu hỏng: pgrep "\|" là ký tự thường → máy chủ bị tắt sau 30 giây). Trạng thái → /workspace/p3_status.log
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
CK=ckpt/aivie_p3_c4e_bf16.pt
log "C4 sớm đo lại (sửa lỗi tắt máy chủ sớm)"
rm -f out/code_eval/*_c4e.jsonl
PIDS=""; port=8831
for s in humaneval mbpp agent office; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $port --no-rag > /workspace/c4e_ui_$s.log 2>&1 &
  PIDS="$PIDS $!"; port=$((port+1))
done
for port in 8831 8832 8833 8834; do
  W=0; until curl -s localhost:$port > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI C4 sớm đo: UI $port"; kill $PIDS; exit 1; }; done
done
port=8831; JOBS=""
for s in humaneval mbpp agent; do
  ( $EV -u /workspace/c1/scripts/code_eval.py run --suite $s --url http://localhost:$port --tag c4e > /workspace/c4e_$s.log 2>&1 \
      && log "C4 sớm $s · $(grep '###' /workspace/c4e_$s.log | tail -1 | cut -c1-200)" || log "LỖI C4 sớm $s" ) &
  JOBS="$JOBS $!"; port=$((port+1))
done
( $EV -u scripts/office_probe_light.py http://localhost:8834 c4e > /workspace/c4e_office.log 2>&1 \
    && log "C4 sớm văn phòng · $(grep '###' /workspace/c4e_office.log)" || log "LỖI C4 sớm văn phòng" ) &
JOBS="$JOBS $!"
wait $JOBS
kill $PIDS
log "C4 sớm đo XONG — so với mốc p3r1: HumanEval 6,1% · MBPP 5,1% · tool-use 0% · Excel (p3r1xl) 13/27 · easy20 16/20"
