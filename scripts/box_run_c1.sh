#!/bin/bash
# TK_108 C1: đo code đầy đủ trên checkpoint CUỐI của P3 — tự chờ "P3.3' XONG" rồi mới chạy. Trạng thái → /workspace/p3_status.log
#   HumanEval 164 · MBPP 257 · tool-use 40 — 3 máy chủ UI chạy song song trên GPU (mỗi bộ một cổng).
#   Script đo nằm riêng ở /workspace/c1/scripts (không đè bản đang dùng của vòng thầy); các module khác lấy từ repo box.
# Dùng: (setsid nohup bash /workspace/c1/scripts/box_run_c1.sh > /workspace/c1_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done   # serve_hybrid_ui chdir về thư mục cha của scripts
cd /workspace/nn-aivie
until grep -q "P3.3' XONG" $S; do
  grep -q "LỖI P3.3'" $S && { log "C1 không chạy: P3.3' báo lỗi"; exit 1; }
  sleep 120
done
CK=ckpt/aivie_p3_r1_bf16.pt
[ -s $CK ] || { log "LỖI C1: thiếu $CK"; exit 1; }
PY=/workspace/vlmenv/bin/python                                     # torch + transformers (đọc ảnh nằm trong checkpoint)
if [ ! -x /workspace/evalenv/bin/python ]; then                     # môi trường riêng cho bộ đo — không đụng vlmenv
  python3 -m venv /workspace/evalenv && /workspace/evalenv/bin/pip -q install datasets > /workspace/c1_pip.log 2>&1 \
    || { log "LỖI C1: cài datasets"; exit 1; }
fi
EV=/workspace/evalenv/bin/python
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
log "C1 bắt đầu: đo code đầy đủ trên $CK (HumanEval 164 · MBPP 257 · tool-use 40, song song)"
PIDS=""; port=8811
for s in humaneval mbpp agent; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $port --no-rag > /workspace/c1_ui_$s.log 2>&1 &
  PIDS="$PIDS $!"; port=$((port+1))
done
for port in 8811 8812 8813; do
  W=0; until curl -s localhost:$port > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI C1: UI cổng $port không lên"; kill $PIDS; exit 1; }; done
done
port=8811; JOBS=""
for s in humaneval mbpp agent; do
  ( $EV -u /workspace/c1/scripts/code_eval.py run --suite $s --url http://localhost:$port --tag p3 > /workspace/c1_$s.log 2>&1 \
      && log "C1 $s xong · $(grep '###' /workspace/c1_$s.log | tail -1 | cut -c1-300)" || log "LỖI C1 $s (c1_$s.log)" ) &
  JOBS="$JOBS $!"; port=$((port+1))
done
wait $JOBS
kill $PIDS
tar -czf /workspace/c1_results.tgz out/code_eval/*_p3.jsonl /workspace/c1_*.log 2>/dev/null
hf upload donaldluong/nn-aivie-2p6b-p3 /workspace/c1_results.tgz results/c1_results.tgz > /dev/null 2>&1
log "C1 XONG — kết quả: /workspace/c1_results.tgz (HF nn-aivie-2p6b-p3/results/c1_results.tgz)"
