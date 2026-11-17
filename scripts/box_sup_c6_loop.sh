#!/bin/bash
# TK_108: giám sát box_run_c6_loop.sh — script từng chết lặng lẽ lúc chờ UI (05/11, chưa rõ ai gửi tín hiệu) → chạy lại
# tối đa 6 lần; các bước của nó chạy lại được (đánh dấu .done, không nhân đôi tiến trình đang sống).
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
for i in 1 2 3 4 5 6; do
  bash scripts/box_run_c6_loop.sh >> /workspace/c6_loop_run.log 2>&1 < /dev/null
  rc=$?
  grep -qaE '^[0-9-]+ [0-9:]+ vòng thầy c6l1 XONG' $S && exit 0
  log "vòng thầy c6l1: script điều phối dừng (mã $rc) — khởi động lại lần $i/6"
  sleep 30
done
log "LỖI vòng thầy c6l1: điều phối dừng 6 lần — cần xem tay (c6_loop_trace.log)"
