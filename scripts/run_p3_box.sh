#!/bin/bash
# P3 trên box NGC mới: chờ setup_ngc_box.sh xong → chạy box_run_p3.sh (tự chờ `hf auth login`, tự chạy tiếp SFT từ bản sao lưu HF).
#   (setsid nohup bash scripts/setup_ngc_box.sh > /workspace/setup_p3.log 2>&1 &)
#   (setsid nohup bash scripts/run_p3_box.sh > /workspace/run_p3.log 2>&1 &)
S=/workspace/p3_status.log
until grep -q SETUP_XONG /workspace/setup_p3.log 2>/dev/null; do
  pgrep -f setup_ngc_box.sh > /dev/null || { echo "$(date '+%F %T') LỖI P3: setup hỏng (xem setup_p3.log)" >> $S; exit 1; }
  sleep 20
done
echo "$(date '+%F %T') P3 setup xong: $(grep -E 'KERNEL OK|HAVE_KERNEL' /workspace/setup_p3.log | tr '\n' ' ')" >> $S
cd /workspace/nn-aivie && PY=/usr/bin/python bash scripts/box_run_p3.sh
