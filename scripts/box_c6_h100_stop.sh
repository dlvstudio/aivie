#!/bin/bash
# TK_108 C6 (user 26/10: "chạy h100 nốt chiều nay thôi"): khi bản lưu bước STOP_STEP đã GỬI VỀ A100 (md5 khớp) VÀ đã đẩy HF
# → dừng C6 trên H100 (runner + torchrun + đẩy HF) → báo A100 (box_run_c6_side.sh chờ dòng này để chạy tiếp C6).
# Dùng: (setsid nohup bash scripts/box_c6_h100_stop.sh 13000 > /workspace/c6_stop.log 2>&1 &)
STOP_STEP=${1:-13000}
S=/workspace/p3_status.log; log() { echo "$(date '+%F %T') $*" >> $S; }
until grep -q "đã gửi bản đầy đủ bước $STOP_STEP về A100" $S && grep -q "C6 đã đẩy HF bước $STOP_STEP " $S; do sleep 30; done
for p in $(pgrep -f "scripts/box_run_c6_h100.sh"); do kill $p; done
pkill -f train_hybrid_scale.py; pkill -f box_c6_hf_sync.sh
sleep 20; pkill -9 -f train_hybrid_scale.py
log "C6 trên H100 DỪNG @ bước $STOP_STEP (bản đầy đủ đã ở A100 + HF) — box H100 có thể huỷ"
ssh -n -i /root/.ssh/id_a100 -p 20343 root@<IP> \
  "echo \"$(date '+%F %T') C6 trên H100 DỪNG @ bước $STOP_STEP — A100 nhận ckpt/c6.pt, chạy tiếp\" >> /workspace/p3_status.log"
