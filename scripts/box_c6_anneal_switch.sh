#!/bin/bash
# TK_108 C6 (28/10): đổi trộn anneal (tiếng Việt 35%→45%) — trainer chỉ đọc --anneal-data lúc khởi động → phải chạy lại.
# Chờ đúng lúc vừa LƯU ckpt/c6.pt @step $1 (torch.save xong rồi mới in dòng LƯU) → dừng runner cũ + torchrun → chạy runner mới
# (tự --init-from ckpt/c6.pt, tiếp đúng bước + optimizer). Mất ~0 bước, chỉ vài phút nạp lại.
# Dùng: (setsid nohup bash scripts/box_c6_anneal_switch.sh 21000 > /workspace/c6_switch.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q "LƯU ckpt/c6.pt @step $1\$" /workspace/c6_train.log; do sleep 20; done
pkill -f "bash scripts/box_run_c6.sh"; sleep 2
pkill -f "torchrun.*train_hybrid_scale"; pkill -f "train_hybrid_scale.py"
for i in $(seq 60); do pgrep -f train_hybrid_scale.py >/dev/null || break; sleep 5; done
pkill -9 -f train_hybrid_scale.py; sleep 10
log "C6: dừng ở bước $1 (vừa lưu) để đổi trộn anneal → tiếng Việt 45% · chạy lại runner"
setsid nohup bash scripts/box_run_c6.sh > /workspace/c6_run.log 2>&1 < /dev/null &
