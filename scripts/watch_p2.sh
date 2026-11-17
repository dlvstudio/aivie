#!/bin/bash
# Canh box P2 từ Mac (TK_107): báo khi có dòng status MỚI chứa LỖI/xong pha, GPU rảnh 2 lần liền (treo/chết),
# sao lưu HF lỗi, hoặc mất kết nối 30 phút. Kiểm mỗi 10 phút. Dùng: bash scripts/watch_p2.sh [ssh-port] [host]
P=${1:-27322}; H=${2:-root@<IP>}
b() { ssh -o ConnectTimeout=20 -i $HOME/.ssh/fpt_h100 -p $P $H "$@" 2>/dev/null | grep -v -E "Welcome|Have fun"; }
N=$(b 'wc -l < /workspace/cpt_status.log' | tail -1); IDLE=0; FAIL=0
while true; do
  R=$(b "tail -n +$((N+1)) /workspace/cpt_status.log | grep -E 'LỖI|xong|XONG'")
  [ -n "$R" ] && { echo "$R"; exit 0; }
  U=$(b "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1")
  if [ -z "$U" ]; then FAIL=$((FAIL+1)); [ $FAIL -ge 3 ] && { echo "MẤT KẾT NỐI box 30 phút"; exit 0; }
  else FAIL=0; [ "$U" -lt 10 ] && IDLE=$((IDLE+1)) || IDLE=0; fi
  [ $IDLE -ge 2 ] && { echo "GPU RẢNH 2 lần liền"; b 'tail -3 /workspace/cpt_status.log; tail -c 1500 $(ls -t /workspace/cpt_a*.log /workspace/cpt_b.log 2>/dev/null | head -1)'; exit 0; }
  E=$(b "grep -h -i -E 'HF backup.*(lỗi|fail|error)|limit reached' /workspace/cpt_b.log 2>/dev/null | tail -1")
  [ -n "$E" ] && { echo "SAO LƯU LỖI: $E"; exit 0; }
  sleep 600
done
