#!/bin/bash
# TK_108 C6 trên H100: sau mỗi lần lưu, gửi BẢN ĐẦY ĐỦ ckpt/c6.pt về A100 (để A100 chạy tiếp đúng bước khi H100 dừng/hết hạn).
# 8 đoạn song song, khởi động lệch 2 giây (sshd A100 chặn >10 kết nối mở cùng lúc — 16 luồng bị "Connection reset"),
# mỗi đoạn thử lại tối đa 5 lần · kiểm md5 · đổi tên nguyên tử bên A100.
S=/workspace/p3_status.log; log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
A="ssh -i /root/.ssh/id_a100 -p 20343 root@<IP>"; T=/workspace/nn-aivie/ckpt/c6_from_h100.pt
LAST=${1:-0}
while true; do
  M=$(stat -c %Y ckpt/c6.pt 2>/dev/null)
  if [ -n "$M" ] && [ "$M" != "$LAST" ]; then
    sleep 150; M=$(stat -c %Y ckpt/c6.pt)                    # mtime SAU khi ghi xong (bắt gặp lúc đang ghi → M cũ → luôn báo lệch)
    STEP=$(python -c "import torch;print(torch.load('ckpt/c6.pt',map_location='cpu',weights_only=False,mmap=True)['step'])" 2>/dev/null)
    SZ=$(stat -c %s ckpt/c6.pt); MB=$(( (SZ + 1048575) / 1048576 )); CH=$(( (MB + 7) / 8 ))
    $A -n "rm -f $T; truncate -s $SZ $T"
    for i in $(seq 0 7); do
      ( for t in 1 2 3 4 5; do
          dd if=ckpt/c6.pt bs=1M skip=$((i*CH)) count=$CH 2>/dev/null | \
            $A "dd of=$T bs=1M seek=$((i*CH)) conv=notrunc iflag=fullblock 2>/dev/null" && exit 0
          sleep $((t*5)); done; exit 1 ) &
      sleep 2
    done; wait
    L=$(md5sum ckpt/c6.pt | cut -c1-32); R=$($A -n "md5sum $T | cut -c1-32")
    if [ "$L" = "$R" ] && [ "$(stat -c %Y ckpt/c6.pt)" = "$M" ] && $A -n "mv $T /workspace/nn-aivie/ckpt/c6.pt"; then
      log "C6 H100: đã gửi bản đầy đủ bước $STEP về A100 (md5 khớp)"; LAST=$M
    else log "LỖI C6 H100: gửi ckpt về A100 bước $STEP (md5 $L vs $R) — thử lại"; sleep 120; fi
  fi
  grep -qE " C6 học thân XONG|C6 trên H100 DỪNG" $S && exit 0
  sleep 60
done
