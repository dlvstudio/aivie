#!/bin/bash
# TK_108 C6 chuyển sang box 4×H100 SXM (user thuê 26/10, $8,55/h, tối đa ~44 giờ): chạy TIẾP từ ckpt/c6.pt của box A100.
# Giữ nguyên lịch: 4 GPU × bs 2 × accum 4 × ctx 8192 = 262 144 token/bước (như 2×A100 × accum 8) → cùng 38 146 bước, cùng WSD.
# Mỗi lần lưu (500 bước): box_c6_hf_sync.sh đẩy bf16 lên HF + vòng dưới đây gửi BẢN ĐẦY ĐỦ (có optimizer) về box A100
# (ckpt/c6.pt), để khi box H100 hết hạn thì A100 chạy tiếp đúng bước bằng box_run_c6.sh.
# Dùng: (setsid nohup bash scripts/box_run_c6_h100.sh > /workspace/c6_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A="ssh -i /root/.ssh/id_a100 -p 20343"; AH=root@<IP>
until grep -q "H100: SETUP_XONG" $S && grep -q "H100: dữ liệu xong" $S && grep -q "H100: c6.pt xong" $S; do
  grep -q "LỖI H100" $S && exit 1; sleep 30; done
# ── dừng C6 trên A100 (nó chạy tới đâu thì bỏ, H100 tiếp từ bản đã chép) + đẩy HF từ H100 ──
scp -q -i /root/.ssh/id_a100 -P 20343 /root/stop_c6_a100.sh $AH:/root/ && $A $AH "bash /root/stop_c6_a100.sh" && log "C6: đã dừng bản chạy trên A100"
scp -q -i /root/.ssh/id_a100 -P 20343 $AH:/workspace/c6_train.log /workspace/c6_train.log      # giữ nhật ký liền mạch
(setsid nohup bash scripts/box_c6_hf_sync.sh > /workspace/c6_sync.log 2>&1 &)
# ── gửi bản đầy đủ về A100 sau mỗi lần lưu ──
( LAST=$(stat -c %Y ckpt/c6.pt)
  while true; do
    M=$(stat -c %Y ckpt/c6.pt 2>/dev/null)
    if [ -n "$M" ] && [ "$M" != "$LAST" ]; then
      sleep 150                                              # chờ torch.save ghi xong
      STEP=$(python -c "import torch;print(torch.load('ckpt/c6.pt',map_location='cpu',weights_only=False,mmap=True)['step'])" 2>/dev/null)
      SZ=$(stat -c %s ckpt/c6.pt); MB=$(( (SZ + 1048575) / 1048576 )); CH=$(( (MB + 15) / 16 )); OK=1
      $A $AH "rm -f /workspace/nn-aivie/ckpt/c6_from_h100.pt; truncate -s $SZ /workspace/nn-aivie/ckpt/c6_from_h100.pt"
      for i in $(seq 0 15); do                               # 1 luồng ssh Nhật→Ba Lan chậm → 16 đoạn song song
        dd if=ckpt/c6.pt bs=1M skip=$((i*CH)) count=$CH 2>/dev/null | \
          $A $AH "dd of=/workspace/nn-aivie/ckpt/c6_from_h100.pt bs=1M seek=$((i*CH)) conv=notrunc iflag=fullblock 2>/dev/null" &
      done; wait
      L=$(md5sum ckpt/c6.pt | cut -c1-32); R=$($A $AH "md5sum /workspace/nn-aivie/ckpt/c6_from_h100.pt | cut -c1-32")
      if [ "$L" = "$R" ] && [ "$(stat -c %Y ckpt/c6.pt)" = "$M" ] && \
         $A $AH "mv /workspace/nn-aivie/ckpt/c6_from_h100.pt /workspace/nn-aivie/ckpt/c6.pt"; then
        log "C6 H100: đã gửi bản đầy đủ bước $STEP về A100 (md5 khớp)"; LAST=$M
      else log "LỖI C6 H100: gửi ckpt về A100 bước $STEP — thử lại"; sleep 300; fi
    fi
    grep -q " C6 học thân XONG" $S && exit 0
    sleep 60
  done ) &
MIX="data/cx3/gh:0.12,data/cx3/opc_algo:0.09,data/cx3/opc_snip:0.06,data/cx3/opc_files:0.05,data/cpt/code:0.13"
MIX="$MIX,data/cx3/web:0.03,data/cpt/en_edu:0.09"
MIX="$MIX,data/cpt/vi_web:0.18,data/cpt/long:0.07,data/cpt/vi_wiki:0.02,data/cx3/vi:0.03"
MIX="$MIX,data/cpt/math:0.08"
ANNEAL="data/cx3/opc_algo:0.22,data/cx3/opc_files:0.15,data/cx3/gh:0.10,data/cpt/math:0.13,data/cpt/vi_wiki:0.10,data/cpt/long:0.15,data/cpt/vi_web:0.10,data/cpt/en_edu:0.05"
for try in 1 2 3 4 5 6; do
  log "C6 học thân trên 4×H100 (lần $try) · tiếp từ ckpt/c6.pt"
  torchrun --standalone --nproc_per_node 4 scripts/train_hybrid_scale.py --data "$MIX" --anneal-data "$ANNEAL" \
     --ctx 8192 --bs 2 --grad-accum 4 --lr 1.5e-4 --min-lr 1.5e-5 --warmup 1000 --tokens 10e9 --sched wsd --decay-frac 0.1 \
     --grad-ckpt --eval-every 100 --save-every 500 --init-from ckpt/c6.pt --out ckpt/c6.pt --hf-repo "" >> /workspace/c6_train.log 2>&1 && break
  log "LỖI C6 H100 lần $try (c6_train.log: $(grep -E 'Error|error' /workspace/c6_train.log | tail -1 | cut -c1-150)) — chạy tiếp sau 2 phút"
  sleep 120
done
grep -q " XONG · LƯU ckpt/c6.pt" /workspace/c6_train.log || { log "LỖI C6: học thân không xong sau 6 lần (H100)"; exit 1; }
log "C6 học thân XONG · $(grep ' XONG · LƯU' /workspace/c6_train.log | tail -1 | cut -c1-200)"
