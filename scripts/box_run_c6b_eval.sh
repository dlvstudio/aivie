#!/bin/bash
# TK_108 (user duyệt 04/11): ĐO SFT-B NHANH trên CẢ 2 GPU — thay bước `ev b 1` của box_run_c6_sft.sh (2 máy chủ, ~8h).
#   Trước khi chạy: dừng bash điều phối box_run_c6_sft.sh (chỉ bash, trainer B chạy tiếp, mồ côi).
#   0  chờ trainer B kết thúc · không có "XONG SFT" → chạy lại tiếp từ ckpt (--resume, như sft() cũ) · ghi " SFT-B XONG ·"
#   1  6 máy chủ UI (3/GPU), mỗi máy chủ một làn việc — làn GPU 0 chờ phần đọc ảnh c6a xong hẳn:
#        GPU 1: tool-use · HumanEval · MBPP        GPU 0: C7 1/3 + văn phòng · C7 2/3 · C7 3/3
#   2  dòng "ĐO c6sftb · …" cùng định dạng ev() + dòng tổng kết A ‖ B
# Dùng: (setsid nohup bash scripts/box_run_c6b_eval.sh > /workspace/c6b_eval_run.log 2>&1 < /dev/null &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
unset PIP_CONSTRAINT
EV=/workspace/evalenv/bin/python; CE=scripts/code_eval.py
CK=ckpt/c6_sft_b.pt L=/workspace/c6sft_b.log TAG=c6sftb D=data/sft_c6/sft_b.jsonl

# ── 0) chờ B học xong ──
log "đo nhanh c6sftb: chờ SFT-B học xong rồi đo trên 6 máy chủ / 2 GPU"
while pgrep -f "train_hybrid_sft.py.*c6_sft_b.pt" > /dev/null; do sleep 60; done
for opt in adamw adamw adamw8bit; do
  grep -q "XONG SFT" $L && break
  log "LỖI SFT-B: trainer dừng khi chưa xong ($(grep -E 'Error' $L | tail -1 | cut -c1-150)) — chạy lại ($opt) tiếp từ $CK"
  sleep 60
  [ $opt = adamw8bit ] && { python -c "import bitsandbytes" 2>/dev/null || python -m pip install -q bitsandbytes >> $L 2>&1; }
  CUDA_VISIBLE_DEVICES=1 python -u scripts/train_hybrid_sft.py --resume --init-from ckpt/c6.pt --data $D \
    --max-len 8192 --tok-budget 16384 --bs 8 --grad-accum 4 --grad-ckpt --optim $opt \
    --lr 1e-5 --min-lr 1e-6 --warmup 50 --epochs 2 --eval-every 200 --save-every 500 --out $CK >> $L 2>&1
done
grep -q "XONG SFT" $L || { log "LỖI sau-C6: SFT-B không xong"; exit 1; }
log " SFT-B XONG · $(grep -h -E 'TRƯỚC SFT|XONG SFT' $L | tr '\n' ' ' | cut -c1-200)"

# ── 1) 6 làn đo ──
gpu0_free() {   # phần đọc ảnh c6a (huấn luyện / sinh / giám khảo vLLM) đã thả GPU 0
  while pgrep -f box_run_c6_vlm.sh > /dev/null || pgrep -f vlm_1p3b.py > /dev/null || pgrep -f "vllm serve" > /dev/null; do sleep 60; done
}
lane() {  # $1 GPU  $2 cổng  $3… các việc (chạy nối tiếp trên máy chủ này)
  local G=$1 P=$2; shift 2
  [ $G = 0 ] && gpu0_free
  CUDA_VISIBLE_DEVICES=$G python -W ignore -u scripts/serve_hybrid_ui.py --ckpt $CK --port $P --no-rag --max-ctx 4096 \
    > /workspace/${TAG}_ui_$P.log 2>&1 &
  local SP=$! W=0
  until curl -s localhost:$P > /dev/null; do
    sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI đo $TAG: UI $P không lên"; kill $SP; return 1; }
  done
  local U=http://localhost:$P
  for job in "$@"; do
    case $job in
      agent)     $EV -u $CE run --suite agent --url $U --tag $TAG > /workspace/${TAG}_agent.log 2>&1 ;;
      humaneval) $EV -u $CE run --suite humaneval --url $U --tag $TAG > /workspace/${TAG}_humaneval.log 2>&1 ;;
      mbpp)      $EV -u $CE run --suite mbpp --url $U --tag $TAG > /workspace/${TAG}_mbpp.log 2>&1 ;;
      c7_*)      $EV -u scripts/c7_real.py eval --url $U --tag $TAG --shard ${job#c7_}/3 --workers 1 > /workspace/${TAG}_c7_${job#c7_}.log 2>&1 ;;
      office)    $EV -u scripts/office_probe_light.py $U $TAG > /workspace/${TAG}_office.log 2>&1 ;;
    esac
  done
  kill $SP
}
lane 1 8891 agent &       J1=$!
lane 1 8892 humaneval &   J2=$!
lane 1 8893 mbpp &        J3=$!
lane 0 8894 c7_1 office & J4=$!
lane 0 8895 c7_2 &        J5=$!
lane 0 8896 c7_3 &        J6=$!
log "đo nhanh c6sftb: đã mở 3 làn GPU 1 · 3 làn GPU 0 (chờ đọc ảnh c6a thả GPU 0 nếu chưa xong)"
wait $J1 $J2 $J3 $J4 $J5 $J6

# ── 2) kết quả ──
log "ĐO $TAG · C7 $($EV scripts/c7_real.py report --tag $TAG 2>/dev/null | cut -c1-200) · HE $(grep '###' /workspace/${TAG}_humaneval.log | tail -1 | cut -c1-90) · MBPP $(grep '###' /workspace/${TAG}_mbpp.log | tail -1 | cut -c1-90) · tool $(grep '###' /workspace/${TAG}_agent.log | tail -1 | cut -c1-90) · văn phòng $(grep '###' /workspace/${TAG}_office.log | cut -c1-120) · lỗi máy chủ $(cat /workspace/${TAG}_ui_*.log | grep -c '\[lỗi /api/chat\]')"
tar czf /workspace/${TAG}_ketqua.tgz out/code_eval/*${TAG}* out/office_probe_${TAG}.jsonl /workspace/${TAG}_*.log $L 2>/dev/null
hf upload donaldluong/nn-aivie-c6-sft-b /workspace/${TAG}_ketqua.tgz results/${TAG}_ketqua.tgz > /dev/null 2>&1 && log "$TAG đã đẩy HF kết quả đo"
log "sau-C6 SFT A ‖ B XONG — so 2 dòng 'ĐO c6sfta / c6sftb' với mốc cx3c (C7 2,5 · HE 15,9 · MBPP 26,5) · cx3e (C7 3,7 · tool 55) → chọn bản cho S2 bù tool-use"
