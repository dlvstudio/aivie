#!/bin/bash
# TK_108 sau C6 (user chọn phương án (a), 01/11): GPU chạy liền mạch ngay khi C6 học thân xong.
#   0  chờ " C6 học thân XONG" (box_run_c6.sh) — C6 lỗi hẳn thì DỪNG
#   1  bits theo lĩnh vực: c6.pt cuối (GPU 0) + c6_stable.pt trước đoạn hạ lr (GPU 1) → thấy anneal (tiếng Việt 45%) giúp bao nhiêu
#   2  SFT toàn bộ tham số SONG SONG, mỗi GPU một bản, cùng công thức P3.1 (lr 1e-5, 2 epoch, ctx 8192):
#        A (GPU 0)  chat P3 + office_fix (thay expert S3, user duyệt 01/11) → ckpt/c6_sft_a.pt
#        B (GPU 1)  A + ~40% code/tool đã kiểm (build_sft_c6.py)        → ckpt/c6_sft_b.pt
#      mỗi bản tự đẩy HF bf16 riêng tư (box_hf_sync_bf16.sh, gộp lịch sử) · AdamW fp32 hết bộ nhớ → AdamW 8-bit
#   3  đo mỗi bản NGAY khi xong, trên GPU của nó (2 máy chủ UI/GPU, ctx 4096): C7 355 đề · HumanEval · MBPP · tool-use v3 · Excel + easy20
#   4  bảng so A ↔ B ↔ mốc cx3c/cx3e → chờ user/Claude chọn bản thắng cho S2 (bù tool-use) – S4
# Dùng: (setsid nohup bash scripts/box_run_c6_sft.sh > /workspace/c6_sft_run.log 2>&1 < /dev/null &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
unset PIP_CONSTRAINT
EV=/workspace/evalenv/bin/python; CE=scripts/code_eval.py

# ── 0) dữ liệu B (chỉ CPU — làm ngay trong lúc C6 còn chạy) rồi chờ C6 ──
for ab in "a --only office" "b"; do set -- $ab
  [ -s data/sft_c6/sft_$1.jsonl ] || python -u scripts/build_sft_c6.py ${@:2} --out data/sft_c6/sft_$1.jsonl >> /workspace/c6sft_build.log 2>&1 \
    || { log "LỖI sau-C6: dựng dữ liệu SFT $1 ($(tail -1 /workspace/c6sft_build.log))"; exit 1; }
done
log "sau-C6: dữ liệu SFT sẵn sàng · $(grep '###' /workspace/c6sft_build.log | tr '\n' ' ') — chờ C6 học thân xong"
until grep -q " C6 học thân XONG" $S; do
  grep -q "LỖI C6: học thân không xong" $S && { log "sau-C6: DỪNG — C6 không xong (xem c6_train.log)"; exit 1; }
  sleep 60
done
sleep 30
pgrep -f train_hybrid_scale.py > /dev/null && { log "LỖI sau-C6: trainer C6 vẫn còn chạy sau dòng XONG"; exit 1; }

# ── 1) bits theo lĩnh vực (~5 phút) ──
( CUDA_VISIBLE_DEVICES=0 python -u scripts/c6_domain_bits.py --ckpt ckpt/c6.pt --out out/c6_domain_bits_final.json > /workspace/c6_bits_final.log 2>&1 ) &
B1=$!
( [ -s ckpt/c6_stable.pt ] && CUDA_VISIBLE_DEVICES=1 python -u scripts/c6_domain_bits.py --ckpt ckpt/c6_stable.pt --out out/c6_domain_bits_stable.json \
    > /workspace/c6_bits_stable.log 2>&1 ) &
B2=$!
wait $B1 $B2
log "C6 đo theo lĩnh vực · cuối: $(grep '###' /workspace/c6_bits_final.log | cut -c1-230) · trước hạ lr: $(grep '###' /workspace/c6_bits_stable.log 2>/dev/null | cut -c1-230)"
hf upload donaldluong/nn-aivie-c6-ckpt out/c6_domain_bits_final.json results/c6_domain_bits_final.json > /dev/null 2>&1
hf upload donaldluong/nn-aivie-c6-ckpt out/c6_domain_bits.json results/c6_domain_bits.json > /dev/null 2>&1

# ── 2) SFT A ‖ B ──
sft() {  # $1 = a|b  $2 = GPU  $3 = dữ liệu
  local T=$1 G=$2 D=$3 O=ckpt/c6_sft_$1.pt L=/workspace/c6sft_$1.log
  (setsid nohup bash scripts/box_hf_sync_bf16.sh $O donaldluong/nn-aivie-c6-sft-$T $L " SFT-${T^^} XONG ·" > /workspace/sync_$T.log 2>&1 < /dev/null &)
  for opt in adamw adamw8bit; do
    [ $opt = adamw8bit ] && { python -c "import bitsandbytes" 2>/dev/null || python -m pip install -q bitsandbytes >> $L 2>&1; }
    for try in 1 2 3; do
      log "sau-C6: SFT-${T^^} (GPU $G · $opt · lần $try) · $(wc -l < $D) mẫu"
      CUDA_VISIBLE_DEVICES=$G python -u scripts/train_hybrid_sft.py --resume --init-from ckpt/c6.pt --data $D \
        --max-len 8192 --tok-budget 16384 --bs 8 --grad-accum 4 --grad-ckpt --optim $opt \
        --lr 1e-5 --min-lr 1e-6 --warmup 50 --epochs 2 --eval-every 200 --save-every 500 --out $O >> $L 2>&1 && {
        log " SFT-${T^^} XONG · $(grep -h -E 'TRƯỚC SFT|XONG SFT' $L | tr '\n' ' ' | cut -c1-200)"; return 0; }
      grep -q "OutOfMemory\|out of memory" $L && break        # hết bộ nhớ → thử 8-bit, không chạy lại y hệt
      log "LỖI SFT-${T^^} lần $try: $(grep -E 'Error' $L | tail -1 | cut -c1-150) — chạy lại sau 2 phút (tự tiếp từ $O)"; sleep 120
    done
  done
  log "LỖI sau-C6: SFT-${T^^} không xong"; return 1
}
ev() {  # $1 = a|b  $2 = GPU — đo trên đúng GPU vừa học xong (GPU kia có thể còn đang học)
  local T=$1 G=$2 CK=ckpt/c6_sft_$1.pt P1=$((8881 + 2 * $2)) P2=$((8882 + 2 * $2)) TAG=c6sft$1 PIDS=""
  for p in $P1 $P2; do
    CUDA_VISIBLE_DEVICES=$G python -W ignore -u scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag --max-ctx 4096 > /workspace/${TAG}_ui_$p.log 2>&1 &
    PIDS="$PIDS $!"
  done
  for p in $P1 $P2; do
    W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI đo $TAG: UI $p"; kill $PIDS; return 1; }; done
  done
  ( $EV -u scripts/c7_real.py eval --url http://localhost:$P1 --tag $TAG --shard 1/2 --workers 1 > /workspace/${TAG}_c7_1.log 2>&1
    $EV -u $CE run --suite humaneval --url http://localhost:$P1 --tag $TAG > /workspace/${TAG}_humaneval.log 2>&1
    $EV -u $CE run --suite mbpp --url http://localhost:$P1 --tag $TAG > /workspace/${TAG}_mbpp.log 2>&1 ) &
  local J1=$!
  ( $EV -u scripts/c7_real.py eval --url http://localhost:$P2 --tag $TAG --shard 2/2 --workers 1 > /workspace/${TAG}_c7_2.log 2>&1
    $EV -u $CE run --suite agent --url http://localhost:$P2 --tag $TAG > /workspace/${TAG}_agent.log 2>&1
    $EV -u scripts/office_probe_light.py http://localhost:$P2 $TAG > /workspace/${TAG}_office.log 2>&1 ) &
  local J2=$!
  wait $J1 $J2                                                  # KHÔNG 'wait' trơn — sẽ chờ cả máy chủ UI
  kill $PIDS
  log "ĐO $TAG · C7 $($EV scripts/c7_real.py report --tag $TAG 2>/dev/null | cut -c1-200) · HE $(grep '###' /workspace/${TAG}_humaneval.log | tail -1 | cut -c1-90) · MBPP $(grep '###' /workspace/${TAG}_mbpp.log | tail -1 | cut -c1-90) · tool $(grep '###' /workspace/${TAG}_agent.log | tail -1 | cut -c1-90) · văn phòng $(grep '###' /workspace/${TAG}_office.log | cut -c1-120) · lỗi máy chủ $(cat /workspace/${TAG}_ui_*.log | grep -c '\[lỗi /api/chat\]')"
  tar czf /workspace/${TAG}_ketqua.tgz out/code_eval/*${TAG}* out/office_probe_${TAG}.jsonl /workspace/${TAG}_*.log /workspace/c6sft_$T.log 2>/dev/null
  hf upload donaldluong/nn-aivie-c6-sft-$T /workspace/${TAG}_ketqua.tgz results/${TAG}_ketqua.tgz > /dev/null 2>&1 && log "$TAG đã đẩy HF kết quả đo"
}
( sft a 0 data/sft_c6/sft_a.jsonl && ev a 0 ) &
RA=$!
sleep 300                                                       # lệch giờ nạp c6.pt (30 GB) của hai tiến trình
( sft b 1 data/sft_c6/sft_b.jsonl && ev b 1 ) &
RB=$!
wait $RA $RB
log "sau-C6 SFT A ‖ B XONG — so 2 dòng 'ĐO c6sfta / c6sftb' với mốc cx3c (C7 2,5 · HE 15,9 · MBPP 26,5) · cx3e (C7 3,7 · tool 55) → chọn bản cho S2 bù tool-use"
