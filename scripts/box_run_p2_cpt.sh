#!/bin/bash
# TK_107 P2 · TRAIN TIẾP thân nuôi lớn 2,6B trên box 4–8×H100/H200 (torchrun; bs/GPU tự chọn theo VRAM) — mốc 1 của lộ trình WSD (~20 tỷ token).
#   A1  chỉ khối mới · ngữ cảnh 2K · RoPE 1e4 (giữ)           ~4 tỷ token
#   A2  chỉ khối mới · ngữ cảnh 8K · RoPE 1e6 (ABF)          ~6 tỷ token
#   B   toàn bộ      · ngữ cảnh 8K · RoPE 1e6 · lịch WSD      ~10 tỷ token (hạ lr 10% cuối; lưu bản *_stable.pt để KÉO DÀI lên 50/100 tỷ)
# Mỗi pha: --init-from pha trước, --reset-step (đổi tập tham số học → optimizer mới), ấm 100–300 bước.
# Lưu mỗi 500 bước + sao lưu HF mỗi 1000 bước (BẮT BUỘC — bài học TK_103: Destroy box = mất hết). Trạng thái → /workspace/cpt_status.log
cd /workspace/nn-aivie
S=/workspace/cpt_status.log
PY=${PY:-/workspace/zenv/bin/python}
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') $*" >> $S; }
NG=$(nvidia-smi -L | wc -l)
HF_CKPT=${HF_CKPT:-donaldluong/nn-aivie-hybrid-2p6b-ckpt}
TR="$PY -m torch.distributed.run --standalone --nproc_per_node $NG scripts/train_hybrid_scale.py"
REPLAY=data/pretrain_vi
MIX2K="data/cpt/vi_web:0.52,data/cpt/long:0.06,data/cpt/vi_wiki:0.025,data/cpt/en_edu:0.145,data/cpt/math:0.05,data/cpt/code:0.08,$REPLAY:0.12"
MIX8K="data/cpt/vi_web:0.42,data/cpt/long:0.20,data/cpt/vi_wiki:0.025,data/cpt/en_edu:0.135,data/cpt/math:0.05,data/cpt/code:0.08,$REPLAY:0.09"
# đoạn HẠ LR cuối pha B (10% ≈ 1 tỷ token) — "chốt" bằng dữ liệu chất lượng cao (kiểu SmolLM3): nhiều toán/code/giáo dục/wiki
MIXANNEAL="data/cpt/vi_web:0.30,data/cpt/long:0.15,data/cpt/vi_wiki:0.08,data/cpt/en_edu:0.17,data/cpt/math:0.12,data/cpt/code:0.12,$REPLAY:0.06"
log "P2: $NG GPU · $(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | sort | uniq -c | tr '\n' ' ')"

# 0. chờ user `hf auth login` (token WRITE: tải ckpt riêng tư + sao lưu) → tải thân 1.3B → chèn khối (chưa cần dữ liệu P1)
until hf auth whoami >/dev/null 2>&1; do
  [ -f /workspace/.p2_wait_logged ] || { log "P2: CHỜ user 'hf auth login' (token WRITE) trên box…"; touch /workspace/.p2_wait_logged; }
  sleep 60
done
HF_DATA=${HF_DATA:-donaldluong/nn-aivie-cpt-data}
mkdir -p ckpt data
[ -f ckpt/hybrid_1p3b.pt ] || { $PY scripts/download_hybrid_ckpt.py > /workspace/cpt_dl.log 2>&1 && \
    ln -sf "$(ls $PWD/checkpoints/hf_hybrid/hybrid_1p3b_step*.pt | head -1)" ckpt/hybrid_1p3b.pt; }
[ -f ckpt/hybrid_1p3b.pt ] || { log "LỖI P2: không tải được thân 1.3B (xem cpt_dl.log)"; exit 1; }
[ -f ckpt/hybrid_2p6b_x.pt ] || $PY scripts/expand_depth.py --src ckpt/hybrid_1p3b.pt --out ckpt/hybrid_2p6b_x.pt --check > /workspace/cpt_expand.log 2>&1
grep -q "sai lệch logit lớn nhất 0.00e+00" /workspace/cpt_expand.log || { log "LỖI P2: thân nuôi lớn KHÔNG khớp bản cũ (xem cpt_expand.log)"; exit 1; }
log "P2 chèn khối OK: $(grep check /workspace/cpt_expand.log)"

# dữ liệu P1: chờ box P1 đặt dấu _P1_XONG lên kho HF (mọi shard đã đẩy) → tải · dữ liệu pretrain cũ (ôn lại)
until $PY -c "from huggingface_hub import file_exists as f; import sys; sys.exit(0 if f('$HF_DATA','_P1_XONG',repo_type='dataset') else 1)" 2>/dev/null; do
  [ -f /workspace/.p2_wait_p1 ] || { log "P2: chờ P1 xong (dấu _P1_XONG trên $HF_DATA)…"; touch /workspace/.p2_wait_p1; }
  [ -d $REPLAY ] || { hf download donaldluong/nn-aivie-pretrain-vi pretrain_vi.tar.zst --repo-type dataset --local-dir /workspace/dl > /workspace/cpt_dl_replay.log 2>&1 \
    && tar -I zstd -xf /workspace/dl/pretrain_vi.tar.zst -C data && rm -f /workspace/dl/pretrain_vi.tar.zst; }
  sleep 120
done
[ -d data/cpt/vi_web ] || hf download $HF_DATA --repo-type dataset --local-dir data/cpt > /workspace/cpt_dl_data.log 2>&1
[ -d $REPLAY ] || { hf download donaldluong/nn-aivie-pretrain-vi pretrain_vi.tar.zst --repo-type dataset --local-dir /workspace/dl \
    && tar -I zstd -xf /workspace/dl/pretrain_vi.tar.zst -C data && rm -f /workspace/dl/pretrain_vi.tar.zst; }
[ -d data/cpt/vi_web ] && [ -d $REPLAY ] || { log "LỖI P2: thiếu dữ liệu (xem cpt_dl*.log)"; exit 1; }
log "P2 đã tải: $(du -sh data/cpt | cut -f1) dữ liệu mới ($(ls data/cpt/*/*.npy | wc -l) shard) · $(du -sh $REPLAY | cut -f1) ôn lại · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}')"

# 1. đo nhanh (60 bước) từng cấu hình ứng viên "bs:grad-ckpt" cho TỪNG pha → chọn cái nhanh nhất mà vừa ≤85% VRAM.
#    (đo thật 4×H100 80GB: 8K toàn bộ tham số KHÔNG có grad-ckpt OOM kể cả bs 1; 2K chỉ-khối-mới bs 8 OOM, bs 4 = 59GB)
MEMG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)        # MiB
LIM=$(( MEMG * 85 / 100 / 1024 ))
if [ $MEMG -ge 120000 ]; then K=2; else K=1; fi                                               # H200: thử lô gấp đôi
bench() {  # $1 ctx · $2 bs · $3 ckpt 0/1 · $4 trộn · $5 cờ thêm → in "VRAM_GB TOK_S" (999 0 nếu lỗi)
  local F="$5"; [ "$3" = 1 ] && F="$F --grad-ckpt"
  local L=/workspace/cpt_bench_$1_$2_$3$( [ -n "$5" ] && echo _new ).log
  $TR --data "$4" --ctx $1 --bs $2 --grad-accum 1 --steps 60 --eval-every 30 --save-every 100000 --warmup 10 \
      --init-from ckpt/hybrid_2p6b_x.pt $F --reset-step --out ckpt/_bench.pt > $L 2>&1
  local l=$(grep -E 'step 60/' $L)
  [ -n "$l" ] && echo "$(echo "$l" | grep -o 'VRAM [0-9]*' | grep -o '[0-9]*') $(echo "$l" | grep -o '[0-9.]*K tok/s' | grep -o '[0-9.]*')" || echo "999 0"; }
pick() {   # $1 tên pha · $2 ctx · $3 trộn · $4 cờ · $5.. ứng viên "bs:ckpt" → đặt BEST_BS BEST_CK
  local name=$1 ctx=$2 mix=$3 fl=$4; shift 4; BEST_BS=0; BEST_CK=0; local bt=0 rep=""
  for c in "$@"; do local bs=$(( ${c%:*} * K )) ck=${c#*:}; set -- $(bench $ctx $bs $ck "$mix" "$fl")
    rep="$rep bs$bs/ckpt$ck=${2}K@${1}GB"
    [ $1 -le $LIM ] && awk "BEGIN{exit !($2 > $bt)}" && { bt=$2; BEST_BS=$bs; BEST_CK=$ck; }
  done
  log "P2 đo $name (ctx $ctx):$rep → chọn bs $BEST_BS ckpt $BEST_CK (${bt}K tok/s, giới hạn ${LIM}GB)"
  [ $BEST_BS -gt 0 ] || { log "LỖI P2: $name không cấu hình nào chạy được"; exit 1; }; }
BENV=/workspace/p2_bench.env                                    # đã đo rồi (chạy lại kịch bản) → dùng lại, khỏi đo 30 phút
if [ -s $BENV ]; then . $BENV; log "P2 dùng lại cấu hình đã đo: A1 $A1_BS/$A1_CK · A2 $A2_BS/$A2_CK · B $B_BS/$B_CK"
else
pick A1 2048 "$MIX2K" "--train-new-only" 4:0 8:0 8:1 16:1;  A1_BS=$BEST_BS; A1_CK=$BEST_CK
pick A2 8192 "$MIX8K" "--train-new-only" 1:0 2:0 2:1 4:1;   A2_BS=$BEST_BS; A2_CK=$BEST_CK
pick B  8192 "$MIX8K" ""                 1:1 2:1 4:1;       B_BS=$BEST_BS;  B_CK=$BEST_CK
echo "A1_BS=$A1_BS A1_CK=$A1_CK A2_BS=$A2_BS A2_CK=$A2_CK B_BS=$B_BS B_CK=$B_CK" > $BENV
fi
ck() { [ "$1" = 1 ] && echo --grad-ckpt; }
rm -f ckpt/_bench*.pt

# 2. các pha (lô toàn cục ~1M token/bước)
GA1=$(( 1048576 / (NG * A1_BS * 2048) )); GA2=$(( 1048576 / (NG * A2_BS * 8192) )); GA3=$(( 1048576 / (NG * B_BS * 8192) ))
if [ ! -f ckpt/cpt_a1.done ]; then
$TR --data "$MIX2K" --ctx 2048 --bs $A1_BS --grad-accum $GA1 $(ck $A1_CK) --tokens 4e9 --lr 3e-4 --min-lr 3e-4 --warmup 200 --sched wsd --decay-frac 0 \
    --init-from ckpt/hybrid_2p6b_x.pt --train-new-only --reset-step --save-every 500 --eval-every 250 \
    --out ckpt/cpt_a1.pt --hf-repo $HF_CKPT --hf-every 1000 --hf-name cpt_a1.pt > /workspace/cpt_a1.log 2>&1 || { log "LỖI P2 A1"; exit 1; }
log "P2 A1 xong: $(grep -E 'bits TRƯỚC|bits cuối' /workspace/cpt_a1.log | tr '\n' ' ')"
touch ckpt/cpt_a1.done
else log "P2 A1 đã xong trước đó → bỏ qua"; fi
if [ ! -f ckpt/cpt_a2.done ]; then
$TR --data "$MIX8K" --ctx 8192 --bs $A2_BS --grad-accum $GA2 $(ck $A2_CK) --tokens 6e9 --lr 3e-4 --min-lr 3e-4 --warmup 100 --sched wsd --decay-frac 0 \
    --init-from ckpt/cpt_a1.pt --train-new-only --reset-step --rope-base 1e6 --save-every 500 --eval-every 250 \
    --out ckpt/cpt_a2.pt --hf-repo $HF_CKPT --hf-every 1000 --hf-name cpt_a2.pt > /workspace/cpt_a2.log 2>&1 || { log "LỖI P2 A2"; exit 1; }
log "P2 A2 xong: $(grep -E 'bits TRƯỚC|bits cuối' /workspace/cpt_a2.log | tr '\n' ' ')"
touch ckpt/cpt_a2.done
else log "P2 A2 đã xong trước đó → bỏ qua"; fi
if [ ! -f ckpt/cpt_b.done ]; then
echo "$MIXANNEAL" > ckpt/cpt_b.anneal                              # train_hybrid_scale đọc khi tới đoạn hạ lr
$TR --data "$MIX8K" --ctx 8192 --bs $B_BS --grad-accum $GA3 $(ck $B_CK) --tokens 10e9 --lr 1.5e-4 --min-lr 1.5e-5 --warmup 300 --sched wsd --decay-frac 0.1 \
    --init-from ckpt/cpt_a2.pt --reset-step --save-every 500 --eval-every 250 \
    --out ckpt/cpt_b.pt --hf-repo $HF_CKPT --hf-every 1000 --hf-name cpt_b.pt > /workspace/cpt_b.log 2>&1 || { log "LỖI P2 B"; exit 1; }
log "P2 B xong: $(grep -E 'bits TRƯỚC|bits cuối' /workspace/cpt_b.log | tr '\n' ' ')"
touch ckpt/cpt_b.done
else log "P2 B đã xong trước đó → bỏ qua"; fi

# 3. đo ngữ cảnh dài: bản 1.3B gốc (2K, RoPE 1e4) vs bản mới
for c in ckpt/hybrid_1p3b.pt ckpt/cpt_b.pt; do
  $PY scripts/eval_longctx.py --ckpt $c --texts data/cpt/long --lens 1024,2048,4096,8192 --n 8 --out ${c%.pt}_longctx.json > /workspace/cpt_longctx_$(basename $c .pt).log 2>&1
done
log "P2 đo ngữ cảnh dài: $(grep -h '###' /workspace/cpt_longctx_*.log | cut -c1-400 | tr '\n' ' ')"
log "P2 XONG (mốc 20 tỷ) · bản ổn định để kéo dài: ckpt/cpt_b_stable.pt"
