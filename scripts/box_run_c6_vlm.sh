#!/bin/bash
# TK_108 lịch rút gọn (user duyệt 03/11): NỐI LẠI ĐỌC ẢNH lên bản SFT-A (thân C6) trên GPU 0 — trong lúc GPU 1 còn học SFT-B.
#   0  tải mắt SigLIP2 ngay (mạng/CPU) rồi chờ dòng "ĐO c6sfta" (A học + đo xong, GPU 0 rảnh)
#   1  thử KHÔNG học: phần đọc ảnh cũ (P3.4, thân p3) gắn thẳng lên thân mới → mốc "trôi" bao nhiêu
#   2  nối lại NHẸ, khởi từ phần cũ: giai đoạn 1 (bộ chiếu) 300 bước · giai đoạn 2 (bộ chiếu + expert thị giác + 6 tầng cuối mắt
#      + LoRA r64 attention) 1500 bước — công thức P3.4, ít bước hơn (P3.4: 1100 + 3500 từ đầu)
#   3  đo trên các bộ test giữ riêng (cũ-gắn-thẳng ↔ nối lại) → giám khảo Gemma-31B (vLLM, GPU 0) chấm
#   4  gói MỘT checkpoint (thân A + mắt + bộ chiếu + expert thị giác, LoRA giữ riêng) → ckpt/c6a_vlm_bf16.pt
# Repo HF riêng tư đang đầy → không đẩy checkpoint; Mac kéo về box_backup/. KHÔNG đẩy repo công khai.
# Dùng: (setsid nohup bash scripts/box_run_c6_vlm.sh > /workspace/c6_vlm_run.log 2>&1 < /dev/null &)
cd /workspace/nn-aivie
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
unset PIP_CONSTRAINT
VLLM=/workspace/vllm_env
B=ckpt/c6_sft_a.pt
OLD=ckpt/vlm/p3_vision_bf16.pt
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
A4="--shuf 2 --no-thumb --max-tiles 8 --max-len 3072"
V="python -W ignore -u scripts/vlm_1p3b.py"
D=/workspace/c6vlm; mkdir -p $D ckpt/vlm ckpt/vision
step() { [ -f $D/$1.done ]; }
done_() { touch $D/$1.done; log "đọc ảnh c6a: $1 xong · $2"; }
TESTS="screen nat chart ovqa news code"

# ── 0. mắt + chờ A ──
[ -s ckpt/vision/siglip2_so400m_384_bf16.safetensors ] || python scripts/fetch_vision_tower.py \
  --out ckpt/vision/siglip2_so400m_384_bf16.safetensors > /workspace/c6vlm_tower.log 2>&1 || { log "LỖI đọc ảnh c6a: tải mắt SigLIP2"; exit 1; }
log "đọc ảnh c6a: mắt SigLIP2 sẵn — chờ SFT-A học + đo xong để dùng GPU 0"
# KHỚP ĐẦU DÒNG sau dấu giờ: chính dòng log của script này từng chứa chuỗi chờ → tự kích hoạt sớm (03/11, chạy chung GPU với A)
until grep -qE '^[0-9-]+ [0-9:]+ ĐO c6sfta · ' $S; do
  grep -q "LỖI sau-C6: SFT-A" $S && { log "đọc ảnh c6a: DỪNG — SFT-A không xong"; exit 1; }
  sleep 120
done
sleep 60
pgrep -f "serve_hybrid_ui.py --ckpt ckpt/c6_sft_a.pt" > /dev/null && { log "LỖI đọc ảnh c6a: máy chủ đo A vẫn còn chạy"; exit 1; }
pgrep -f "train_hybrid_sft.py.*c6_sft_a.pt" > /dev/null && { log "LỖI đọc ảnh c6a: SFT-A vẫn đang học"; exit 1; }

# ── 1. phần cũ gắn thẳng (không học) ──
if ! step zero; then
  for t in $TESTS; do
    $V evalvlm --base $B $T --vision $OLD --device cuda --tag c6a0 --max-tiles 8 --test data/vlm/test_$t.jsonl --n 200 \
      > /workspace/c6vlm_eval0_$t.log 2>&1 || log "LỖI đọc ảnh c6a: đo gắn thẳng $t ($(grep -m1 Error /workspace/c6vlm_eval0_$t.log | cut -c1-120))"
  done
  done_ zero "sinh $(ls ckpt/vlm/eval_c6a0_*.json 2>/dev/null | wc -l) bộ test"
fi

# ── 2. nối lại nhẹ ──
if ! step smoke; then
  TB=0
  for tb in 8000 6000 4000; do
    if $V train --stage 2 --base $B $T $A4 --init $OLD --data data/vlm/sft5.jsonl --steps 3 --bs 32 --n-val 32 --val-every 100000 \
         --tower-last 6 --lora 64 --tok-budget $tb --out ckpt/vlm/c6a_smoke2.pt > /workspace/c6vlm_smoke2.log 2>&1; then
      TB=$tb; break
    fi
    log "đọc ảnh c6a: thử giai đoạn 2 lỗi ở ngân sách $tb: $(grep -m1 -E 'Error' /workspace/c6vlm_smoke2.log | cut -c1-150)"
  done
  rm -f ckpt/vlm/c6a_smoke2.pt
  [ "$TB" -gt 0 ] || { log "LỖI đọc ảnh c6a: thử giai đoạn 2 (c6vlm_smoke2.log)"; exit 1; }
  echo $TB > $D/tok_budget
  done_ smoke "ngân sách $TB"
fi
TB=$(cat $D/tok_budget)
if ! step s1; then
  $V train --stage 1 --base $B $T $A4 --init $OLD --data data/vlm/align.jsonl --steps 300 --bs 64 --lr 5e-4 --warmup 50 \
     --val-every 150 --out ckpt/vlm/c6a_s1.pt > /workspace/c6vlm_s1.log 2>&1 || { log "LỖI đọc ảnh c6a: giai đoạn 1 (c6vlm_s1.log)"; exit 1; }
  done_ s1 "$(grep 'val trước' /workspace/c6vlm_s1.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/c6vlm_s1.log)"
fi
if ! step s2; then
  $V train --stage 2 --base $B $T $A4 --init ckpt/vlm/c6a_s1.pt --data data/vlm/sft5.jsonl --steps 1500 --bs 32 --lr 1e-4 \
     --lr-expert 2e-4 --tower-last 6 --lr-tower 2e-5 --lora 64 --lr-lora 1e-4 --warmup 100 --val-every 500 --save-every 500 \
     --tok-budget $TB --out ckpt/vlm/c6a_s2.pt > /workspace/c6vlm_s2.log 2>&1 || { log "LỖI đọc ảnh c6a: giai đoạn 2 (c6vlm_s2.log)"; exit 1; }
  python -c "
import torch
sd = torch.load('ckpt/vlm/c6a_s2.pt', map_location='cpu')
out = {'proj': {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd['proj'].items()}, 'hid': sd['hid'], 'cfg': sd['cfg'],
       'expert': {k: v.to(torch.bfloat16) for k, v in sd['expert'].items()}, 'lora': sd.get('lora', {})}
if 'tower' in sd: out['tower'] = sd['tower']
torch.save(out, 'ckpt/vlm/c6a_vision_bf16.pt')"
  done_ s2 "$(grep 'val trước' /workspace/c6vlm_s2.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/c6vlm_s2.log)"
fi

# ── 3. đo nối lại + giám khảo chấm cả hai (c6a0 gắn thẳng · c6a nối lại) ──
if ! step eval; then
  for t in $TESTS; do
    $V evalvlm --base $B $T --vision ckpt/vlm/c6a_s2.pt --device cuda --tag c6a --max-tiles 8 --test data/vlm/test_$t.jsonl --n 200 \
      > /workspace/c6vlm_eval_$t.log 2>&1 || log "LỖI đọc ảnh c6a: đo $t ($(grep -m1 Error /workspace/c6vlm_eval_$t.log | cut -c1-120))"
  done
  setsid nohup $VLLM/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
     --max-model-len 6144 --gpu-memory-utilization 0.85 --max-num-seqs 64 --port 8000 > /workspace/c6vlm_vllm.log 2>&1 < /dev/null &
  VP=$!
  W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI đọc ảnh c6a: vLLM không lên"; break; }; done
  $V judgevlm --files "ckpt/vlm/eval_c6a*_*.json" > /workspace/c6vlm_judge.log 2>&1
  kill -- -$VP 2>/dev/null || kill $VP; sleep 15
  tar -czf /workspace/c6vlm_results.tgz ckpt/vlm/eval_c6a*_*.json /workspace/c6vlm_*.log 2>/dev/null
  done_ eval "$(grep '###' /workspace/c6vlm_judge.log | sed 's/### //' | tr '\n' '|' | cut -c1-900)"
fi

# ── 4. MỘT checkpoint ──
if ! step pack; then
  $V pack --base $B --vision ckpt/vlm/c6a_vision_bf16.pt $T --max-tiles 8 --out ckpt/c6a_vlm.pt > /workspace/c6vlm_pack.log 2>&1 \
    || { log "LỖI đọc ảnh c6a: gói checkpoint"; exit 1; }
  python - <<'EOF' >> /workspace/c6vlm_pack.log 2>&1
import torch                                                     # bản gọn chạy trên Mac: thân bf16, bỏ optimizer
o = torch.load("ckpt/c6a_vlm.pt", map_location="cpu", weights_only=False)
o["state"] = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in o["state"].items()}
o.pop("opt", None)
torch.save(o, "ckpt/c6a_vlm_bf16.pt")
EOF
  rm -f ckpt/c6a_vlm.pt
  done_ pack "$(du -h ckpt/c6a_vlm_bf16.pt | cut -f1) · Mac kéo về box_backup/c6sft/"
fi
log "đọc ảnh c6a XONG — so 'c6a0' (gắn thẳng) ↔ 'c6a' (nối lại) ở dòng eval trên · GPU 0 rảnh chờ B"
