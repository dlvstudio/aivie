#!/bin/bash
# TK_106 V4 (hướng B): V3 cho thấy thêm dữ liệu chữ KHÔNG giúp đọc chữ → sửa kiến trúc:
#   gộp 2×2 (196 token/ô thay vì 81) · bỏ ảnh toàn cảnh khi nhiều ô · ≤8 ô · LoRA r64 trên 6 khối attention của thân
#   (cho thân học CHÉP từ token ảnh) · dữ liệu = V1 + 40K ảnh chữ. Trạng thái → /workspace/vlm_status.log
cd /workspace/nn-aivie
S=/workspace/vlm_status.log
PY=/workspace/zenv/bin/python
export PYTHONPATH=scripts:src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
log() { echo "$(date '+%F %T') $*" >> $S; }
V="$PY -W ignore -u scripts/vlm_1p3b.py"
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
B=ckpt/hybrid_1p3b_assistant_v5.pt
A4="--shuf 2 --no-thumb --max-tiles 8 --max-len 2304"
pkill -f "vllm serve"; sleep 5
rm -f ckpt/vlm/s3_vision_bf16.pt ckpt/vlm/s1.pt data/vlm/ocr_mix.jsonl            # chẩn đoán V3 / giai đoạn 1 V1 — đỡ ổ
$PY -c "
import json, random
r = random.Random(4)
rows = [json.loads(l) for l in open('data/vlm/sft.jsonl')]
for l in open('data/vlm/ocr/train.jsonl'):
    x = json.loads(l); x['images'] = ['ocr/' + p for p in x['images']]; rows.append(x)
r.shuffle(rows)
with open('data/vlm/sft_ocr.jsonl', 'w') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(len(rows))" > /workspace/vlm_v4_mix.log 2>&1
log "V4 (hướng B): dữ liệu giai đoạn 2 = $(cat /workspace/vlm_v4_mix.log) mẫu · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}')"

# thử nhỏ 2 giai đoạn (bắt lỗi/hết bộ nhớ sớm)
$V train --stage 1 --base $B $T $A4 --data data/vlm/align.jsonl --steps 4 --bs 64 --n-val 32 --val-every 100000 \
   --out ckpt/vlm/smoke4.pt > /workspace/vlm_s4a_smoke.log 2>&1 || { log "LỖI V4 thử giai đoạn 1 (xem vlm_s4a_smoke.log)"; exit 1; }
TB2=0
for tb in 6000 4000 2500; do
  if $V train --stage 2 --base $B $T $A4 --init ckpt/vlm/smoke4.pt --data data/vlm/sft_ocr.jsonl --steps 3 --bs 32 --n-val 32 \
       --val-every 100000 --tower-last 6 --lora 64 --tok-budget $tb --out ckpt/vlm/smoke4b.pt > /workspace/vlm_s4b_smoke.log 2>&1; then
    TB2=$tb; break
  fi
  log "V4 thử giai đoạn 2 lỗi ở ngân sách $tb: $(grep -m1 -E 'Error' /workspace/vlm_s4b_smoke.log | cut -c1-150)"
done
rm -f ckpt/vlm/smoke4b.pt
[ "$TB2" -gt 0 ] || { log "LỖI V4 thử giai đoạn 2 (xem vlm_s4b_smoke.log)"; exit 1; }
log "V4 thử OK · gđ1 $(grep -E 'bước 3/' /workspace/vlm_s4a_smoke.log | grep -o '[0-9]*s$') /4 bước · gđ2 ngân sách $TB2: $(grep -E 'bước 2/' /workspace/vlm_s4b_smoke.log | grep -o 'loss.*')"

# giai đoạn 1: bộ chiếu mới (cỡ vào 4608) học lại căn chỉnh
$V train --stage 1 --base $B $T $A4 --data data/vlm/align.jsonl --steps 1081 --bs 64 --lr 1e-3 --warmup 100 \
   --val-every 500 --out ckpt/vlm/s4a.pt > /workspace/vlm_s4a.log 2>&1 || { log "LỖI V4 giai đoạn 1"; exit 1; }
rm -f ckpt/vlm/smoke4.pt
log "V4 giai đoạn 1 xong: $(grep 'val trước' /workspace/vlm_s4a.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/vlm_s4a.log)"

# giai đoạn 2: bộ chiếu + expert thị giác + 6 tầng cuối mắt + LoRA attention
$V train --stage 2 --base $B $T $A4 --init ckpt/vlm/s4a.pt --data data/vlm/sft_ocr.jsonl --steps 3500 --bs 32 --lr 1e-4 \
   --lr-expert 2e-4 --tower-last 6 --lr-tower 2e-5 --lora 64 --lr-lora 1e-4 --warmup 200 --val-every 1000 --save-every 500 \
   --tok-budget $TB2 --out ckpt/vlm/s4.pt > /workspace/vlm_s4.log 2>&1 || { log "LỖI V4 giai đoạn 2 (xem vlm_s4.log)"; exit 1; }
log "V4 giai đoạn 2 xong: $(grep 'val trước' /workspace/vlm_s4.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/vlm_s4.log)"

# đo: sinh (5 tiến trình, vLLM tắt) → bật vLLM → chấm
E="$V evalvlm --base $B $T --vision ckpt/vlm/s4.pt --device cuda --tag v4 --max-tiles 8"
for t in screen nat chart ovqa; do $E --test data/vlm/test_$t.jsonl --n 200 > /workspace/vlm_eval4_$t.log 2>&1 & done
$E --test data/vlm/test_sgk_text.jsonl --n 150 > /workspace/vlm_eval4_sgk.log 2>&1 &
wait
log "V4 sinh xong: $(grep -h '### sinh' /workspace/vlm_eval4_*.log | wc -l)/5 bộ · lỗi: $(grep -l -E 'Error|Traceback' /workspace/vlm_eval4_*.log | tr '\n' ' ')"
/workspace/vllm_vlm.sh
W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 60 ] && { log "LỖI V4: vLLM không lên"; break; }; done
$V judgevlm --files "ckpt/vlm/eval_v4_*.json" > /workspace/vlm_judge4.log 2>&1
log "V4 đo: $(grep '###' /workspace/vlm_judge4.log | sed 's/### //' | tr '\n' '|')"
pkill -f "vllm serve"
$PY -c "
import torch
sd=torch.load('ckpt/vlm/s4.pt',map_location='cpu')
out={'proj':{k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in sd['proj'].items()},'hid':sd['hid'],'cfg':sd['cfg'],
     'expert':{k:v.to(torch.bfloat16) for k,v in sd['expert'].items()}, 'lora': sd.get('lora', {})}
if 'tower' in sd: out['tower']=sd['tower']
torch.save(out,'ckpt/vlm/s4_vision_bf16.pt')"
log "V4 XONG · ckpt/vlm/s4_vision_bf16.pt $(du -h ckpt/vlm/s4_vision_bf16.pt | cut -f1) · GPU rảnh (vLLM đã tắt)"
