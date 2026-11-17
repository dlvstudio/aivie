#!/bin/bash
# TK_106 V3 (chẩn đoán ~2h): V1 đọc chữ kém (chép lại giống 0,205) — do THIẾU dữ liệu đọc chữ hay do NÉN 3×3 quá mạnh?
# → thêm 40K ảnh chữ tổng hợp (PIL), train tiếp 1500 bước từ V1 trên trộn 50% ảnh chữ + 50% dữ liệu V1, đo lại ảnh màn hình.
cd /workspace/nn-aivie
S=/workspace/vlm_status.log
PY=/workspace/zenv/bin/python
export PYTHONPATH=scripts:src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
log() { echo "$(date '+%F %T') $*" >> $S; }
V="$PY -W ignore -u scripts/vlm_1p3b.py"
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
B=ckpt/hybrid_1p3b_assistant_v5.pt
pkill -f "vllm serve"; sleep 10
rm -f ckpt/vlm/s2.pt                                   # bản bf16 (s2_vision_bf16.pt) đã có + đã về M4 → đỡ ổ
$PY scripts/gen_ocr.py --n 40000 --out data/vlm/ocr --workers 24 > /workspace/vlm_ocr_gen.log 2>&1
$PY -c "
import json, random
r = random.Random(3)
ocr = [json.loads(l) for l in open('data/vlm/ocr/train.jsonl')]
for x in ocr: x['images'] = ['ocr/' + p for p in x['images']]
sft = [json.loads(l) for l in open('data/vlm/sft.jsonl')]
r.shuffle(sft)
mix = ocr + sft[:len(ocr)]
r.shuffle(mix)
with open('data/vlm/ocr_mix.jsonl', 'w') as f:
    for x in mix: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(len(mix))" > /workspace/vlm_ocr_mix.log 2>&1
log "V3 chẩn đoán: $(tail -1 /workspace/vlm_ocr_gen.log) · trộn $(cat /workspace/vlm_ocr_mix.log) mẫu · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}')"
$V train --stage 2 --base $B $T --init ckpt/vlm/s2_vision_bf16.pt --data data/vlm/ocr_mix.jsonl --steps 1500 --bs 32 --lr 1e-4 \
   --lr-expert 2e-4 --tower-last 6 --lr-tower 2e-5 --warmup 100 --val-every 500 --save-every 750 --tok-budget 6000 \
   --out ckpt/vlm/s3.pt > /workspace/vlm_s3.log 2>&1 || { log "LỖI V3 train (xem vlm_s3.log)"; exit 1; }
log "V3 train xong: $(grep 'val trước' /workspace/vlm_s3.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/vlm_s3.log)"
E="$V evalvlm --base $B $T --vision ckpt/vlm/s3.pt --device cuda --tag v3ocr"
$E --test data/vlm/test_screen.jsonl --n 200 > /workspace/vlm_eval3_screen.log 2>&1 &
$E --test data/vlm/test_chart.jsonl --n 200 > /workspace/vlm_eval3_chart.log 2>&1 &
$E --test data/vlm/test_sgk_text.jsonl --n 150 > /workspace/vlm_eval3_sgk.log 2>&1 &
wait
/workspace/vllm_vlm.sh
W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 60 ] && { log "LỖI V3: vLLM không lên"; break; }; done
$V judgevlm --files "ckpt/vlm/eval_v3ocr_*.json" > /workspace/vlm_judge3.log 2>&1
log "V3 đo: $(grep '###' /workspace/vlm_judge3.log | sed 's/### //' | tr '\n' '|')"
$PY -c "
import torch
sd=torch.load('ckpt/vlm/s3.pt',map_location='cpu')
out={'proj':{k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in sd['proj'].items()},'hid':sd['hid'],
     'expert':{k:v.to(torch.bfloat16) for k,v in sd['expert'].items()}}
if 'tower' in sd: out['tower']=sd['tower']
torch.save(out,'ckpt/vlm/s3_vision_bf16.pt')"
pkill -f "vllm serve"
log "V3 XONG · ckpt/vlm/s3_vision_bf16.pt · GPU rảnh (vLLM đã tắt)"
