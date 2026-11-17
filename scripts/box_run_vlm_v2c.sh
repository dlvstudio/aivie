#!/bin/bash
# TK_106 V2 trên box: train model đọc ảnh trên v5 (thân + 3 expert đóng băng). Trạng thái → /workspace/vlm_status.log
cd /workspace/nn-aivie
S=/workspace/vlm_status.log
PY=/workspace/zenv/bin/python
export PYTHONPATH=scripts:src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
log() { echo "$(date '+%F %T') $*" >> $S; }
V="$PY -W ignore -u scripts/vlm_1p3b.py"
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
B=ckpt/hybrid_1p3b_assistant_v5.pt
mkdir -p ckpt/vlm

# V2c: chỉ phần đo + xuất (giai đoạn 2 đã xong)
log "V2c: đo 2 đợt × 3 tiến trình"
until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; done
$PY -c "
import json
rows=[json.loads(l) for l in open('data/sgk/test.jsonl')][:150]
with open('data/vlm/test_sgk_text.jsonl','w') as f:
    for r in rows: f.write(json.dumps({'images':[],'messages':[{'role':'user','content':r['q']},{'role':'assistant','content':r['ref']}],'src':'sgk'},ensure_ascii=False)+'\n')"
E="$V evalvlm --base $B $T --vision ckpt/vlm/s2.pt --device cuda --tag v1"
# 2 đợt × 3 tiến trình (mỗi tiến trình ~9,5GB GPU + vLLM 36GB → 6 cùng lúc sẽ vượt 80GB)
for t in screen nat chart; do $E --test data/vlm/test_$t.jsonl --n 200 > /workspace/vlm_eval_$t.log 2>&1 & done
wait
$E --test data/vlm/test_ovqa.jsonl --n 200 > /workspace/vlm_eval_ovqa.log 2>&1 &
$E --test data/vlm/test_sgk_text.jsonl --n 150 > /workspace/vlm_eval_sgk.log 2>&1 &
$E --test data/vlm/test_sgk_text.jsonl --n 150 --no-vexpert --tag v5goc > /workspace/vlm_eval_sgk0.log 2>&1 &
wait
log "V2 đo: $(grep -h '###' /workspace/vlm_eval_*.log | sed 's/### //' | tr '\n' ' | ')"
# phần thị giác bf16 để đưa về M4 (Mac tự ghép với assistant_v5 + mắt có sẵn)
$PY -c "
import torch
sd=torch.load('ckpt/vlm/s2.pt',map_location='cpu')
out={'proj':{k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in sd['proj'].items()},'hid':sd['hid'],
     'expert':{k:v.to(torch.bfloat16) for k,v in sd['expert'].items()}}
if 'tower' in sd: out['tower']=sd['tower']
torch.save(out,'ckpt/vlm/s2_vision_bf16.pt')"
log "V2 XONG · ckpt/vlm/s2_vision_bf16.pt $(du -h ckpt/vlm/s2_vision_bf16.pt | cut -f1)"
