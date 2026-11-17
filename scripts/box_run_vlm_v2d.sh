#!/bin/bash
# V2d: đo = sinh (6 tiến trình, KHÔNG vLLM) → tắt → bật vLLM 0,85 → chấm → xuất phần thị giác bf16
cd /workspace/nn-aivie
S=/workspace/vlm_status.log
PY=/workspace/zenv/bin/python
export PYTHONPATH=scripts:src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
log() { echo "$(date '+%F %T') $*" >> $S; }
V="$PY -W ignore -u scripts/vlm_1p3b.py"
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
B=ckpt/hybrid_1p3b_assistant_v5.pt
log "V2d: đo — sinh 6 tiến trình (vLLM tắt) rồi chấm"
pkill -f "vllm serve"; sleep 5
[ -f data/vlm/test_sgk_text.jsonl ] || $PY -c "
import json
rows=[json.loads(l) for l in open('data/sgk/test.jsonl')][:150]
with open('data/vlm/test_sgk_text.jsonl','w') as f:
    for r in rows: f.write(json.dumps({'images':[],'messages':[{'role':'user','content':r['q']},{'role':'assistant','content':r['ref']}],'src':'sgk'},ensure_ascii=False)+'\n')"
E="$V evalvlm --base $B $T --vision ckpt/vlm/s2.pt --device cuda --tag v1"
for t in screen nat chart ovqa; do $E --test data/vlm/test_$t.jsonl --n 200 > /workspace/vlm_eval_$t.log 2>&1 & done
$E --test data/vlm/test_sgk_text.jsonl --n 150 > /workspace/vlm_eval_sgk.log 2>&1 &
$E --test data/vlm/test_sgk_text.jsonl --n 150 --no-vexpert --tag v5goc > /workspace/vlm_eval_sgk0.log 2>&1 &
wait
log "V2d sinh xong: $(grep -h '### sinh' /workspace/vlm_eval_*.log | wc -l)/6 bộ · lỗi: $(grep -l -E 'Error|Traceback' /workspace/vlm_eval_*.log | tr '\n' ' ')"
sed -i 's/--gpu-memory-utilization 0.45/--gpu-memory-utilization 0.85/' /workspace/vllm_vlm.sh
/workspace/vllm_vlm.sh
W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 60 ] && { log "LỖI V2d: vLLM không lên (xem vllm_vlm.log)"; break; }; done
$V judgevlm --files "ckpt/vlm/eval_*.json" > /workspace/vlm_judge.log 2>&1
log "V2 đo: $(grep '###' /workspace/vlm_judge.log | sed 's/### //' | tr '\n' '|')"
$PY -c "
import torch
sd=torch.load('ckpt/vlm/s2.pt',map_location='cpu')
out={'proj':{k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in sd['proj'].items()},'hid':sd['hid'],
     'expert':{k:v.to(torch.bfloat16) for k,v in sd['expert'].items()}}
if 'tower' in sd: out['tower']=sd['tower']
torch.save(out,'ckpt/vlm/s2_vision_bf16.pt')"
log "V2 XONG · ckpt/vlm/s2_vision_bf16.pt $(du -h ckpt/vlm/s2_vision_bf16.pt | cut -f1)"
