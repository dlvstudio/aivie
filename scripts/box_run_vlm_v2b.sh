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

# V2b: chạy lại từ giai đoạn 2 (lần trước hết ổ khi lưu bước 500 — đã dọn ổ + chỉ lưu tầng mắt được học, ghi tạm rồi đổi tên)
log "V2b: chạy lại giai đoạn 2 từ s1.pt · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}')"
TB2=6000
# giai đoạn 2: hỏi–đáp — bộ chiếu + expert thị giác + 6 tầng cuối của mắt (1 lượt dữ liệu)
N2=$(( $(wc -l < data/vlm/sft.jsonl) / 32 ))
$V train --stage 2 --base $B $T --init ckpt/vlm/s1.pt --data data/vlm/sft.jsonl --steps $N2 --bs 32 --lr 1e-4 \
   --lr-expert 2e-4 --tower-last 6 --lr-tower 2e-5 --warmup 200 --val-every 1000 --tok-budget $TB2 --out ckpt/vlm/s2.pt \
   > /workspace/vlm_s2.log 2>&1 || { log "LỖI V2 giai đoạn 2"; exit 1; }
log "V2 giai đoạn 2 xong ($N2 bước): $(grep 'val trước' /workspace/vlm_s2.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/vlm_s2.log)"

# đo: thầy chấm (vLLM nhỏ) + 4 bộ test ảnh song song + kiểm kỹ năng chữ (SGK) có/không expert thị giác
sed -i 's/--gpu-memory-utilization 0.90/--gpu-memory-utilization 0.45/; s/--max-num-seqs 128/--max-num-seqs 64/' /workspace/vllm_vlm.sh
/workspace/vllm_vlm.sh
until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; done
$PY -c "
import json
rows=[json.loads(l) for l in open('data/sgk/test.jsonl')][:150]
with open('data/vlm/test_sgk_text.jsonl','w') as f:
    for r in rows: f.write(json.dumps({'images':[],'messages':[{'role':'user','content':r['q']},{'role':'assistant','content':r['ref']}],'src':'sgk'},ensure_ascii=False)+'\n')"
E="$V evalvlm --base $B $T --vision ckpt/vlm/s2.pt --device cuda --tag v1"
for t in screen nat chart ovqa; do $E --test data/vlm/test_$t.jsonl --n 200 > /workspace/vlm_eval_$t.log 2>&1 & done
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
