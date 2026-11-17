#!/bin/bash
# TK_106 V5: đọc chữ tốt hơn + ĐỌC BÁO (web tin tức, báo in, ảnh chụp báo). Tiếp từ V4 (gộp 2×2, ≤8 ô).
#   dữ liệu mới: 40K trang báo (gen_news) · 80K ảnh chữ (40% nhiều số/hoá đơn, 30% giống ảnh chụp) · 40K Excel/biểu đồ
#   kiểu ĐỌC-TRƯỚC-RỒI-TRẢ-LỜI · ôn lại 60K mẫu V1 (đã gồm ~10% chỉ chữ)
#   mô hình: LoRA r64 cả 42 khối Mamba (in/out_proj) + 6 khối attention · mở 12 tầng cuối của mắt
# Trạng thái → /workspace/vlm_status.log
cd /workspace/nn-aivie
S=/workspace/vlm_status.log
PY=/workspace/zenv/bin/python
export PYTHONPATH=scripts:src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 HF_HUB_DISABLE_XET=1
log() { echo "$(date '+%F %T') $*" >> $S; }
V="$PY -W ignore -u scripts/vlm_1p3b.py"
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
B=ckpt/hybrid_1p3b_assistant_v5.pt
A5="--shuf 2 --no-thumb --max-tiles 8 --max-len 2304"
pkill -f "vllm serve"; sleep 5
rm -f ckpt/vlm/s4.pt ckpt/vlm/s2_vision_bf16.pt                 # V4 fp32 (đã có bản bf16) · V1 — đỡ ổ
# dọn ổ (user đồng ý 2026-10-06): model Ollama Gemma-26B (tải lại được), expert fp32 cũ, cache tinystories
pkill -x ollama; rm -rf /root/.ollama/models
rm -f ckpt/expert_code/code_v2.pt ckpt/expert_code/office_v2.pt ckpt/expert_code/v4c_*.pt
rm -rf /root/.cache/huggingface/datasets/nampdn-ai___tinystories-vietnamese /root/.cache/huggingface/hub/datasets--nampdn-ai--tinystories-vietnamese
log "V5: bắt đầu · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}')"

# 0. thử mô hình NGAY (LoRA cả Mamba + 12 tầng mắt) trên dữ liệu cũ — bắt lỗi trước khi tốn 1,5h dựng dữ liệu
TB=0
for tb in 6000 4000 2500; do
  if $V train --stage 2 --base $B $T $A5 --init ckpt/vlm/s4_vision_bf16.pt --data data/vlm/sft_ocr.jsonl --steps 3 --bs 32 --n-val 32 \
       --val-every 100000 --tower-last 12 --lora 64 --lora-mamba --tok-budget $tb --out ckpt/vlm/smoke5.pt > /workspace/vlm_s5_smoke.log 2>&1; then
    TB=$tb; break
  fi
  log "V5 thử lỗi ở ngân sách $tb: $(grep -m1 -E 'Error' /workspace/vlm_s5_smoke.log | cut -c1-150)"
done
rm -f ckpt/vlm/smoke5.pt
[ "$TB" -gt 0 ] || { log "LỖI V5 thử (xem vlm_s5_smoke.log)"; exit 1; }
log "V5 thử mô hình OK (ngân sách $TB): $(grep -E 'bước 2/' /workspace/vlm_s5_smoke.log)"

# 1. dữ liệu (CPU; ≤8 trình duyệt cùng lúc — container chỉ 1792 luồng)
mkdir -p data/vlm/news data/vlm/ocr5 data/vlm/screens5
[ -s data/vlm/news/docs.jsonl ] || $PY scripts/gen_news.py --dump --out data/vlm/news --fineweb "" --docs 60000 --seed 1 > /workspace/vlm5_dump.log 2>&1
log "V5 bài báo nguồn: $(grep '###' /workspace/vlm5_dump.log)"
for k in 0 1 2 3 4 5 6 7; do
  $PY scripts/gen_news.py --n 5000 --out data/vlm/news --split train_p$k --seed $((500+k)) --fineweb data/vlm/news/docs.jsonl \
     --docs 100000 > /workspace/vlm5_news_$k.log 2>&1 &
done
wait
cat data/vlm/news/train_p*.jsonl > data/vlm/news/train.jsonl && rm -f data/vlm/news/train_p*.jsonl
$PY scripts/gen_news.py --n 600 --out data/vlm/news --split test --seed 999 --fineweb data/vlm/news/docs_test.jsonl --docs 100000 \
   > /workspace/vlm5_news_test.log 2>&1
log "V5 trang báo: $(wc -l < data/vlm/news/train.jsonl) train · $(wc -l < data/vlm/news/test.jsonl) test · $(du -sh data/vlm/news | cut -f1)"
$PY scripts/gen_ocr.py --n 80000 --out data/vlm/ocr5 --workers 24 --seed 5 --num-frac 0.4 --photo-frac 0.3 > /workspace/vlm5_ocr.log 2>&1
for k in 0 1 2 3 4 5 6 7; do
  $PY scripts/gen_screens.py render --n 5000 --out data/vlm/screens5 --split train_p$k --seed $((700+k)) --read-first \
     --weights excel:4,chart:3,email:1,doc:1,code:1.5,chat:0.5 > /workspace/vlm5_scr_$k.log 2>&1 &
done
wait
cat data/vlm/screens5/train_p*.jsonl > data/vlm/screens5/train.jsonl && rm -f data/vlm/screens5/train_p*.jsonl
log "V5 ảnh chữ $(tail -1 /workspace/vlm5_ocr.log) · ảnh màn hình đọc-trước $(wc -l < data/vlm/screens5/train.jsonl) · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}')"

# 2. thầy điền câu cần hiểu (tóm tắt bài báo, giải thích code…) — vLLM bật SAU khi xong trình duyệt
sed -i 's/--gpu-memory-utilization 0.[0-9]*/--gpu-memory-utilization 0.85/' /workspace/vllm_vlm.sh
/workspace/vllm_vlm.sh
W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 60 ] && { log "LỖI V5: vLLM không lên"; exit 1; }; done
$PY scripts/gen_screens.py teach --out data/vlm/news --split train --workers 96 > /workspace/vlm5_teach.log 2>&1
$PY scripts/gen_screens.py teach --out data/vlm/news --split test --workers 96 >> /workspace/vlm5_teach.log 2>&1
$PY scripts/gen_screens.py teach --out data/vlm/screens5 --split train --workers 96 >> /workspace/vlm5_teach.log 2>&1
pkill -f "vllm serve"; sleep 10
log "V5 thầy điền: $(grep '###' /workspace/vlm5_teach.log | tr '\n' ' ')"

# 3. trộn (đường dẫn ảnh tương đối theo data/vlm)
$PY -c "
import json, random
r = random.Random(5)
def rows(p, pre, n=None):
    xs = [json.loads(l) for l in open(p)]
    xs = [x for x in xs if all(m['content'] for m in x['messages'])]
    for x in xs: x['images'] = [pre + q for q in x['images']]
    r.shuffle(xs); return xs[:n] if n else xs
mix = rows('data/vlm/news/train.jsonl', 'news/') + rows('data/vlm/ocr5/train.jsonl', 'ocr5/') + \
      rows('data/vlm/screens5/train.jsonl', 'screens5/') + rows('data/vlm/sft_ocr.jsonl', '', 60000)
r.shuffle(mix)
with open('data/vlm/sft5.jsonl', 'w') as f:
    for x in mix: f.write(json.dumps(x, ensure_ascii=False) + '\n')
t = rows('data/vlm/news/test.jsonl', 'news/')
with open('data/vlm/test_news.jsonl', 'w') as f:
    for x in t: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(len(mix), len(t))" > /workspace/vlm5_mix.log 2>&1
N=$(cut -d' ' -f1 /workspace/vlm5_mix.log); STEPS=$(( N / 32 )); [ $STEPS -gt 6500 ] && STEPS=6500
log "V5 trộn: $(cat /workspace/vlm5_mix.log) (train, test báo) → $STEPS bước"

# 4. train
$V train --stage 2 --base $B $T $A5 --init ckpt/vlm/s4_vision_bf16.pt --data data/vlm/sft5.jsonl --steps $STEPS --bs 32 \
   --lr 1e-4 --lr-expert 2e-4 --tower-last 12 --lr-tower 2e-5 --lora 64 --lora-mamba --lr-lora 1e-4 --warmup 200 \
   --val-every 1000 --save-every 500 --tok-budget $TB --out ckpt/vlm/s5.pt > /workspace/vlm_s5.log 2>&1 \
   || { log "LỖI V5 train (xem vlm_s5.log)"; exit 1; }
log "V5 train xong: $(grep 'val trước' /workspace/vlm_s5.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/vlm_s5.log)"

# 5. đo (sinh 6 tiến trình, vLLM tắt) → chấm → xuất
E="$V evalvlm --base $B $T --vision ckpt/vlm/s5.pt --device cuda --tag v5 --max-tiles 8"
for t in screen nat chart ovqa news; do $E --test data/vlm/test_$t.jsonl --n 200 > /workspace/vlm_eval5_$t.log 2>&1 & done
$V evalvlm --base $B $T --vision ckpt/vlm/s4_vision_bf16.pt --device cuda --tag v4 --max-tiles 8 --test data/vlm/test_news.jsonl --n 200 \
   > /workspace/vlm_eval4_news.log 2>&1 &
wait
/workspace/vllm_vlm.sh
until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; done
$V judgevlm --files "ckpt/vlm/eval_v5_*.json" > /workspace/vlm_judge5.log 2>&1
$V judgevlm --files "ckpt/vlm/eval_v4_test_news.json" >> /workspace/vlm_judge5.log 2>&1
log "V5 đo: $(grep '###' /workspace/vlm_judge5.log | sed 's/### //' | tr '\n' '|')"
pkill -f "vllm serve"
$PY -c "
import torch
sd=torch.load('ckpt/vlm/s5.pt',map_location='cpu')
out={'proj':{k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in sd['proj'].items()},'hid':sd['hid'],'cfg':sd['cfg'],
     'expert':{k:v.to(torch.bfloat16) for k,v in sd['expert'].items()}, 'lora': sd.get('lora', {})}
if 'tower' in sd: out['tower']=sd['tower']
torch.save(out,'ckpt/vlm/s5_vision_bf16.pt')"
log "V5 XONG · ckpt/vlm/s5_vision_bf16.pt $(du -h ckpt/vlm/s5_vision_bf16.pt | cut -f1) · GPU rảnh (vLLM đã tắt)"
