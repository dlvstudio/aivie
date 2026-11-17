#!/bin/bash
# TK_107 · BỔ SUNG ẢNH CODE cho đọc ảnh (P3.4) + vòng thầy chung (P3.3') — chạy SONG SONG với box_vlm_data.sh B trên box P2.
#   1  vẽ (CPU): 20K ảnh code/terminal để train · 3K ảnh RIÊNG cho vòng thầy · 300 ảnh test — 25% làm giống ảnh chụp màn hình
#      (code thường · code cài 1 lỗi logic · trình soạn + TERMINAL có traceback trỏ đúng dòng)
#   2  thầy (GPU2, khi đã dịch xong) viết đáp án câu cần hiểu: giải thích code · chỉ lỗi + sửa · giải thích traceback
#   3  khi box_vlm_data.sh B XONG: gộp vào sft5.jsonl + test_code.jsonl, đẩy HF (công khai — ta tự tạo, nguồn code của ta)
# Trạng thái → /workspace/vlm_data_status.log (chung)
cd /workspace/nn-aivie
S=/workspace/vlm_data_status.log
VE=/workspace/vllm_env; PY=$VE/bin/python
export PATH=$VE/bin:$PATH PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4
unset PIP_CONSTRAINT
log() { echo "$(date '+%F %T') $*" >> $S; }
D=/workspace/vlmdata; O=data/vlm/screens_code
W="--weights excel:0,email:0,doc:0,chart:0,chat:0,code:1.2,term:1 --photo-frac 0.25"
HF_VLM=${HF_VLM:-donaldluong/nn-aivie-vlm-data}

if [ ! -f $D/code_render.done ]; then
  for k in 0 1 2 3 4 5 6 7; do
    $PY scripts/gen_screens.py render --n 2500 --out $O --split train_p$k --seed $((800+k)) $W > /workspace/vlmd_code_$k.log 2>&1 &
  done
  wait
  cat $O/train_p*.jsonl > $O/train.jsonl && rm -f $O/train_p*.jsonl
  $PY scripts/gen_screens.py render --n 3000 --out $O --split loop --seed 900 $W > /workspace/vlmd_code_loop.log 2>&1
  $PY scripts/gen_screens.py render --n 300 --out $O --split test --seed 990 $W > /workspace/vlmd_code_test.log 2>&1
  touch $D/code_render.done
  log "ảnh code xong · train $(wc -l < $O/train.jsonl) · vòng thầy $(wc -l < $O/loop.jsonl) · test $(wc -l < $O/test.jsonl) · $(du -sh $O | cut -f1)"
fi
if [ ! -f $D/code_teach.done ]; then
  until [ -f $D/translate.done ]; do sleep 60; done                 # GPU2 rảnh khi dịch xong
  for sp in train test; do
    $PY scripts/gen_screens.py teach --out $O --split $sp --llm http://127.0.0.1:8002/v1 --workers 96
  done > /workspace/vlmd_code_teach.log 2>&1
  touch $D/code_teach.done
  log "ảnh code: thầy điền $(grep '###' /workspace/vlmd_code_teach.log | tr '\n' ' ')"
fi
until grep -q "VLM DATA XONG" $S; do sleep 120; done
$PY - > /workspace/vlmd_code_mix.log 2>&1 <<'EOF'
import json, random
def rows(p, pre):
    xs = [json.loads(l) for l in open(p)]
    xs = [x for x in xs if all(m['content'] for m in x['messages'])]
    for x in xs: x['images'] = [pre + q for q in x['images']]
    return xs
code = rows('data/vlm/screens_code/train.jsonl', 'screens_code/')
mix = [json.loads(l) for l in open('data/vlm/sft5.jsonl')] + code
random.Random(7).shuffle(mix)
with open('data/vlm/sft5.jsonl', 'w') as f:
    for x in mix: f.write(json.dumps(x, ensure_ascii=False) + '\n')
with open('data/vlm/test_code.jsonl', 'w') as f:
    for x in rows('data/vlm/screens_code/test.jsonl', 'screens_code/'): f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(f"sft5 + code {len(code)} → {len(mix)}")
EOF
tar -cf /workspace/vlm_up/screens_code.tar $O
tar -I 'zstd -10 -T0' -cf /workspace/vlm_up/meta.tar.zst --exclude='*/img/*' --exclude='*.jpg' --exclude='*.png' data/vlm
hf upload-large-folder $HF_VLM /workspace/vlm_up --repo-type dataset > /workspace/vlmd_code_up.log 2>&1 \
  && log "ảnh code: gộp $(cat /workspace/vlmd_code_mix.log) · đã đẩy HF" || log "LỖI ảnh code: đẩy HF (vlmd_code_up.log)"
log "VLM CODE XONG"
