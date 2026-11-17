#!/bin/bash
# TK_107 P3 · đóng gói dữ liệu P3 trên Mac → 1 file p3_data.tar.zst để box tải từ HF (upload Mac→box trực tiếp chỉ ~13KB/s).
# Gồm: SFT (sft_p3) · dữ liệu 3 expert · bộ chấm (eval_v1, easy20, văn phòng, SGK, toán) · văn bản đệm cho eval_longctx
#      · dữ liệu vòng thầy v5 (cho P3.3 sau này).
#   bash scripts/pack_p3_data.sh            → out/p3_data.tar.zst (chỉ đóng gói)
#   bash scripts/pack_p3_data.sh --upload   → + đẩy lên HF dataset $HF_DATA (cần quyết định riêng tư/công khai trước)
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
HF_DATA=${HF_DATA:-donaldluong/nn-aivie-p3-data}
[ -f data/sft_p3/sft_p3.jsonl ] || PYTHONPATH=scripts:src $PY scripts/build_sft_p3.py --n-long 3000
# easy20: file gốc không còn trên Mac → dựng lại từ kết quả chấm v5 (cùng 20 câu + đáp án)
[ -f data/eval_clean/easy20.jsonl ] || $PY -c "
import json
d = json.load(open('out/box_results/expert_code/easy20_v5.json'))
with open('data/eval_clean/easy20.jsonl', 'w', encoding='utf-8') as f:
    for r in d['rows']:
        f.write(json.dumps({'q': r['q'], 'gold': r['gold']}, ensure_ascii=False) + '\n')"
# văn bản đệm cho bài "kim" 8K (eval_longctx đọc jsonl {text})
[ -f data/sft_p3/longctx_texts.jsonl ] || $PY -c "
import json, random
r = random.Random(7); out = []
for l in open('data/text_corpus/viwiki.jsonl', encoding='utf-8'):
    if r.random() < 0.02:
        t = json.loads(l)
        if len(t) > 2000: out.append({'text': t})
    if len(out) >= 4000: break
with open('data/sft_p3/longctx_texts.jsonl', 'w', encoding='utf-8') as f:
    for x in out: f.write(json.dumps(x, ensure_ascii=False) + '\n')"
mkdir -p data/v5 && cp -n out/box_results/v5/sft.jsonl out/box_results/v5/dpo.jsonl data/v5/ 2>/dev/null || true
tar -cf - data/sft_p3 data/sgk data/math data/multiturn data/office data/office_v2 \
    data/code_sft/vi_code.jsonl data/code_tb data/calib data/eval_clean data/v5 | zstd -10 -T0 -q -f -o out/p3_data.tar.zst
echo "đóng gói: $(du -h out/p3_data.tar.zst | cut -f1) → out/p3_data.tar.zst"
if [ "$1" = "--upload" ]; then
  PATH="$PWD/.venv/bin:$PATH"
  hf repo create $HF_DATA --repo-type dataset ${HF_PRIVATE:+--private} > /dev/null 2>&1 || true
  hf upload $HF_DATA out/p3_data.tar.zst p3_data.tar.zst --repo-type dataset
fi
