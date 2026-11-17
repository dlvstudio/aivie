#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/learn/nn-aivie
source .venv/bin/activate
export PYTHONPATH=src

# ── BƯỚC 10d (TK_042) — 42M v2 trên v11-WEIGHTED (repeat), KHÔNG cần data mới ──
# 10c chốt bằng số: uniform v11 TỆ HƠN v10 trên qa_heldout (6.508 > 6.065) — viwiki phẳng nhấn
# chìm QA. 10d test: repeat-weights (nâng QA/grade) có CỨU QA không?
#   v11w < v11 (6.508) → weighting giúp; v11w ≤ v10 (6.065) → weighting là CHÌA KHOÁ ⇒ đáng CulturaX
#   v11w ≈ v11 → weighting không đủ → vấn đề khác
# Held-out = qa_heldout. fp32 b16 (bf16/fp16=0 lợi MPS). ~9h/30k step (cùng budget 10c, so sạch).

test -f data/text_corpus/viwiki.jsonl || { echo "✗ thiếu viwiki — chạy ingest_viwiki trước"; exit 1; }
test -f data/sft/qa_heldout.jsonl || { echo "✗ thiếu qa_heldout — build_qa_heldout.py"; exit 1; }

python -u scripts/train_neural_lm.py --arch v2 --opt muon --batch 16 \
  --manifest config/train_v11_weighted.yaml --d-model 512 --layers 8 --heads 8 --ctx 512 \
  --steps 30000 --eval-every 500 \
  --holdout data/sft/qa_heldout.jsonl --n-holdout 300 \
  --save out/lm_v2_v11w_qa.pt --log-file out/lm_v2_v11w_qa.log

# ── BẢNG so qa_heldout (answer-only): v10 < v11(uniform) vs v11w(weighted)? ──
echo "======== bits/token qa_heldout ANSWER-ONLY (thấp = QA tốt) ========"
for CK in out/lm_v2_v10_fp32.pt out/lm_v2_v11_qa.pt out/lm_v2_v11w_qa.pt; do
  test -f "$CK" || { echo "  (bỏ $CK — chưa có)"; continue; }
  python -u scripts/eval_lm_bits.py --ckpt "$CK" --holdout data/sft/qa_heldout.jsonl --answer-only --gen-samples 0 2>&1 | tail -2
done | tee out/eval_10d_table.txt
# đọc tay sinh QA của v11w:
python -u scripts/eval_lm_bits.py --ckpt out/lm_v2_v11w_qa.pt --gen-samples 6 2>&1 | tee -a out/eval_10d_table.txt

# NẾU v11w ≤ v10 rõ ⇒ SFT trên v11w (recipe lành) rồi so QA sinh:
#   python -u scripts/sft_neural_lm.py --ckpt out/lm_v2_v11w_qa.pt --save out/lm_v2_v11w_sft.pt \
#     --log-file out/sft_v11w.log
