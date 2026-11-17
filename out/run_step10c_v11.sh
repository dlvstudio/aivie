#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/learn/nn-aivie
source .venv/bin/activate
export PYTHONPATH=src

# ── BƯỚC 10c (TK_042 §7.20b) — 42M v2 trên corpus v11 (v10 + viwiki 160M ≈ 180M token) ──
# Câu hỏi: thêm DATA (9×) có NÂNG chất lượng QA không? (G19c chốt: thêm THAM SỐ thì không).
# Held-out = qa_heldout (MIỀN ĐÍCH QA) → drives best-checkpoint + count baseline trên QA.
# fp32 b16 (đã đo: bf16/fp16 = 0 lợi trên MPS, batch>16 SWAP). ~9h/30k step.
# Corpus 9× to ⇒ ÍT lặp epoch hơn ⇒ bớt overfit — đúng nút G19c (gap +0.962).
#
# ⚠️ CHẠY build_qa_heldout.py TRƯỚC (cần data/sft/qa_heldout.jsonl). out/*.pt KHÔNG push (gitignore).

test -f data/sft/qa_heldout.jsonl || { echo "✗ thiếu qa_heldout — chạy scripts/build_qa_heldout.py trước"; exit 1; }

python -u scripts/train_neural_lm.py --arch v2 --opt muon --batch 16 \
  --manifest config/train_v11.yaml --d-model 512 --layers 8 --heads 8 --ctx 512 \
  --steps 30000 --eval-every 500 \
  --holdout data/sft/qa_heldout.jsonl --n-holdout 300 \
  --save out/lm_v2_v11_qa.pt --log-file out/lm_v2_v11_qa.log

# ── BẢNG so 2×2 (v10 vs v11 · qa_heldout vs essays) — số quyết định ──
echo "======== BẢNG SO bits/token (thấp hơn = tốt hơn) ========"
for CK in out/lm_v2_v10_fp32.pt out/lm_v2_v11_qa.pt; do
  test -f "$CK" || { echo "  (bỏ $CK — chưa có)"; continue; }
  for HO in data/sft/qa_heldout.jsonl data/real_vi/essays.jsonl; do
    python -u scripts/eval_lm_bits.py --ckpt "$CK" --holdout "$HO" --answer-only --gen-samples 0 2>&1 | tail -2
  done
done | tee out/eval_10c_table.txt
# đọc tay sinh QA của bản v11:
python -u scripts/eval_lm_bits.py --ckpt out/lm_v2_v11_qa.pt --gen-samples 6 2>&1 | tee -a out/eval_10c_table.txt
