#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/learn/nn-aivie
source .venv/bin/activate
export PYTHONPATH=src

# ── CulturaX base (TK_042, sau 10c) — 42M v2 trên v12 (CulturaX ~500M + v11, repeat-weighted) ──
#
# ⚠️ ĐIỂM MẤU CHỐT — token SEEN ≠ corpus size:
#   Chinchilla 42M cần ~840M token SEEN. token seen = steps × batch × ctx = steps × 16 × 512.
#     30k step = 245M seen (0.29 epoch của 840M corpus) — CHƯA vắt hết data ⇒ underfit data lớn.
#     60k step = 490M seen (~17h)  ← mặc định: đủ thấy khác biệt, chưa full Chinchilla.
#     100k step = 819M seen (~29h) ← gần Chinchilla; chạy nếu 60k còn dốc xuống (eval chưa phẳng).
#   ⇒ Corpus lớn CHỈ giúp nếu train ĐỦ LÂU để thấy nó. Đây là chi phí thật của "thêm data".
#
# Held-out = qa_heldout (miền đích). fp32 b16 (bf16/fp16=0 lợi MPS, b>16 swap).
# ⚠️ TIÊN QUYẾT: huggingface-cli login + ingest_culturax.py + build_qa_heldout.py đã chạy.

test -f data/text_corpus/culturax_vi.jsonl || { echo "✗ thiếu culturax_vi — chạy ingest_culturax.py (đã login HF)"; exit 1; }
test -f data/sft/qa_heldout.jsonl || { echo "✗ thiếu qa_heldout — chạy build_qa_heldout.py"; exit 1; }

STEPS=${STEPS:-60000}   # override: STEPS=100000 bash out/run_culturax.sh
python -u scripts/train_neural_lm.py --arch v2 --opt muon --batch 16 \
  --manifest config/train_v12.yaml --d-model 512 --layers 8 --heads 8 --ctx 512 \
  --steps "$STEPS" --eval-every 1000 \
  --holdout data/sft/qa_heldout.jsonl --n-holdout 300 \
  --save out/lm_v2_v12_qa.pt --log-file out/lm_v2_v12_qa.log

# ── BẢNG so bits/token trên qa_heldout: v10 (19.7M) < v11 (180M) < v12 (~750M)? ──
echo "======== bits/token qa_heldout — thấp hơn = QA tốt hơn ========"
for CK in out/lm_v2_v10_fp32.pt out/lm_v2_v11_qa.pt out/lm_v2_v12_qa.pt; do
  test -f "$CK" || { echo "  (bỏ $CK — chưa có)"; continue; }
  python -u scripts/eval_lm_bits.py --ckpt "$CK" --holdout data/sft/qa_heldout.jsonl --answer-only --gen-samples 0 2>&1 | tail -2
done | tee out/eval_v12_table.txt
python -u scripts/eval_lm_bits.py --ckpt out/lm_v2_v12_qa.pt --gen-samples 6 2>&1 | tee -a out/eval_v12_table.txt

# NẾU v12 < v11 rõ trên qa_heldout ⇒ base đã mạnh lên ⇒ SFT LẠI trên v12 (recipe đã lành):
#   python -u scripts/sft_neural_lm.py --ckpt out/lm_v2_v12_qa.pt --save out/lm_v2_v12_sft.pt \
#     --log-file out/sft_v12.log
