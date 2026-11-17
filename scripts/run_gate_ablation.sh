#!/bin/bash
# Kiểm chứng cổng expert P3 trên Mac: normal (cổng đã học) · open (bỏ cổng) · off (tắt expert chữ)
# mỗi chế độ: HumanEval 40 pass@1 + Excel 27 + 6 văn phòng — chạy lại được, bài đã xong tự bỏ qua
cd "$(dirname "$0")/.."
CK=checkpoints/hf_hybrid/aivie_2p6b_p3_text_bf16.pt
for mode in normal open off; do
  .venv/bin/python -W ignore scripts/serve_hybrid_ui.py --ckpt $CK --port 8801 --no-rag --expert-mode $mode > out/ui_gate_$mode.log 2>&1 & P=$!
  until curl -s localhost:8801 >/dev/null; do sleep 5; done; sleep 20
  echo "== $mode $(date +%T) $(grep -c '\[expert\]' out/ui_gate_$mode.log) expert"
  .venv/bin/python -W ignore scripts/code_eval.py run --suite humaneval --url http://localhost:8801 --tag gate_$mode --n 40 --rounds 0 2>&1 | tail -1
  .venv/bin/python -W ignore scripts/gate_probe_office.py http://localhost:8801 $mode 2>&1 | tail -1
  kill $P; sleep 10
done
echo ALLDONE
