#!/bin/bash
# chạy riêng chế độ off (open đã thấy hỏng: ra rác) — resume được
cd "$(dirname "$0")/.."
CK=checkpoints/hf_hybrid/aivie_2p6b_p3_text_bf16.pt
.venv/bin/python -W ignore scripts/serve_hybrid_ui.py --ckpt $CK --port 8801 --no-rag --expert-mode off > out/ui_gate_off.log 2>&1 & P=$!
until curl -s localhost:8801 >/dev/null; do sleep 5; done; sleep 20
echo "== off $(date +%T)"
.venv/bin/python -W ignore scripts/code_eval.py run --suite humaneval --url http://localhost:8801 --tag gate_off --n 40 --rounds 0 2>&1 | tail -1
.venv/bin/python -W ignore scripts/gate_probe_office.py http://localhost:8801 off 2>&1 | tail -1
kill $P
echo ALLDONE
