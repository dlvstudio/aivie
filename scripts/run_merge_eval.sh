#!/bin/bash
# so 3 bản trên Mac: p3 (trước vòng thầy) · p3r1 (sau) · p3r1vp (sau + expert văn phòng cũ) — Excel 27 + văn phòng 6 + easy20
cd "$(dirname "$0")/.."
for pair in "p3:checkpoints/hf_hybrid/aivie_2p6b_p3_text_bf16.pt" "p3r1:checkpoints/hf_hybrid/aivie_2p6b_p3_r1_bf16.pt" "p3r1vp:checkpoints/hf_hybrid/aivie_2p6b_p3_r1vp_bf16.pt"; do
  tag=${pair%%:*}; ck=${pair#*:}
  .venv/bin/python -W ignore scripts/serve_hybrid_ui.py --ckpt $ck --port 8801 --no-rag > out/ui_merge_$tag.log 2>&1 & P=$!
  until curl -s localhost:8801 >/dev/null; do sleep 5; done; sleep 10
  echo "== $tag $(date +%T) $(grep -c '\[expert\]' out/ui_merge_$tag.log) expert"
  [ "$tag" = p3 ] && cp -n out/gate_probe_office_normal.jsonl out/gate_probe_office_p3.jsonl     # p3 đã đo hôm qua (chế độ normal)
  .venv/bin/python -W ignore scripts/gate_probe_office.py http://localhost:8801 $tag 2>&1 | tail -1
  .venv/bin/python -W ignore scripts/easy20_probe.py http://localhost:8801 $tag 2>&1 | tail -1
  kill $P; sleep 10
done
echo ALLDONE
