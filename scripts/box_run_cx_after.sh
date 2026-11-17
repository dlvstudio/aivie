#!/bin/bash
# TK_108 sau khi đo lại cx2 + cx: 1) đo c4c bằng khung v3 (so công bằng với cx2r 47,5%)
# 2) THỬ NHANH trainCPTX nhiều-GPU trên 1 GPU (torchrun nproc 1, data/cx3): 60 bước có lưu trạng thái → chạy tiếp tới 90 bước.
# Dùng: (setsid nohup bash scripts/box_run_cx_after.sh > /workspace/cx_after.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q "Đo lại cx2 + cx XONG" $S; do sleep 60; done
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
CE=/workspace/c1/scripts/code_eval.py
$PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt ckpt/aivie_p3_c4c_bf16.pt --port 8897 --no-rag > /workspace/c4cv3_ui.log 2>&1 &
UP=$!
W=0; until curl -s localhost:8897 > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI c4c v3: UI"; kill $UP; break; }; done
$EV -u $CE run --suite agent --url http://localhost:8897 --tag c4c_v3 > /workspace/c4cv3_agent.log 2>&1 \
  && log "c4c agent (khung v3; so với cx2r 47,5) · $(grep '###' /workspace/c4cv3_agent.log | tail -1 | cut -c1-200)" || log "LỖI c4c agent v3"
kill $UP; sleep 15
export PYTHONPATH=scripts:src
cd /workspace/nn-aivie
T="$PY -m torch.distributed.run --standalone --nproc_per_node 1 scripts/expert_office_1p3b.py trainCPTX --base ckpt/p3_sft.pt --code '' \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --pos every4 --raw-hid 8192 --cx-dir data/cx3 \
   --cx-mix opc_algo:0.23,opc_snip:0.15,opc_files:0.13,gh:0.26,web:0.08,vi:0.15 --seq 2048 --bs 4 --accum 4 --lr 3e-4 \
   --eval-every 30 --resume --tag cx3smoke"
eval "$T --steps 60" > /workspace/cx3_smoke.log 2>&1 && eval "$T --steps 90" >> /workspace/cx3_smoke.log 2>&1 \
  && log "Thử nhiều-GPU (1 GPU) OK · $(grep -E 'chạy tiếp|bước 50 |bước 90' /workspace/cx3_smoke.log | tr '\n' ' ' | cut -c1-300)" \
  || log "LỖI thử nhiều-GPU (cx3_smoke.log)"
rm -f ckpt/expert_code/cx3smoke_*.pt ckpt/expert_code/trainCPTX_cx3smoke.json
log "cx after XONG"
