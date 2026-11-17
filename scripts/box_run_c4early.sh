#!/bin/bash
# TK_108 C4 SỚM: tự chờ "C2 XONG" → học EXPERT CODE MỚI 2 giai đoạn (trainGEN2) trên dữ liệu C2 đã kiểm bằng chạy thật
#   thân p3_sft + expert văn phòng (p3r1xl) + học thuật (p3r1) ĐÓNG BĂNG · A: 5000 bước cổng mở, chỉ dữ liệu code
#   B: 600 bước học cổng trên dữ liệu trộn (code + hội thoại + văn phòng) → gói aivie_p3_c4e = p3r1xl với expert code mới
#   → đo lại bộ C1 (HumanEval / MBPP / tool-use) + Excel 27 + easy20 (kiểm không làm hỏng phần khác) → đẩy HF.
# Dùng: (setsid nohup bash scripts/box_run_c4early.sh > /workspace/c4e_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q "C2 XONG" $S; do sleep 120; done
pgrep -f "vllm serve" > /dev/null && { pkill -f "vllm serve"; sleep 30; }
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
log "C4 sớm bắt đầu: expert code MỚI 2 giai đoạn trên dữ liệu C2 (thân + văn phòng + học thuật đóng băng)"
$PY -W ignore -u scripts/expert_office_1p3b.py trainGEN2 --base ckpt/p3_sft.pt --code "" \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code \
   --domain "code:c2oci+c2vi+c2dbg+c2pred+c2sql" \
   --mix "c2oci:0.15,c2vi:0.15,c2dbg:0.1,c2pred:0.05,c2sql:0.05,mt:0.15,fixfix:0.15,offfix:0.1,sgk:0.05,calib:0.05" \
   --steps 5000 --gate-steps 600 --bs 4 --accum 4 --lr 3e-4 --max-len 1535 --tag c4e > /workspace/c4e_train.log 2>&1 \
   || { log "LỖI C4 sớm: học (c4e_train.log)"; exit 1; }
log "C4 sớm học xong · $(grep '\[gen2\] sau B' /workspace/c4e_train.log | cut -c1-300)"
$PY - <<'EOF' > /workspace/c4e_pack.log 2>&1 || { log "LỖI C4 sớm: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/c4e_code.pt", map_location="cpu")
assert set(sd) == set(o["experts"]["code"]["state"]), "khác khoá expert code"
o["experts"]["code"]["state"] = {k: v.to(torch.bfloat16) for k, v in sd.items()}
o.setdefault("merged_from", []).append({"code": "c4e_code (TK_108 C4 sớm: trainGEN2 trên C2)"})
torch.save(o, "ckpt/aivie_p3_c4e_bf16.pt"); print("ok")
EOF
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
CK=ckpt/aivie_p3_c4e_bf16.pt
PIDS=""; port=8831
for s in humaneval mbpp agent office; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $port --no-rag > /workspace/c4e_ui_$s.log 2>&1 &
  PIDS="$PIDS $!"; port=$((port+1))
done
for port in 8831 8832 8833 8834; do
  W=0; until curl -s localhost:$port > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI C4 sớm: UI $port"; kill $PIDS; exit 1; }; done
done
port=8831
for s in humaneval mbpp agent; do
  ( $EV -u /workspace/c1/scripts/code_eval.py run --suite $s --url http://localhost:$port --tag c4e > /workspace/c4e_$s.log 2>&1 \
      && log "C4 sớm $s · $(grep '###' /workspace/c4e_$s.log | tail -1 | cut -c1-200)" || log "LỖI C4 sớm $s" ) &
  port=$((port+1))
done
( $EV -u scripts/office_probe_light.py http://localhost:8834 c4e > /workspace/c4e_office.log 2>&1 \
    && log "C4 sớm văn phòng · $(grep '###' /workspace/c4e_office.log)" || log "LỖI C4 sớm văn phòng" ) &
sleep 30
while pgrep -f "code_eval.py run|office_probe_light" > /dev/null; do sleep 60; done
kill $PIDS
hf upload donaldluong/nn-aivie-2p6b-p3-snap $CK aivie_2p6b_p3_c4e_bf16.pt > /dev/null 2>&1 \
  && log "C4 sớm đã đẩy HF nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_c4e_bf16.pt"
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/c4e_code.pt expert_code/c4e_code.pt > /dev/null 2>&1
log "C4 sớm XONG — so với C1 (p3r1): HumanEval 6,1% · MBPP 5,1% · tool-use 0% · Excel p3r1xl 13/27 · easy20 16/20"
