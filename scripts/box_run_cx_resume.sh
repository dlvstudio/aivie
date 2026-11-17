#!/bin/bash
# Resume CX pilot SAU pretrain (SFT OOM với bs=4). Dùng bs=1 accum=16 (cùng tok/bước).
# Không đụng lại pretrain. Ghi status → /workspace/p3_status.log
set -euo pipefail
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AIVIE_SANDBOX_USER=sbx
PY=/workspace/vlmenv/bin/python
EV=/workspace/evalenv/bin/python

[ -s ckpt/expert_code/cxpt_code.pt ] || { log "LỖI resume: thiếu cxpt_code.pt"; exit 1; }
grep -q "Pilot pretrain xong" $S || { log "LỖI resume: chưa thấy Pilot pretrain xong"; exit 1; }

# Nếu đã có cx_code.pt và aivie_p3_cx → bỏ qua SFT, chỉ đo
if [ -s ckpt/expert_code/cx_code.pt ] && [ -s ckpt/aivie_p3_cx_bf16.pt ]; then
  log "Resume: đã có cx_code + aivie_p3_cx — bỏ qua SFT, vào đo"
else
  MIX="c2bV:0.22,c2bA:0.11,c2bP:0.11,c2bF:0.08,c3r:0.12,c3:0.07,c2oci:0.04,c2dbg2:0.04,tb:0.04,mt:0.08,fixfix:0.05,offfix:0.04"
  log "Resume SFT: expert lớn cxpt · bs=1 accum=16 (tránh OOM bs=4) · 1500 bước"
  $PY -W ignore -u scripts/expert_office_1p3b.py trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/cxpt_code.pt \
     --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code --mix "$MIX" \
     --lr-old 1e-4 --steps 1500 --bs 1 --accum 16 --tag cx > /workspace/cx_sft.log 2>&1 \
     || { log "LỖI pilot: SFT resume (cx_sft.log)"; exit 1; }
  log "Pilot SFT xong · $(grep '\[gen\] sau' /workspace/cx_sft.log | cut -c1-250)"

  $PY - <<'PY' > /workspace/cx_pack.log 2>&1 || { log "LỖI pilot: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/cx_code.pt", map_location="cpu")
pos = sorted({int(k.split(".")[0]) for k in sd})
o["experts"]["code"] = {"state": {k: v.to(torch.bfloat16) for k, v in sd.items()}, "gated": True,
                        "hid": sd[f"{pos[0]}.w3.weight"].shape[1], "pos": pos}
o.setdefault("merged_from", []).append({"code": "cx_code (TK_108 pilot: expert code LỚN 24 điểm, pretrain riêng trên code thô + SFT C4 lần 3; SFT bs=1/accum=16 sau OOM bs=4)"})
torch.save(o, "ckpt/aivie_p3_cx_bf16.pt"); print("ok", len(pos))
PY
  log "Pilot gói xong aivie_p3_cx_bf16.pt"
fi

for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
cp scripts/code_agent.py scripts/serve_hybrid_ui.py scripts/expert_ffn_1p3b.py /workspace/c1/scripts/
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
CK=ckpt/aivie_p3_cx_bf16.pt; CE=/workspace/c1/scripts/code_eval.py
# tắt UI cũ nếu còn
pkill -f 'serve_hybrid_ui.py --ckpt ckpt/aivie_p3_cx' 2>/dev/null || true
pkill -f 'serve_hybrid_ui.py' 2>/dev/null || true
sleep 2
PIDS=""
for p in 8881 8882 8883 8884; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag > /workspace/cx_ui_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in 8881 8882 8883 8884; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI pilot: UI $p"; kill $PIDS; exit 1; }; done
done
log "Pilot: $(grep '\[expert\] code' /workspace/cx_ui_8881.log | head -1)"
ml() { $EV -u scripts/code_eval_ml.py run --url http://localhost:$2 --tag $1 --langs $3 --n 30 --workers 1 > /workspace/cx_ml_$1.log 2>&1; }
JOBS=""
( $EV -u $CE run --suite humaneval --url http://localhost:8881 --tag cx > /workspace/cx_humaneval.log 2>&1 \
    && log "Pilot humaneval · $(grep '###' /workspace/cx_humaneval.log | tail -1 | cut -c1-200)" || log "LỖI pilot humaneval"
  $EV -u scripts/office_probe_light.py http://localhost:8881 cx > /workspace/cx_office.log 2>&1 \
    && log "Pilot văn phòng · $(grep '###' /workspace/cx_office.log)" || log "LỖI pilot văn phòng" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite mbpp --url http://localhost:8882 --tag cx > /workspace/cx_mbpp.log 2>&1 \
    && log "Pilot mbpp · $(grep '###' /workspace/cx_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI pilot mbpp" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite agent --url http://localhost:8883 --tag cx > /workspace/cx_agent.log 2>&1 \
    && log "Pilot agent (khung v3) · $(grep '###' /workspace/cx_agent.log | tail -1 | cut -c1-200)" || log "LỖI pilot agent"
  AIVIE_TOOL_GUARD=0 $EV -u $CE run --suite agent --url http://localhost:8883 --tag cx_khungcu > /workspace/cx_agent0.log 2>&1 \
    && log "Pilot agent (khung cũ, so với c4c 35%) · $(grep '###' /workspace/cx_agent0.log | tail -1 | cut -c1-200)"
  ml cx-1 8883 js,ts,java,cs ) &
JOBS="$JOBS $!"
( ml cx-2 8884 php,go,sh ) &
JOBS="$JOBS $!"
wait $JOBS
log "Pilot MultiPL-E · $($EV scripts/code_eval_ml.py report --tags cx-1+cx-2 | cut -c1-220)"
kill $PIDS 2>/dev/null || true
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/cxpt_code.pt expert_code/cxpt_code.pt > /dev/null 2>&1 || true
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/cx_code.pt expert_code/cx_code.pt > /dev/null 2>&1 || true
hf upload donaldluong/nn-aivie-2p6b-p3-snap $CK aivie_2p6b_p3_cx_bf16.pt > /dev/null 2>&1 \
  && log "Pilot đã đẩy HF aivie_2p6b_p3_cx_bf16.pt" || log "LỖI pilot: đẩy HF"
log "Pilot expert lớn XONG — so với c4c (expert nhỏ 0,3 tỷ, cùng SFT): HE 11,0 · MBPP 9,7 · tool cũ 35 · Excel 16/27 · easy20 15/20 · ML TB 2,4"
