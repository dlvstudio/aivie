#!/bin/bash
# TK_108 pilot bước 2 — SFT BÙ TOOL-USE cho expert code LỚN (user đồng ý 18/10). Pilot cx: HE 13,4 · MBPP 17,1 · ML TB 5,7
# (hơn c4c) nhưng tool-use khung cũ 10% (c4c 35%), 71% lượt không gọi công cụ → học tiếp từ cx_code 600 bước với
# ~55% dữ liệu tool-use (C3-R + C3), giữ code đã kiểm + văn phòng chống quên, LR 5e-5 (thấp hơn 1e-4 để giữ phần code)
# → gói aivie_p3_cx2 → đo tool-use (khung v3 + khung cũ) / HumanEval / MBPP / Excel + easy20.
# Dùng: (setsid nohup bash scripts/box_run_cx2.sh > /workspace/cx2_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
MIX="c3r:0.30,c3:0.25,c2bV:0.12,c2bF:0.06,c2bA:0.05,c2oci:0.04,c2dbg2:0.04,mt:0.06,offfix:0.04,fixfix:0.04"
log "cx2 bắt đầu: SFT bù tool-use từ expert lớn cx · 600 bước · lr 5e-5 · $MIX"
$PY -W ignore -u scripts/expert_office_1p3b.py trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/cx_code.pt \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code --mix "$MIX" \
   --lr-old 5e-5 --steps 600 --bs 1 --accum 16 --tag cx2 > /workspace/cx2_sft.log 2>&1 \
   || { log "LỖI cx2: SFT (cx2_sft.log)"; exit 1; }
log "cx2 SFT xong · $(grep '\[gen\] sau' /workspace/cx2_sft.log | cut -c1-250)"
$PY - <<'PY' > /workspace/cx2_pack.log 2>&1 || { log "LỖI cx2: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/cx2_code.pt", map_location="cpu")
pos = sorted({int(k.split(".")[0]) for k in sd})
o["experts"]["code"] = {"state": {k: v.to(torch.bfloat16) for k, v in sd.items()}, "gated": True,
                        "hid": sd[f"{pos[0]}.w3.weight"].shape[1], "pos": pos}
o.setdefault("merged_from", []).append({"code": "cx2_code (TK_108 pilot: expert code LỚN + SFT bù tool-use 600 bước)"})
torch.save(o, "ckpt/aivie_p3_cx2_bf16.pt"); print("ok", len(pos))
PY
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
cp scripts/code_agent.py scripts/serve_hybrid_ui.py scripts/expert_ffn_1p3b.py /workspace/c1/scripts/
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
CK=ckpt/aivie_p3_cx2_bf16.pt; CE=/workspace/c1/scripts/code_eval.py
PIDS=""
for p in 8891 8892 8893 8894; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag > /workspace/cx2_ui_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in 8891 8892 8893 8894; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI cx2: UI $p"; kill $PIDS; exit 1; }; done
done
JOBS=""
( $EV -u $CE run --suite agent --url http://localhost:8891 --tag cx2 > /workspace/cx2_agent.log 2>&1 \
    && log "cx2 agent (khung v3; cx 22,5) · $(grep '###' /workspace/cx2_agent.log | tail -1 | cut -c1-200)" || log "LỖI cx2 agent" ) &
JOBS="$JOBS $!"
( AIVIE_TOOL_GUARD=0 $EV -u $CE run --suite agent --url http://localhost:8892 --tag cx2_khungcu > /workspace/cx2_agent0.log 2>&1 \
    && log "cx2 agent (khung cũ; cx 10, c4c 35) · $(grep '###' /workspace/cx2_agent0.log | tail -1 | cut -c1-200)" || log "LỖI cx2 agent khung cũ" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite humaneval --url http://localhost:8893 --tag cx2 > /workspace/cx2_humaneval.log 2>&1 \
    && log "cx2 humaneval (cx 13,4) · $(grep '###' /workspace/cx2_humaneval.log | tail -1 | cut -c1-200)" || log "LỖI cx2 humaneval"
  $EV -u scripts/office_probe_light.py http://localhost:8893 cx2 > /workspace/cx2_office.log 2>&1 \
    && log "cx2 văn phòng (cx Excel 14 · easy20 16) · $(grep '###' /workspace/cx2_office.log)" || log "LỖI cx2 văn phòng" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite mbpp --url http://localhost:8894 --tag cx2 > /workspace/cx2_mbpp.log 2>&1 \
    && log "cx2 mbpp (cx 17,1) · $(grep '###' /workspace/cx2_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI cx2 mbpp" ) &
JOBS="$JOBS $!"
wait $JOBS
kill $PIDS
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/cx2_code.pt expert_code/cx2_code.pt > /dev/null 2>&1
log "cx2 XONG — so với cx (tool cũ 10 · v3 22,5 · HE 13,4 · MBPP 17,1) và c4c (tool cũ 35)"
