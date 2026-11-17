#!/bin/bash
# TK_108 PILOT "EXPERT CODE LỚN, TỰ MANG KIẾN THỨC" (user đồng ý 17/10): thân = ngôn ngữ chung + cầu nối, expert = kiến thức miền.
# Tự chờ " C5b XONG — " → 1) PRETRAIN RIÊNG expert code mới 24 điểm × cỡ ẩn 8192 (1,21 tỷ tham số; thân + expert văn phòng/
# học thuật đóng băng) trên code thô THÂN CHƯA THẤY (data/cx: OpenCoder annealing + fineweb-code + 15% tiếng Việt ôn) ${CX_HOURS:-10} giờ
# → 2) SFT bằng đúng trộn của C4 lần 3 (C2b đã chạy thật, sai→sửa, C3-R, C3, C2, hội thoại, văn phòng) 1500 bước
# → 3) gói aivie_p3_cx (thay expert code) → 4) đo HumanEval / MBPP / tool-use (khung v3 + khung cũ) / Excel + easy20 / MultiPL-E
# So với c4c (cùng SFT, expert nhỏ 0,3 tỷ không pretrain). Dùng: (setsid nohup bash scripts/box_run_cx_pilot.sh > /workspace/cx_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q " C5b XONG — " $S; do sleep 120; done
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
log "Pilot expert lớn bắt đầu: pretrain riêng expert code 24 điểm × 8192 (1,21 tỷ) · ${CX_HOURS:-10} giờ · $(cat data/cx/manifest.json | tr -d '\n ' | cut -c1-200)"
$PY -W ignore -u scripts/expert_office_1p3b.py trainCPTX --base ckpt/p3_sft.pt --code "" \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --pos every4 --raw-hid 8192 \
   --cx-mix "opc_algo:0.25,opc_snip:0.2,opc_files:0.25,web:0.15,vi:0.15" --seq 2048 --bs 4 --accum 4 --lr 3e-4 \
   --steps 1000000 --hours ${CX_HOURS:-10} --eval-every 1000 --tag cxpt > /workspace/cx_pt.log 2>&1 \
   || { log "LỖI pilot: pretrain expert (cx_pt.log)"; [ -s ckpt/expert_code/cxpt_code.pt ] || exit 1; }
log "Pilot pretrain xong · $(grep '###' /workspace/cx_pt.log | sed 's/### //' | cut -c1-300)"
MIX="c2bV:0.22,c2bA:0.11,c2bP:0.11,c2bF:0.08,c3r:0.12,c3:0.07,c2oci:0.04,c2dbg2:0.04,tb:0.04,mt:0.08,fixfix:0.05,offfix:0.04"
# bs=1 accum=16: expert lớn 1,2B OOM với bs=4 trên H100 80G (đã gặp 17/10) — giữ cùng tok/bước
$PY -W ignore -u scripts/expert_office_1p3b.py trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/cxpt_code.pt \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code --mix "$MIX" \
   --lr-old 1e-4 --steps 1500 --bs 1 --accum 16 --tag cx > /workspace/cx_sft.log 2>&1 \
   || { log "LỖI pilot: SFT (cx_sft.log)"; exit 1; }
log "Pilot SFT xong · $(grep '\[gen\] sau' /workspace/cx_sft.log | cut -c1-250)"
$PY - <<'PY' > /workspace/cx_pack.log 2>&1 || { log "LỖI pilot: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/cx_code.pt", map_location="cpu")
pos = sorted({int(k.split(".")[0]) for k in sd})
o["experts"]["code"] = {"state": {k: v.to(torch.bfloat16) for k, v in sd.items()}, "gated": True,
                        "hid": sd[f"{pos[0]}.w3.weight"].shape[1], "pos": pos}
o.setdefault("merged_from", []).append({"code": "cx_code (TK_108 pilot: expert code LỚN 24 điểm, pretrain riêng trên code thô + SFT C4 lần 3)"})
torch.save(o, "ckpt/aivie_p3_cx_bf16.pt"); print("ok", len(pos))
PY
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
cp scripts/code_agent.py scripts/serve_hybrid_ui.py scripts/expert_ffn_1p3b.py /workspace/c1/scripts/   # bản đọc vị trí cắm từ trọng số
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
CK=ckpt/aivie_p3_cx_bf16.pt; CE=/workspace/c1/scripts/code_eval.py
PIDS=""
for p in 8881 8882 8883 8884; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag > /workspace/cx_ui_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in 8881 8882 8883 8884; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI pilot: UI $p"; kill $PIDS; exit 1; }; done
done
log "Pilot: $(grep '\[expert\] code' /workspace/cx_ui_8881.log)"
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
kill $PIDS
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/cxpt_code.pt expert_code/cxpt_code.pt > /dev/null 2>&1
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/cx_code.pt expert_code/cx_code.pt > /dev/null 2>&1
hf upload donaldluong/nn-aivie-2p6b-p3-snap $CK aivie_2p6b_p3_cx_bf16.pt > /dev/null 2>&1 && log "Pilot đã đẩy HF aivie_2p6b_p3_cx_bf16.pt"
log "Pilot expert lớn XONG — so với c4c (expert nhỏ 0,3 tỷ, cùng SFT): HE 11,0 · MBPP 9,7 · tool cũ 35 · Excel 16/27 · easy20 15/20 · ML TB 2,4"
