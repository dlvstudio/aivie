#!/bin/bash
# TK_108 C4 lần 2: tự chờ "C3 XONG" → gom quỹ đạo C3 thành công → học TIẾP expert code c4e (trainGEN, thân + văn phòng
# + học thuật đóng băng) trên C3 + sách giáo khoa code + ôn C2 + hội thoại → gói aivie_p3_c4b → đo HumanEval / MBPP / tool-use + Excel + easy20 → HF.
# Dùng: (setsid nohup bash scripts/box_run_c4b.sh > /workspace/c4b_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q " C3 XONG · " $S; do sleep 120; done
pgrep -f "vllm serve" > /dev/null && { pkill -f "vllm serve"; sleep 30; }
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
python3 - <<'EOF' > /workspace/c4b_data.log 2>&1 || { log "LỖI C4 lần 2: gom quỹ đạo"; exit 1; }
import json
n = 0
with open("data/c3/train.jsonl", "w", encoding="utf-8") as f:
    for p in ["data/c3/traj_pilot_gemma.jsonl", "data/c3/traj_main.jsonl"]:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["ok"] and r["valid"]:
                f.write(json.dumps({"id": r["id"], "kind": r["kind"], "messages": r["messages"]}, ensure_ascii=False) + "\n"); n += 1
print(n)
EOF
log "C4 lần 2 bắt đầu: học tiếp expert code c4e trên $(cat /workspace/c4b_data.log) quỹ đạo C3 + sách giáo khoa code + ôn C2"
$PY -W ignore -u scripts/expert_office_1p3b.py trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/c4e_code.pt \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code \
   --mix "c3:0.45,c2oci:0.1,c2vi:0.05,c2dbg:0.05,tb:0.1,c2pred:0.03,c2sql:0.02,mt:0.1,fixfix:0.05,offfix:0.05" \
   --lr-old 1e-4 --steps 1500 --bs 4 --accum 4 --tag c4b > /workspace/c4b_train.log 2>&1 \
   || { log "LỖI C4 lần 2: học (c4b_train.log)"; exit 1; }
log "C4 lần 2 học xong · $(grep '\[gen\] sau' /workspace/c4b_train.log | cut -c1-300)"
$PY - <<'EOF' > /workspace/c4b_pack.log 2>&1 || { log "LỖI C4 lần 2: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/c4b_code.pt", map_location="cpu")
assert set(sd) == set(o["experts"]["code"]["state"]), "khác khoá expert code"
o["experts"]["code"]["state"] = {k: v.to(torch.bfloat16) for k, v in sd.items()}
o.setdefault("merged_from", []).append({"code": "c4b_code (TK_108 C4 lần 2: c4e + C3 quỹ đạo công cụ)"})
torch.save(o, "ckpt/aivie_p3_c4b_bf16.pt"); print("ok")
EOF
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
CK=ckpt/aivie_p3_c4b_bf16.pt
PIDS=""; port=8841
for s in humaneval mbpp agent office; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $port --no-rag > /workspace/c4b_ui_$s.log 2>&1 &
  PIDS="$PIDS $!"; port=$((port+1))
done
for port in 8841 8842 8843 8844; do
  W=0; until curl -s localhost:$port > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI C4 lần 2: UI $port"; kill $PIDS; exit 1; }; done
done
port=8841; JOBS=""
for s in humaneval mbpp agent; do
  ( $EV -u /workspace/c1/scripts/code_eval.py run --suite $s --url http://localhost:$port --tag c4b > /workspace/c4b_$s.log 2>&1 \
      && log "C4 lần 2 $s · $(grep '###' /workspace/c4b_$s.log | tail -1 | cut -c1-200)" || log "LỖI C4 lần 2 $s" ) &
  JOBS="$JOBS $!"; port=$((port+1))
done
( $EV -u scripts/office_probe_light.py http://localhost:8844 c4b > /workspace/c4b_office.log 2>&1 \
    && log "C4 lần 2 văn phòng · $(grep '###' /workspace/c4b_office.log)" || log "LỖI C4 lần 2 văn phòng" ) &
JOBS="$JOBS $!"
wait $JOBS
kill $PIDS
hf upload donaldluong/nn-aivie-2p6b-p3-snap $CK aivie_2p6b_p3_c4b_bf16.pt > /dev/null 2>&1 \
  && log "C4 lần 2 đã đẩy HF nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_c4b_bf16.pt"
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/c4b_code.pt expert_code/c4b_code.pt > /dev/null 2>&1
log "C4 lần 2 XONG — mốc: p3r1 HE 6,1 · MBPP 5,1 · tool 0 | c4e HE 7,3 · MBPP 9,3 · tool 0 · Excel 13/27 · easy20 16/20"
