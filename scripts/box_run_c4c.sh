#!/bin/bash
# TK_108 C4 lần 3 (code ĐA NGÔN NGỮ): tự chờ dòng " C2b B XONG — " → gộp dữ liệu C2b (c2b_mix.py) → học TIẾP expert code c4b
# (trainGEN; thân + văn phòng + học thuật đóng băng) trên: C2b đã chạy thật (B) + C2b kiểm cú pháp (A) + C2b+ hiện đại
# + quỹ đạo PHỤC HỒI C3-R + ôn C3 + C2 + sách giáo khoa + hội thoại/văn phòng → gói aivie_p3_c4c → đo:
#   HumanEval / MBPP / tool-use / Excel + easy20 (không được tụt) + MultiPL-E 7 ngôn ngữ × 30 bài cho CẢ c4b (mốc) và c4c → HF.
#   tool-use đo 3 lần: c4c khung MỚI (chống lặp) · c4c khung CŨ (so thẳng với c4b 32,5%) · c4b khung MỚI (tách phần nhờ khung)
# Dùng: (setsid nohup bash scripts/box_run_c4c.sh > /workspace/c4c_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q " C3-R XONG — " $S; do sleep 120; done       # sau C2b B → C3-R (box_run_c3r.sh luôn ghi dòng XONG, kể cả khi lỗi)
pgrep -f "vllm serve" > /dev/null && { pkill -f "vllm serve"; sleep 30; }
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
$EV scripts/c2b_mix.py > /workspace/c4c_mix.log 2>&1 || { log "LỖI C4 lần 3: gộp dữ liệu"; exit 1; }
$EV scripts/c2b_fixmix.py >> /workspace/c4c_mix.log 2>&1 || log "C4 lần 3: không dựng được mix_fix (bỏ khoá c2bF)"
log "C4 lần 3 bắt đầu: $(grep '###' /workspace/c4c_mix.log | sed 's/### //' | tr '\n' ' ')"
MIX="c2bV:0.22,c2bA:0.11,c2bP:0.11,c2bF:0.08,c3r:0.12,c3:0.07,c2oci:0.04,c2dbg2:0.04,tb:0.04,mt:0.08,fixfix:0.05,offfix:0.04"
[ -s data/c2/debug_v2.jsonl ] || MIX=${MIX/c2dbg2/c2dbg}
MIX=$($PY - "$MIX" <<'PY'                         # bỏ khoá thiếu dữ liệu (C3-R / phần B / mix_fix hỏng) rồi chia lại tỉ lệ
import os, sys, re
sys.path.insert(0, "scripts")
src = open("scripts/expert_office_1p3b.py", encoding="utf-8").read()
D = dict(re.findall(r'"(\w+)": "(data/[^"]+)"', src))
kv = [(k, float(v)) for k, v in (x.split(":") for x in sys.argv[1].split(","))]
kv = [(k, v) for k, v in kv if k in D and os.path.exists(D[k]) and os.path.getsize(D[k]) > 0]
t = sum(v for _, v in kv)
print(",".join(f"{k}:{v / t:.3f}" for k, v in kv))
PY
)
$PY -W ignore -u scripts/expert_office_1p3b.py trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/c4b_code.pt \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code --mix "$MIX" \
   --lr-old 1e-4 --steps 2500 --bs 4 --accum 4 --tag c4c > /workspace/c4c_train.log 2>&1 \
   || { log "LỖI C4 lần 3: học (c4c_train.log)"; exit 1; }
log "C4 lần 3 học xong · trộn $MIX · $(grep '\[gen\] sau' /workspace/c4c_train.log | cut -c1-250)"
$PY - <<'PY' > /workspace/c4c_pack.log 2>&1 || { log "LỖI C4 lần 3: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/c4c_code.pt", map_location="cpu")
assert set(sd) == set(o["experts"]["code"]["state"]), "khác khoá expert code"
o["experts"]["code"]["state"] = {k: v.to(torch.bfloat16) for k, v in sd.items()}
o.setdefault("merged_from", []).append({"code": "c4c_code (TK_108 C4 lần 3: c4b + code đa ngôn ngữ C2b)"})
torch.save(o, "ckpt/aivie_p3_c4c_bf16.pt"); print("ok")
PY
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
# 4 máy chủ (mỗi cái ~18 GB GPU — 8 cái sẽ tràn): 3 × c4c (8851–8853) + 1 × c4b mốc (8857); việc xếp nối tiếp trên từng máy
#   8851 c4c: humaneval → văn phòng
#   8852 c4c: mbpp → MultiPL-E php,go,sh
#   8853 c4c: agent khung mới → agent khung cũ → MultiPL-E js,ts,java,cs
#   8857 c4b: MultiPL-E 7 ngôn ngữ (mốc) → agent khung mới (tách phần nhờ khung)
PIDS=""
up() { $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $1 --port $2 --no-rag > /workspace/c4c_ui_$2.log 2>&1 & PIDS="$PIDS $!"; }
for p in 8851 8852 8853; do up ckpt/aivie_p3_c4c_bf16.pt $p; done
up ckpt/aivie_p3_c4b_bf16.pt 8857
for p in 8851 8852 8853 8857; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI C4 lần 3: UI $p"; kill $PIDS; exit 1; }; done
done
cp scripts/code_agent.py /workspace/c1/scripts/code_agent.py              # khung công cụ v2 (chống lặp, báo lệch TAB)
CE=/workspace/c1/scripts/code_eval.py
ml() {   # $1 tag  $2 cổng  $3 ngôn ngữ
  $EV -u scripts/code_eval_ml.py run --url http://localhost:$2 --tag $1 --langs $3 --n 30 --workers 1 > /workspace/c4c_ml_$1.log 2>&1
}
rep() { $EV scripts/code_eval_ml.py report --tags $1 | cut -c1-220; }
JOBS=""
( $EV -u $CE run --suite humaneval --url http://localhost:8851 --tag c4c > /workspace/c4c_humaneval.log 2>&1 \
    && log "C4 lần 3 humaneval · $(grep '###' /workspace/c4c_humaneval.log | tail -1 | cut -c1-200)" || log "LỖI C4 lần 3 humaneval"
  $EV -u scripts/office_probe_light.py http://localhost:8851 c4c > /workspace/c4c_office.log 2>&1 \
    && log "C4 lần 3 văn phòng · $(grep '###' /workspace/c4c_office.log)" || log "LỖI C4 lần 3 văn phòng" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite mbpp --url http://localhost:8852 --tag c4c > /workspace/c4c_mbpp.log 2>&1 \
    && log "C4 lần 3 mbpp · $(grep '###' /workspace/c4c_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI C4 lần 3 mbpp"
  ml c4c-2 8852 php,go,sh ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite agent --url http://localhost:8853 --tag c4c > /workspace/c4c_agent.log 2>&1 \
    && log "C4 lần 3 agent (khung mới) · $(grep '###' /workspace/c4c_agent.log | tail -1 | cut -c1-200)" || log "LỖI C4 lần 3 agent"
  AIVIE_TOOL_GUARD=0 $EV -u $CE run --suite agent --url http://localhost:8853 --tag c4c_khungcu > /workspace/c4c_agent0.log 2>&1 \
    && log "C4 lần 3 agent (khung cũ, so với c4b 32,5%) · $(grep '###' /workspace/c4c_agent0.log | tail -1 | cut -c1-200)"
  ml c4c-1 8853 js,ts,java,cs ) &
JOBS="$JOBS $!"
( ml c4b-1 8857 js,ts,java,cs; ml c4b-2 8857 php,go,sh
  log "C4 lần 3 MultiPL-E mốc c4b · $(rep c4b-1+c4b-2)"
  $EV -u $CE run --suite agent --url http://localhost:8857 --tag c4b_khungmoi > /workspace/c4b_agent_new.log 2>&1 \
    && log "C4 lần 3 agent c4b khung mới (tách phần nhờ khung) · $(grep '###' /workspace/c4b_agent_new.log | tail -1 | cut -c1-200)" ) &
JOBS="$JOBS $!"
wait $JOBS
log "C4 lần 3 MultiPL-E c4c · $(rep c4c-1+c4c-2)"
kill $PIDS
hf upload donaldluong/nn-aivie-2p6b-p3-snap ckpt/aivie_p3_c4c_bf16.pt aivie_2p6b_p3_c4c_bf16.pt > /dev/null 2>&1 \
  && log "C4 lần 3 đã đẩy HF nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_c4c_bf16.pt"
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/c4c_code.pt expert_code/c4c_code.pt > /dev/null 2>&1
log "C4 lần 3 XONG — so với c4b (HumanEval / MBPP / tool-use / Excel 13/27 / easy20 16/20 + MultiPL-E mốc c4b ở trên)"
