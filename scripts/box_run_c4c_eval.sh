#!/bin/bash
# TK_108 C4 lần 3 — ĐO LẠI (lần đo đầu 17/10 03:26 hỏng: fork bomb bash từ bài MultiPL-E làm cạn tiến trình box).
# Đã sửa khung chạy (diệt cả nhóm tiến trình kiểu đóng băng→diệt, trần tiến trình sbx). Giữ kết quả hợp lệ trước sự cố,
# bỏ phần có thể bị ảnh hưởng, đo tiếp. Ghi " C4 lần 3 đo lại XONG — " → C5a chờ dòng này.
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
O=out/code_eval
for f in humaneval_c4c mbpp_c4c; do [ -s $O/$f.jsonl ] && head -n -5 $O/$f.jsonl > $O/$f.tmp && mv $O/$f.tmp $O/$f.jsonl; done
rm -f $O/agent_c4c.jsonl $O/agent_c4c_khungcu.jsonl $O/agent_c4b_khungmoi.jsonl $O/ml_c4b-2.jsonl $O/ml_c4c-1.jsonl $O/ml_c4c-2.jsonl
log "C4 lần 3: đo lại (lần đầu hỏng vì fork bomb bash làm cạn tiến trình box; đã sửa khung chạy) · giữ HumanEval $(wc -l < $O/humaneval_c4c.jsonl) · MBPP $(wc -l < $O/mbpp_c4c.jsonl) · MultiPL-E mốc c4b phần 1"
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
log "C4 lần 3 đo lại XONG — so với c4b (HumanEval / MBPP / tool-use / Excel 13/27 / easy20 16/20 + MultiPL-E mốc c4b ở trên)"
