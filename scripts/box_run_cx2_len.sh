#!/bin/bash
# TK_108: tool-use cx2 khung v3 = 47,5% với max_new 600 + ngữ cảnh UI 2000 — viết hàm hỏng vì edit_file bị CẮT ở 600 token,
# và quỹ đạo C3 train tới 4096 token nhưng UI cắt ở 2000 (bỏ lượt đầu = có thể mất đề). Đo tách 2 yếu tố (không train lại):
#   A) max_new 1024 · ctx 2000      B) max_new 1024 · ctx 4096
# Tự chờ " cx after XONG". Dùng: (setsid nohup bash scripts/box_run_cx2_len.sh > /workspace/cx2_len.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q " cx after XONG" $S; do sleep 60; done
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
cp scripts/code_agent.py scripts/serve_hybrid_ui.py scripts/expert_ffn_1p3b.py /workspace/c1/scripts/
CE=/workspace/c1/scripts/code_eval.py; CK=ckpt/aivie_p3_cx2_bf16.pt
log "cx2 đo độ dài: A) max_new 1024 · ctx 2000  B) max_new 1024 · ctx 4096 (khung v3; gốc 47,5% ở 600/2000)"
$PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port 8895 --no-rag --max-ctx 2000 > /workspace/cx2A_ui.log 2>&1 &
P1=$!
$PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port 8896 --no-rag --max-ctx 4096 > /workspace/cx2B_ui.log 2>&1 &
P2=$!
for p in 8895 8896; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI cx2 đo độ dài: UI $p"; kill $P1 $P2; exit 1; }; done
done
( $EV -u $CE run --suite agent --url http://localhost:8895 --tag cx2_n1024 --max-new 1024 > /workspace/cx2A_agent.log 2>&1 \
    && log "cx2 A (max_new 1024 · ctx 2000) · $(grep '###' /workspace/cx2A_agent.log | tail -1 | cut -c1-200)" || log "LỖI cx2 A" ) &
J1=$!
( $EV -u $CE run --suite agent --url http://localhost:8896 --tag cx2_n1024_c4k --max-new 1024 > /workspace/cx2B_agent.log 2>&1 \
    && log "cx2 B (max_new 1024 · ctx 4096) · $(grep '###' /workspace/cx2B_agent.log | tail -1 | cut -c1-200)" || log "LỖI cx2 B" ) &
J2=$!
wait $J1 $J2
log "cx2 đo độ dài: lỗi máy chủ A $(grep -c '\[lỗi /api/chat\]' /workspace/cx2A_ui.log) · B $(grep -c '\[lỗi /api/chat\]' /workspace/cx2B_ui.log)"
kill $P1 $P2
log "cx2 đo độ dài XONG"
