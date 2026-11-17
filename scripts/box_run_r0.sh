#!/bin/bash
# TK_108 R0 (user duyệt 05/11): ĐO RAG TRƯỚC KHI LÀM — tự chạy sau "vòng thầy c6l1 XONG", đo cả B lẫn c6l1.
#   1  thầy Gemma-31B (vLLM, GPU 0) dựng 350 câu sự kiện sạch từ viwiki (single / extract / multi / fake)
#   2  tắt vLLM → 6 máy chủ: B ×3 (GPU 1) · c6l1 ×3 (GPU 0) — mỗi bản trả lời 3 cách: hỏi chay · RAG BM25F · có sẵn đoạn đúng
#   3  bật lại vLLM → thầy chấm → bảng so (dòng "ĐO R0 · …")
# Dùng: (setsid nohup bash scripts/box_run_r0.sh > /workspace/r0_run.log 2>&1 < /dev/null &)
cd /workspace/nn-aivie
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM=/workspace/vllm_env
R="python -u scripts/rag_r0.py"
D=/workspace/r0; mkdir -p $D
step() { [ -f $D/$1.done ]; }
vllm_on() {
  curl -s localhost:8000/v1/models | grep -q gemma4-31b && return
  CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
    --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 64 --port 8000 > /workspace/r0_vllm.log 2>&1 < /dev/null &
  local W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI R0: vLLM không lên"; exit 1; }; done
}
vllm_off() { for p in $(pgrep -f "vllm serve"); do kill $p; done; sleep 30; }
ui() {  # $1 GPU $2 cổng $3 ckpt
  CUDA_VISIBLE_DEVICES=$1 setsid nohup python -W ignore -u scripts/serve_hybrid_ui.py --ckpt $3 --port $2 --no-rag --max-ctx 4096 \
    > /workspace/r0_ui_$2.log 2>&1 < /dev/null &
  local W=0; until curl -s localhost:$2 > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI R0: UI $2 không lên"; return 1; }; done
  return 0   # until trả mã của lệnh cuối trong thân ([ W -gt 150 ] = 1) → không có dòng này ui "lỗi" mỗi khi phải chờ
}
ui_off() { for p in $(pgrep -f "serve_hybrid_ui.py --ckpt"); do kill $p; done; sleep 10; }

# chờ vòng thầy xong hẳn (mẫu khớp ĐẦU DÒNG — không tự kích hoạt bởi dòng log của chính script này)
until grep -qaE '^[0-9-]+ [0-9:]+ vòng thầy c6l1 XONG' $S; do
  grep -qa "LỖI vòng thầy c6l1: điều phối dừng 6 lần" $S && { log "R0: DỪNG — vòng thầy không xong"; exit 1; }
  sleep 120
done
# 05/11: bước đo chữ c6l1 hỏng (ui trả mã 1) → đo lại tay trên cổng 8891-8893 → chờ xong mới dùng các cổng đó
until grep -qaE '^[0-9-]+ [0-9:]+ ĐO c6l1 lại ' $S; do sleep 120; done
sleep 60
log "R0 bắt đầu: đo RAG trên B và c6l1 (350 câu sự kiện × 3 cách)"
[ -s ckpt/c6_loop1.pt ] && TAGS="c6b c6l1" || TAGS="c6b"

if ! step build; then
  vllm_on
  $R build --n 350 --workers 32 > /workspace/r0_build.log 2>&1 || { log "LỖI R0: dựng câu hỏi (r0_build.log)"; exit 1; }
  touch $D/build.done; log "R0 build xong · $(grep '###' /workspace/r0_build.log) · $(grep -E '^\[r0\] (single|extract|multi|fake):' /workspace/r0_build.log | tr '\n' ' ')"
fi
if ! step eval; then
  vllm_off
  ui 1 8891 ckpt/c6_sft_b.pt && ui 1 8892 ckpt/c6_sft_b.pt && ui 1 8893 ckpt/c6_sft_b.pt || exit 1
  $R eval --url http://localhost:8891,http://localhost:8892,http://localhost:8893 --tag c6b > /workspace/r0_eval_c6b.log 2>&1 & E1=$!
  if [ "$TAGS" != "c6b" ]; then
    ui 0 8894 ckpt/c6_loop1.pt && ui 0 8895 ckpt/c6_loop1.pt && ui 0 8896 ckpt/c6_loop1.pt || exit 1
    $R eval --url http://localhost:8894,http://localhost:8895,http://localhost:8896 --tag c6l1 > /workspace/r0_eval_c6l1.log 2>&1 & E2=$!
  fi
  wait $E1 $E2
  ui_off
  touch $D/eval.done
fi
if ! step judge; then
  vllm_on
  for t in $TAGS; do $R judge --tag $t --workers 32 > /workspace/r0_judge_$t.log 2>&1; done
  vllm_off
  touch $D/judge.done
fi
$R report --tags $(echo $TAGS | tr ' ' ',') > /workspace/r0_report.txt 2>&1
tar czf /workspace/r0_ketqua.tgz data/rag/r0_eval.jsonl out/rag_r0 /workspace/r0_*.log /workspace/r0_report.txt 2>/dev/null
log "ĐO R0 · $(head -n -1 /workspace/r0_report.txt | tail -n +2 | tr -s ' ' | tr '\n' '|' | cut -c1-700)"
log "R0 XONG — bảng đầy đủ /workspace/r0_report.txt · GPU rảnh chờ chọn R1/R2"
