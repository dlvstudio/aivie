#!/bin/bash
# TK_108 (user duyệt 04/11, phương án a): NỐI LẠI ĐỌC ẢNH lên bản SFT-B — chạy khi GPU 0 xong 3 làn đo B (không để GPU rảnh đêm).
#   A và B cùng thân C6, chỉ khác SFT → khởi THẲNG từ phần đọc ảnh đã nối lên A (c6a_s2.pt), bỏ giai đoạn 1 + mốc gắn thẳng
#   0  chờ " SFT-B XONG" + 3 làn GPU 0 của box_run_c6b_eval.sh đo xong (máy chủ 8894-8896 đã tắt)
#   1  giai đoạn 2: 1000 bước, công thức như c6a (ngân sách 8000 đã thử trên A, cùng cỡ)
#   2  đo 6 bộ test — chia 2 GPU nếu GPU 1 đã rảnh (đo B xong) → giám khảo Gemma-31B chấm c6b
#   3  gói MỘT checkpoint → ckpt/c6b_vlm_bf16.pt
# Repo HF riêng tư có thể đầy → không đẩy checkpoint; Mac kéo về box_backup/. KHÔNG đẩy repo công khai.
# Dùng: (setsid nohup bash scripts/box_run_c6b_vlm.sh > /workspace/c6b_vlm_run.log 2>&1 < /dev/null &)
cd /workspace/nn-aivie
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PIP_CONSTRAINT
VLLM=/workspace/vllm_env
B=ckpt/c6_sft_b.pt
INIT=ckpt/vlm/c6a_s2.pt
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
A4="--shuf 2 --no-thumb --max-tiles 8 --max-len 3072"
V="python -W ignore -u scripts/vlm_1p3b.py"
D=/workspace/c6bvlm; mkdir -p $D
step() { [ -f $D/$1.done ]; }
done_() { touch $D/$1.done; log "đọc ảnh c6b: $1 xong · $2"; }
TESTS="screen nat chart ovqa news code"

# ── 0. chờ ── (KHÔNG ghi chuỗi chờ vào log — tránh tự kích hoạt như 03/11)
log "đọc ảnh c6b: chờ B học xong + GPU 0 đo B xong rồi nối đọc ảnh lên B"
lanes0_done() {
  grep -qaE '^[0-9-]+ [0-9:]+ ĐO c6sftb · ' $S && return 0
  for f in c7_1 c7_2 c7_3 office; do [ -s /workspace/c6sftb_$f.log ] || return 1; done
  ! pgrep -f "serve_hybrid_ui.py --ckpt ckpt/c6_sft_b.pt --port 889[456]" > /dev/null
}
until grep -qaE '^[0-9-]+ [0-9:]+  SFT-B XONG' $S && lanes0_done; do
  grep -qa "LỖI sau-C6: SFT-B" $S && { log "đọc ảnh c6b: DỪNG — SFT-B không xong"; exit 1; }
  sleep 120
done
sleep 60
[ -s $INIT ] || { log "LỖI đọc ảnh c6b: thiếu $INIT"; exit 1; }

# ── 1. giai đoạn 2 trên B ──
if ! step s2; then
  CUDA_VISIBLE_DEVICES=0 $V train --stage 2 --base $B $T $A4 --init $INIT --data data/vlm/sft5.jsonl --steps 1000 --bs 32 --lr 1e-4 \
     --lr-expert 2e-4 --tower-last 6 --lr-tower 2e-5 --lora 64 --lr-lora 1e-4 --warmup 50 --val-every 500 --save-every 500 \
     --tok-budget 8000 --out ckpt/vlm/c6b_s2.pt > /workspace/c6bvlm_s2.log 2>&1 || { log "LỖI đọc ảnh c6b: giai đoạn 2 (c6bvlm_s2.log)"; exit 1; }
  python -c "
import torch
sd = torch.load('ckpt/vlm/c6b_s2.pt', map_location='cpu')
out = {'proj': {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd['proj'].items()}, 'hid': sd['hid'], 'cfg': sd['cfg'],
       'expert': {k: v.to(torch.bfloat16) for k, v in sd['expert'].items()}, 'lora': sd.get('lora', {})}
if 'tower' in sd: out['tower'] = sd['tower']
torch.save(out, 'ckpt/vlm/c6b_vision_bf16.pt')"
  done_ s2 "$(grep 'val trước' /workspace/c6bvlm_s2.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/c6bvlm_s2.log)"
fi

# ── 2. đo (chia GPU 1 nếu đo B đã xong) + giám khảo ──
if ! step eval; then
  ev1() {  # $1 GPU  $2… bộ test
    local G=$1; shift
    for t in "$@"; do
      CUDA_VISIBLE_DEVICES=$G $V evalvlm --base $B $T --vision ckpt/vlm/c6b_s2.pt --device cuda --tag c6b --max-tiles 8 \
        --test data/vlm/test_$t.jsonl --n 200 > /workspace/c6bvlm_eval_$t.log 2>&1 \
        || log "LỖI đọc ảnh c6b: đo $t ($(grep -m1 Error /workspace/c6bvlm_eval_$t.log | cut -c1-120))"
    done
  }
  if pgrep -f box_run_c6b_eval.sh > /dev/null; then
    ev1 0 $TESTS
  else
    ev1 0 screen nat chart & P0=$!
    ev1 1 ovqa news code & P1=$!
    wait $P0 $P1
  fi
  CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
     --max-model-len 6144 --gpu-memory-utilization 0.85 --max-num-seqs 64 --port 8000 > /workspace/c6bvlm_vllm.log 2>&1 < /dev/null &
  VP=$!
  W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI đọc ảnh c6b: vLLM không lên"; break; }; done
  $V judgevlm --files "ckpt/vlm/eval_c6b_*.json" > /workspace/c6bvlm_judge.log 2>&1
  kill -- -$VP 2>/dev/null || kill $VP; sleep 15
  tar -czf /workspace/c6bvlm_results.tgz ckpt/vlm/eval_c6b_*.json /workspace/c6bvlm_*.log 2>/dev/null
  done_ eval "$(grep '###' /workspace/c6bvlm_judge.log | sed 's/### //' | tr '\n' '|' | cut -c1-900)"
fi

# ── 3. MỘT checkpoint ──
if ! step pack; then
  CUDA_VISIBLE_DEVICES=0 $V pack --base $B --vision ckpt/vlm/c6b_vision_bf16.pt $T --max-tiles 8 --out ckpt/c6b_vlm.pt > /workspace/c6bvlm_pack.log 2>&1 \
    || { log "LỖI đọc ảnh c6b: gói checkpoint"; exit 1; }
  python - <<'EOF' >> /workspace/c6bvlm_pack.log 2>&1
import torch
o = torch.load("ckpt/c6b_vlm.pt", map_location="cpu", weights_only=False)
o["state"] = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in o["state"].items()}
o.pop("opt", None)
torch.save(o, "ckpt/c6b_vlm_bf16.pt")
EOF
  rm -f ckpt/c6b_vlm.pt
  done_ pack "$(du -h ckpt/c6b_vlm_bf16.pt | cut -f1) · Mac kéo về box_backup/c6sft/"
fi
log "đọc ảnh c6b XONG — so dòng eval 'c6a' ↔ 'c6b' · cả 2 GPU rảnh chờ chọn bản"
