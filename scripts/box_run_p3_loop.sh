#!/bin/bash
# TK_107 P3.3' · VÒNG THẦY CHUNG chữ + ảnh trên aivie 2,5B — box 1×H100 P3, chạy SAU box_run_p3_vlm.sh (chờ "P3.4 XONG").
#   1  thầy Gemma-4-31B (vLLM, nhìn được ảnh) chat NHIỀU LƯỢT với aivie (3 tiến trình, bf16): ~30% hội thoại có ảnh (35% là ảnh
#      CODE), phần chữ có ~12% hội thoại an toàn / danh tính (luật chấm riêng) → chấm 0–10 + bản sửa (vlm_loop.py run)
#   2  tách: lượt chữ → buildV5 → SFT (trainGEN, trộn ôn) + DPO cho 3 expert chữ; lượt có ảnh → fix_r1.jsonl
#   3  gộp thân + 3 expert MỚI → train tiếp phần đọc ảnh trên bản sửa của thầy (+ ôn 20K mẫu sft5) — đọc ảnh học TRÊN expert mới
#      → không lệch nhau
#   4  chấm lại cả chữ (eval_v1, easy20, văn phòng, SGK, toán) lẫn ảnh (6 bộ, có ảnh code) → gói MỘT checkpoint cuối → HF
# Trạng thái → /workspace/p3_status.log. Dùng: (setsid nohup bash scripts/box_run_p3_loop.sh > /workspace/run_p3_loop.log 2>&1 &)
cd /workspace/nn-aivie
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PIP_CONSTRAINT
PY=/workspace/vlmenv/bin/python                                    # torch/mamba NGC + transformers (dựng ở P3.4)
VLLM=/workspace/vllm_env
TAG=${TAG:-r1}; NCONV=${NCONV:-1500}
X="$PY -W ignore -u scripts/expert_office_1p3b.py"
V="$PY -W ignore -u scripts/vlm_1p3b.py"
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
A4="--shuf 2 --no-thumb --max-tiles 8 --max-len 3072"
EX0="--code ckpt/expert_code/p3_code.pt --office ckpt/expert_code/p3_vp.pt --hoc ckpt/expert_code/p3_hoc.pt"
EX1="--code ckpt/expert_code/p3${TAG}_code.pt --office ckpt/expert_code/p3${TAG}_vp.pt --hoc ckpt/expert_code/p3${TAG}_hoc.pt"
D=/workspace/p3loop; mkdir -p $D
step() { [ -f $D/$1.done ]; }
done_() { touch $D/$1.done; log "P3.3' $1 xong · $2"; }
vllm_on() {
  curl -s localhost:8000/v1/models | grep -q gemma4-31b && return
  CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
    --max-model-len 8192 --gpu-memory-utilization ${1:-0.45} --max-num-seqs 64 --port 8000 > /workspace/p3loop_vllm.log 2>&1 < /dev/null &
  W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI P3.3': vLLM không lên"; exit 1; }; done
}
vllm_off() { pkill -f "vllm serve"; sleep 20; }

until grep -q "P3.4 XONG" $S; do
  pgrep -f box_run_p3_vlm.sh > /dev/null || grep -q "P3.4 XONG" $S || { log "LỖI P3.3': box_run_p3_vlm.sh dừng mà chưa XONG"; exit 1; }
  sleep 120
done
log "P3.3' bắt đầu: vòng thầy chung chữ + ảnh ($NCONV hội thoại, tag $TAG)"

# ── 1. thầy chat – chấm – sửa ──────────────────────────────────────────────────────────────────
if ! step loop; then
  vllm_on 0.55
  $PY -u scripts/vlm_loop.py run --base ckpt/p3_assistant.pt --vision ckpt/vlm/p3_s2.pt $T --n-conv $NCONV --workers 3 --bf16 \
     --tag $TAG --llm http://127.0.0.1:8000/v1 > /workspace/p3loop_run.log 2>&1 || { log "LỖI P3.3' vòng thầy (p3loop_run.log)"; exit 1; }
  $PY -u scripts/vlm_loop.py build --tags $TAG > /workspace/p3loop_build.log 2>&1 || { log "LỖI P3.3' tách dữ liệu"; exit 1; }
  hf upload donaldluong/nn-aivie-p3-data data/v6/$TAG.jsonl v6/$TAG.jsonl --repo-type dataset > /dev/null 2>&1   # quỹ đạo thầy (công khai)
  done_ loop "$(grep 'điểm TB' /workspace/p3loop_run.log | cut -c1-300) · $(grep '###' /workspace/p3loop_build.log | cut -c1-200)"
fi
vllm_off

# ── 2. expert chữ: SFT trên bản sửa (trộn ôn) → DPO ─────────────────────────────────────────────
if ! step text; then
  $X buildV5 --v5-tags ${TAG}_txt > /workspace/p3loop_buildv5.log 2>&1 || { log "LỖI P3.3' buildV5"; exit 1; }
  $X trainGEN --base ckpt/p3_sft.pt $EX0 --train code,vp,hoc --mix fix:0.5,mt:0.15,off:0.1,sgk:0.1,math:0.05,tb:0.05,calib:0.05 \
     --lr-old 2e-5 --steps 600 --tag p3${TAG}s > /workspace/p3loop_sft.log 2>&1 || { log "LỖI P3.3' SFT expert"; exit 1; }
  $X trainDPO --base ckpt/p3_sft.pt --code ckpt/expert_code/p3${TAG}s_code.pt --office ckpt/expert_code/p3${TAG}s_vp.pt \
     --hoc ckpt/expert_code/p3${TAG}s_hoc.pt --train code,vp,hoc --lr-old 5e-6 --beta 0.1 --steps 300 --bs 4 --tag p3$TAG \
     > /workspace/p3loop_dpo.log 2>&1 || { log "LỖI P3.3' DPO"; exit 1; }
  for n in code vp hoc; do hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/p3${TAG}_$n.pt expert_code/p3${TAG}_$n.pt > /dev/null 2>&1; done
  done_ text "$(cat /workspace/p3loop_buildv5.log | tail -1) · $(grep -E '^\[dpo\] bước' /workspace/p3loop_dpo.log | tail -1)"
fi

# ── 3. đọc ảnh học tiếp TRÊN expert mới ──────────────────────────────────────────────────────────
B1=ckpt/p3_assistant_$TAG.pt
if ! step vlm; then
  $V assemble --base ckpt/p3_sft.pt $EX1 --out $B1 > /workspace/p3loop_assemble.log 2>&1 || { log "LỖI P3.3' gộp checkpoint"; exit 1; }
  $PY - <<EOF > /workspace/p3loop_vmix.log 2>&1
import json, random
r = random.Random(33)
fix = [json.loads(l) for l in open("data/vlm/fix_$TAG.jsonl")]
old = [json.loads(l) for l in open("data/vlm/sft5.jsonl")]; r.shuffle(old)
mix = fix * 2 + old[:20000]                    # bản sửa ×2 + ôn 20K mẫu cũ (chống quên)
r.shuffle(mix)
with open("data/vlm/fixmix_$TAG.jsonl", "w") as f:
    for x in mix: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(len(fix), len(mix))
EOF
  N=$(awk '{print $2}' /workspace/p3loop_vmix.log); STEPS=$(( N / 32 )); [ $STEPS -gt 1500 ] && STEPS=1500; [ $STEPS -lt 200 ] && STEPS=200
  TB=$(cat /workspace/p3vlm/tok_budget 2>/dev/null || echo 4000)
  $V train --stage 2 --base $B1 $T $A4 --init ckpt/vlm/p3_s2.pt --data data/vlm/fixmix_$TAG.jsonl --steps $STEPS --bs 32 --lr 5e-5 \
     --lr-expert 1e-4 --tower-last 6 --lr-tower 1e-5 --lora 64 --lr-lora 5e-5 --warmup 50 --val-every 500 --save-every 500 \
     --tok-budget $TB --out ckpt/vlm/p3_s2$TAG.pt > /workspace/p3loop_vtrain.log 2>&1 || { log "LỖI P3.3' train đọc ảnh"; exit 1; }
  done_ vlm "bản sửa có ảnh $(awk '{print $1}' /workspace/p3loop_vmix.log) · $STEPS bước · $(grep 'val sau' /workspace/p3loop_vtrain.log)"
fi

# ── 4. chấm lại chữ + ảnh → gói checkpoint cuối ─────────────────────────────────────────────────
if ! step eval; then
  O=http://127.0.0.1:11434
  curl -s $O/api/tags > /dev/null || { OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_KEEP_ALIVE=24h setsid nohup ollama serve \
    >> /workspace/ollama_serve.log 2>&1 < /dev/null & sleep 10; }
  TV="--base ckpt/p3_sft.pt $EX1 --llm $O"
  $X evalv1 $TV --tag p3$TAG > /workspace/p3loop_evalv1.log 2>&1
  $X easy20 $TV --tag p3$TAG > /workspace/p3loop_easy20.log 2>&1
  $X evaloff $TV --ollama $O --tag office_p3$TAG > /workspace/p3loop_evaloff.log 2>&1
  $X evalsgk $TV --n 400 --tag p3$TAG > /workspace/p3loop_evalsgk.log 2>&1
  $X evalmath $TV --tag p3$TAG > /workspace/p3loop_evalmath.log 2>&1
  python -u scripts/eval_clean.py judge out/evalclean_p3$TAG.jsonl > /workspace/p3loop_judge.log 2>&1
  python -u scripts/eval_clean.py report p3 p3$TAG > /workspace/p3loop_report.txt 2>&1
  pkill -x ollama; sleep 5
  E="$V evalvlm --base $B1 $T --vision ckpt/vlm/p3_s2$TAG.pt --device cuda --tag p3$TAG --max-tiles 8"
  for t in screen nat chart ovqa news code; do [ -s data/vlm/test_$t.jsonl ] && $E --test data/vlm/test_$t.jsonl --n 200 > /workspace/p3loop_veval_$t.log 2>&1; done
  vllm_on 0.85
  $V judgevlm --files "ckpt/vlm/eval_p3${TAG}_*.json" > /workspace/p3loop_vjudge.log 2>&1
  vllm_off
  tar -czf /workspace/p3loop_results.tgz out/*p3$TAG* ckpt/vlm/eval_p3${TAG}_*.json /workspace/p3loop_*.log /workspace/p3loop_report.txt 2>/dev/null
  hf upload donaldluong/nn-aivie-2p6b-p3 /workspace/p3loop_results.tgz results/p3loop_${TAG}_results.tgz > /dev/null 2>&1
  done_ eval "chữ: $(tail -4 /workspace/p3loop_report.txt | tr '\n' ' ' | cut -c1-400) · ảnh: $(grep '###' /workspace/p3loop_vjudge.log | sed 's/### //' | tr '\n' '|' | cut -c1-500)"
fi
if ! step pack; then
  $PY -c "
import torch
sd = torch.load('ckpt/vlm/p3_s2$TAG.pt', map_location='cpu')
out = {'proj': {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd['proj'].items()}, 'hid': sd['hid'], 'cfg': sd['cfg'],
       'expert': {k: v.to(torch.bfloat16) for k, v in sd['expert'].items()}, 'lora': sd.get('lora', {})}
if 'tower' in sd: out['tower'] = sd['tower']
torch.save(out, 'ckpt/vlm/p3_vision_${TAG}_bf16.pt')"
  $V pack --base $B1 --vision ckpt/vlm/p3_vision_${TAG}_bf16.pt $T --max-tiles 8 --out ckpt/aivie_p3_$TAG.pt > /workspace/p3loop_pack.log 2>&1
  $PY - <<EOF >> /workspace/p3loop_pack.log 2>&1
import torch
o = torch.load("ckpt/aivie_p3_$TAG.pt", map_location="cpu", weights_only=False)
o["state"] = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in o["state"].items()}
o.pop("opt", None); torch.save(o, "ckpt/aivie_p3_${TAG}_bf16.pt")
EOF
  hf upload donaldluong/nn-aivie-2p6b-p3-snap ckpt/aivie_p3_${TAG}_bf16.pt aivie_2p6b_p3_${TAG}_bf16.pt > /dev/null 2>&1 \
    && log "P3.3' đã đẩy checkpoint CUỐI (bf16) lên HF nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_${TAG}_bf16.pt"
  done_ pack "$(du -h ckpt/aivie_p3_${TAG}_bf16.pt | cut -f1)"
fi
log "P3.3' XONG — aivie 2,5B: SFT + 3 expert + đọc ảnh + vòng thầy chung · P3 hoàn tất"
