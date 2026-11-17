#!/bin/bash
# TK_107 P3.4 · ĐỌC ẢNH trên thân 2,5B (SFT + 3 expert) — box 1×H100 P3, chạy SAU box_run_p3.sh (chờ dấu ckpt/p3_eval.done).
#   0  môi trường: venv thừa hưởng torch + mamba của NGC (--system-site-packages) + transformers (mắt SigLIP2);
#      venv vLLM riêng (giám khảo Gemma nhìn ảnh) · gộp thân + 3 expert → ckpt/p3_assistant.pt
#   1  dữ liệu đọc ảnh từ HF `nn-aivie-vlm-data` (~30GB: V1 + V4 ảnh chữ + V5 báo/ảnh chữ nhiều số/ảnh màn hình đọc-trước
#      + 20K ảnh code) · mắt SigLIP2 (HTTP range, bf16)
#   2  thử nhỏ 2 giai đoạn (bắt lỗi / hết bộ nhớ, chọn ngân sách token)
#   3  giai đoạn 1: bộ chiếu học căn chỉnh (align) · giai đoạn 2: bộ chiếu + expert thị giác + 6 tầng cuối mắt + LoRA r64 trên
#      12 khối attention (công thức V4) trên sft5 (V4 + V5 + code, ~240K mẫu)
#   4  đo: sinh trên các bộ test giữ riêng (ảnh màn hình, ảnh tự nhiên, biểu đồ, OpenViVQA, báo, CODE) → giám khảo chấm
#   5  gói MỘT checkpoint (thân + 3 expert + mắt + bộ chiếu + expert thị giác + LoRA giữ riêng) → HF
# Trạng thái → /workspace/p3_status.log (chung với P3). Dùng: (setsid nohup bash scripts/box_run_p3_vlm.sh > /workspace/run_p3_vlm.log 2>&1 &)
cd /workspace/nn-aivie
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PIP_CONSTRAINT
VENV=/workspace/vlmenv; PY=$VENV/bin/python                         # torch/mamba của NGC + transformers
VLLM=/workspace/vllm_env                                           # vLLM riêng (có torch riêng)
HF_P3=${HF_P3:-donaldluong/nn-aivie-2p6b-p3}
HF_VLM=${HF_VLM:-donaldluong/nn-aivie-vlm-data}
B=ckpt/p3_assistant.pt
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
A4="--shuf 2 --no-thumb --max-tiles 8 --max-len 3072"
V="$PY -W ignore -u scripts/vlm_1p3b.py"
D=/workspace/p3vlm; mkdir -p $D ckpt/vlm ckpt/vision
step() { [ -f $D/$1.done ]; }
done_() { touch $D/$1.done; log "P3.4 $1 xong · $2 · ổ còn $(df -h / | tail -1 | awk '{print $4}')"; }

until [ -f ckpt/p3_eval.done ]; do                                    # chờ P3 (SFT → expert → chấm) xong, GPU rảnh
  pgrep -f box_run_p3.sh > /dev/null || { [ -f ckpt/p3_eval.done ] || { log "LỖI P3.4: box_run_p3.sh dừng mà chưa có p3_eval.done"; exit 1; }; }
  sleep 120
done
log "P3.4 bắt đầu (đọc ảnh trên thân 2,5B)"

# ── 0. môi trường + gộp checkpoint ─────────────────────────────────────────────────────────────
if ! step env; then
  [ -x $PY ] || python -m venv --system-site-packages $VENV
  $PY -m pip install -q "transformers>=4.49" pillow > /workspace/p3vlm_pip.log 2>&1 || { log "LỖI P3.4: cài transformers"; exit 1; }
  $PY -c "import torch, mamba_ssm, transformers; from transformers import SiglipVisionModel; print('torch', torch.__version__, 'tf', transformers.__version__)" \
    >> /workspace/p3vlm_pip.log 2>&1 || { log "LỖI P3.4: venv không nạp được torch/mamba/transformers"; exit 1; }
  if [ ! -x $VLLM/bin/vllm ]; then
    python -m venv $VLLM && $VLLM/bin/python -m pip install -q vllm >> /workspace/p3vlm_pip.log 2>&1 || { log "LỖI P3.4: cài vLLM"; exit 1; }
  fi
  done_ env "$(tail -1 /workspace/p3vlm_pip.log)"
fi
if ! step assemble; then
  $V assemble --base ckpt/p3_sft.pt --code ckpt/expert_code/p3_code.pt --office ckpt/expert_code/p3_vp.pt \
     --hoc ckpt/expert_code/p3_hoc.pt --out $B > /workspace/p3vlm_assemble.log 2>&1 || { log "LỖI P3.4: gộp checkpoint"; exit 1; }
  done_ assemble "$(tail -1 /workspace/p3vlm_assemble.log)"
fi

# ── 1. dữ liệu + mắt ───────────────────────────────────────────────────────────────────────────
if ! step data; then
  mkdir -p /workspace/dl_vlm
  hf download $HF_VLM --repo-type dataset --local-dir /workspace/dl_vlm > /workspace/p3vlm_dl.log 2>&1 || { log "LỖI P3.4: tải dữ liệu ảnh"; exit 1; }
  for f in /workspace/dl_vlm/*.tar /workspace/dl_vlm/*.tar.zst; do
    case $f in *.zst) tar -I zstd -xf $f -C . ;; *) tar -xf $f -C . ;; esac && rm -f $f    # giải nén xong xoá ngay (ổ)
  done
  [ -s data/vlm/sft5.jsonl ] && [ -s data/vlm/align.jsonl ] || { log "LỖI P3.4: thiếu sft5/align sau giải nén"; exit 1; }
  done_ data "align $(wc -l < data/vlm/align.jsonl) · sft5 $(wc -l < data/vlm/sft5.jsonl) · $(du -sh data/vlm | cut -f1)"
fi
[ -s ckpt/vision/siglip2_so400m_384_bf16.safetensors ] || $PY scripts/fetch_vision_tower.py \
  --out ckpt/vision/siglip2_so400m_384_bf16.safetensors > /workspace/p3vlm_tower.log 2>&1 || { log "LỖI P3.4: tải mắt SigLIP2"; exit 1; }

# ── 2. thử nhỏ (bắt lỗi trước khi đốt giờ GPU) ────────────────────────────────────────────────
if ! step smoke; then
  $V train --stage 1 --base $B $T $A4 --data data/vlm/align.jsonl --steps 4 --bs 64 --n-val 32 --val-every 100000 \
     --out ckpt/vlm/p3_smoke1.pt > /workspace/p3vlm_smoke1.log 2>&1 || { log "LỖI P3.4 thử giai đoạn 1 (p3vlm_smoke1.log)"; exit 1; }
  TB=0
  for tb in 8000 6000 4000; do
    if $V train --stage 2 --base $B $T $A4 --init ckpt/vlm/p3_smoke1.pt --data data/vlm/sft5.jsonl --steps 3 --bs 32 --n-val 32 \
         --val-every 100000 --tower-last 6 --lora 64 --tok-budget $tb --out ckpt/vlm/p3_smoke2.pt > /workspace/p3vlm_smoke2.log 2>&1; then
      TB=$tb; break
    fi
    log "P3.4 thử giai đoạn 2 lỗi ở ngân sách $tb: $(grep -m1 -E 'Error' /workspace/p3vlm_smoke2.log | cut -c1-150)"
  done
  rm -f ckpt/vlm/p3_smoke1.pt ckpt/vlm/p3_smoke2.pt
  [ "$TB" -gt 0 ] || { log "LỖI P3.4 thử giai đoạn 2 (p3vlm_smoke2.log)"; exit 1; }
  echo $TB > $D/tok_budget
  done_ smoke "ngân sách $TB · $(grep -E 'bước 2/' /workspace/p3vlm_smoke2.log | grep -o 'loss.*')"
fi
TB=$(cat $D/tok_budget)

# ── 3. train ──────────────────────────────────────────────────────────────────────────────────
if ! step s1; then
  $V train --stage 1 --base $B $T $A4 --data data/vlm/align.jsonl --steps 1100 --bs 64 --lr 1e-3 --warmup 100 \
     --val-every 500 --out ckpt/vlm/p3_s1.pt > /workspace/p3vlm_s1.log 2>&1 || { log "LỖI P3.4 giai đoạn 1 (p3vlm_s1.log)"; exit 1; }
  done_ s1 "$(grep 'val trước' /workspace/p3vlm_s1.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/p3vlm_s1.log)"
fi
if ! step s2; then
  $V train --stage 2 --base $B $T $A4 --init ckpt/vlm/p3_s1.pt --data data/vlm/sft5.jsonl --steps 3500 --bs 32 --lr 1e-4 \
     --lr-expert 2e-4 --tower-last 6 --lr-tower 2e-5 --lora 64 --lr-lora 1e-4 --warmup 200 --val-every 1000 --save-every 500 \
     --tok-budget $TB --out ckpt/vlm/p3_s2.pt > /workspace/p3vlm_s2.log 2>&1 || { log "LỖI P3.4 giai đoạn 2 (p3vlm_s2.log)"; exit 1; }
  $PY -c "
import torch
sd = torch.load('ckpt/vlm/p3_s2.pt', map_location='cpu')
out = {'proj': {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd['proj'].items()}, 'hid': sd['hid'], 'cfg': sd['cfg'],
       'expert': {k: v.to(torch.bfloat16) for k, v in sd['expert'].items()}, 'lora': sd.get('lora', {})}
if 'tower' in sd: out['tower'] = sd['tower']
torch.save(out, 'ckpt/vlm/p3_vision_bf16.pt')"
  hf upload $HF_P3 ckpt/vlm/p3_vision_bf16.pt vlm/p3_vision_bf16.pt > /dev/null 2>&1 && log "P3.4 sao lưu phần đọc ảnh lên HF"
  done_ s2 "$(grep 'val trước' /workspace/p3vlm_s2.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/p3vlm_s2.log)"
fi

# ── 4. đo: sinh (GPU) → giám khảo Gemma nhìn chữ đáp án chuẩn chấm ───────────────────────────
if ! step eval; then
  E="$V evalvlm --base $B $T --vision ckpt/vlm/p3_s2.pt --device cuda --tag p3 --max-tiles 8"
  for t in screen nat chart ovqa news code; do
    [ -s data/vlm/test_$t.jsonl ] && $E --test data/vlm/test_$t.jsonl --n 200 > /workspace/p3vlm_eval_$t.log 2>&1
  done
  [ -s data/vlm/test_sgk_text.jsonl ] && $E --test data/vlm/test_sgk_text.jsonl --n 150 > /workspace/p3vlm_eval_sgk.log 2>&1
  CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
     --max-model-len 6144 --gpu-memory-utilization 0.85 --max-num-seqs 64 --port 8000 > /workspace/p3vlm_vllm.log 2>&1 < /dev/null &
  W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI P3.4: vLLM không lên"; break; }; done
  $V judgevlm --files "ckpt/vlm/eval_p3_*.json" > /workspace/p3vlm_judge.log 2>&1
  pkill -f "vllm serve"; sleep 10
  tar -czf /workspace/p3vlm_results.tgz ckpt/vlm/eval_p3_*.json /workspace/p3vlm_*.log 2>/dev/null
  hf upload $HF_P3 /workspace/p3vlm_results.tgz results/p3vlm_results.tgz > /dev/null 2>&1
  done_ eval "$(grep '###' /workspace/p3vlm_judge.log | sed 's/### //' | tr '\n' '|' | cut -c1-600)"
fi

# ── 5. MỘT checkpoint: thân + 3 expert + đọc ảnh (LoRA giữ riêng → chat chữ không đổi khi không có ảnh) ─────
if ! step pack; then
  $V pack --base $B --vision ckpt/vlm/p3_vision_bf16.pt $T --max-tiles 8 --out ckpt/p3_vlm.pt > /workspace/p3vlm_pack.log 2>&1 \
    || { log "LỖI P3.4: gói checkpoint"; exit 1; }
  $PY - <<'EOF' >> /workspace/p3vlm_pack.log 2>&1
import torch                                                     # bản gọn để chạy trên Mac: thân bf16, bỏ optimizer
o = torch.load("ckpt/p3_vlm.pt", map_location="cpu", weights_only=False)
o["state"] = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in o["state"].items()}
o.pop("opt", None)
torch.save(o, "ckpt/p3_vlm_bf16.pt")
EOF
  hf upload donaldluong/nn-aivie-2p6b-p3-snap ckpt/p3_vlm_bf16.pt aivie_2p6b_p3_vlm_bf16.pt > /dev/null 2>&1 \
    && log "P3.4 đã đẩy MỘT checkpoint (bf16) lên HF nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_vlm_bf16.pt"
  done_ pack "$(tail -1 /workspace/p3vlm_pack.log) · bf16 $(du -h ckpt/p3_vlm_bf16.pt | cut -f1)"
fi
log "P3.4 XONG — đọc ảnh trên thân 2,5B · tiếp: P3.3' vòng thầy chung chữ + ảnh (vlm_loop.py)"
