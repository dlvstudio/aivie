#!/bin/bash
# TK_108 C7 trên box 2×A100 (user đồng ý 22/10): ĐỀ THỰC TẾ tiếng Việt, thầy Gemma-4-31B FP8 (2 vLLM, mỗi GPU một bản).
#  1) thử nhỏ 40 đề (tỉ lệ giữ + tốc độ) → 2) bộ ĐO 300 đề (data/c7/eval.jsonl, KHÔNG train) → đẩy HF
#  3) bộ TRAIN 3 000 đề (user hạ từ 10 000 → 5 000 → 3 000 ngày 22/10 để kịp đo trong ngày) (data/c7/train.jsonl; đẩy HF TỪNG PHẦN mỗi 30 phút) → tắt vLLM
#  4) SFT cx3c = cx3b + C7 (700 bước; GPU 1 đo mốc cx3b trên bộ C7 cùng lúc, lr 5e-5, C7 35% + ôn tool-use / code / văn phòng) → đẩy HF → gói → snap
#  5) đo: tool-use v3 · HumanEval · MBPP · Excel/easy20 cho cx3c + bộ đo C7 cho CẢ cx3b (mốc) và cx3c → HF
# Dùng: (setsid nohup bash scripts/box_run_c7.sh > /workspace/c7_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
P3=donaldluong/nn-aivie-2p6b-p3
EV=/workspace/evalenv/bin/python
up() { hf upload $P3 "$1" "$2" > /dev/null 2>&1; }
# ── thầy: 2 vLLM độc lập, mỗi GPU một bản (TP 2 lỗi custom_all_reduce 'invalid argument' trong container, 22/10) ──
# FP8 trên A100 dùng kernel JIT (humming/NVRTC) cần libnvrtc-builtins.so.13.0 trong venv → thêm vào LD_LIBRARY_PATH
LLM=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1
for g in 0 1; do
  curl -s localhost:800$g/v1/models | grep -q gemma4-31b && continue
  LD_LIBRARY_PATH=/workspace/vllm_env/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH VLLM_USE_DEEP_GEMM=0 \
  CUDA_VISIBLE_DEVICES=$g setsid nohup /workspace/vllm_env/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 128 --port 800$g > /workspace/c7_vllm_$g.log 2>&1 < /dev/null &
  sleep 60                                                     # lệch giờ khởi động: 2 bản cùng biên dịch JIT / nạp đĩa
done
for g in 0 1; do
  W=0; until curl -s localhost:800$g/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI C7: vLLM GPU $g không lên (c7_vllm_$g.log)"; exit 1; }; done
done
log "C7 bắt đầu: thầy Gemma-4-31B FP8 (2 vLLM × 1 GPU) · thử nhỏ 40 → đo 300 → train 3 000"
# ── 1) thử nhỏ ──
mkdir -p /workspace/c7_smoke
if [ ! -s /workspace/c7_smoke/eval.jsonl ]; then
$EV -u scripts/c7_real.py gen --split eval --keep 30 --workers 40 --llm $LLM --out /workspace/c7_smoke > /workspace/c7_smoke.log 2>&1
log "C7 thử nhỏ · $(grep -E '^\[c7\] eval thử|###' /workspace/c7_smoke.log | tail -n 2 | tr '\n' ' ' | cut -c1-250)"
fi
# ── 2) bộ đo ──
$EV -u scripts/c7_real.py gen --split eval --keep 300 --workers 200 --llm $LLM > /workspace/c7_eval_gen.log 2>&1 || { log "LỖI C7: sinh bộ đo"; exit 1; }
log "C7 bộ đo xong · $(grep '###' /workspace/c7_eval_gen.log | cut -c1-250)"
up data/c7/eval.jsonl data/c7/eval.jsonl && log "C7 đã đẩy HF data/c7/eval.jsonl"
# ── 3) bộ train (đẩy HF từng phần) ──
( while sleep 1800; do [ -s data/c7/train.jsonl ] && up data/c7/train.jsonl data/c7/train.jsonl; [ -f /workspace/c7_train_done ] && break; done ) &
SYNC=$!
$EV -u scripts/c7_real.py gen --split train --keep 3000 --workers 200 --llm $LLM > /workspace/c7_train_gen.log 2>&1 || { log "LỖI C7: sinh bộ train"; exit 1; }
touch /workspace/c7_train_done
up data/c7/train.jsonl data/c7/train.jsonl
tar czf /workspace/c7_teacher_log.tgz data/c7/teacher_log && up /workspace/c7_teacher_log.tgz data/c7/teacher_log.tgz
log "C7 bộ train xong · $(grep '###' /workspace/c7_train_gen.log | cut -c1-250) · đã đẩy HF"
pkill -f "vllm serve"; sleep 20
# ── 4) SFT cx3c ──
MIX="c7:0.35,c3r:0.12,c3r2:0.08,c3:0.10,c2bV:0.10,c2bF:0.05,c2bA:0.04,c2oci:0.03,mt:0.06,offfix:0.04,fixfix:0.03"
log "cx3c bắt đầu: SFT từ cx3b + C7 · 700 bước · lr 5e-5 · $MIX (GPU 1: đo mốc cx3b trên bộ C7)"
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
c7e() { $EV -u scripts/c7_real.py eval --url http://localhost:$1 --tag $2 --shard $3 --workers 1 > /workspace/c7_eval_$2_${3/\//of}.log 2>&1 \
        || log "LỖI C7 đo $2 phần $3"; }
c7r() { log "C7 đo $1 · $($EV scripts/c7_real.py report --tag $1 | cut -c1-260)"; }
BP=""
for p in 8885 8886; do
  CUDA_VISIBLE_DEVICES=1 python -W ignore -u scripts/serve_hybrid_ui.py --ckpt ckpt/aivie_p3_cx3b_bf16.pt --port $p --no-rag --max-ctx 4096 > /workspace/c7_ui_$p.log 2>&1 &
  BP="$BP $!"
done
( for p in 8885 8886; do W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && exit 1; done; done
  c7e 8885 cx3b 1/2 & J1=$!; c7e 8886 cx3b 2/2 & J2=$!; wait $J1 $J2; c7r cx3b; kill $BP ) &
BASE=$!
CUDA_VISIBLE_DEVICES=0 python -W ignore -u scripts/expert_office_1p3b.py trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/cx3b_code.pt \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code --mix "$MIX" \
   --lr-old 5e-5 --steps 700 --bs 1 --accum 16 --tag cx3c > /workspace/cx3c_sft.log 2>&1 \
   || { log "LỖI cx3c: SFT (cx3c_sft.log)"; exit 1; }
log "cx3c SFT xong · $(grep '\[gen\] sau' /workspace/cx3c_sft.log | cut -c1-250)"
up ckpt/expert_code/cx3c_code.pt expert_code/cx3c_code.pt && log "cx3c đã đẩy HF expert_code/cx3c_code.pt"
python - <<'PY' > /workspace/cx3c_pack.log 2>&1 || { log "LỖI cx3c: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/cx3c_code.pt", map_location="cpu")
pos = sorted({int(k.split(".")[0]) for k in sd})
o["experts"]["code"] = {"state": {k: v.to(torch.bfloat16) for k, v in sd.items()}, "gated": True,
                        "hid": sd[f"{pos[0]}.w3.weight"].shape[1], "pos": pos}
o.setdefault("merged_from", []).append({"code": "cx3c_code (TK_108 C7: cx3b + SFT đề thực tế tiếng Việt 700 bước)"})
torch.save(o, "ckpt/aivie_p3_cx3c_bf16.pt"); print("ok", len(pos))
PY
hf upload donaldluong/nn-aivie-2p6b-p3-snap ckpt/aivie_p3_cx3c_bf16.pt aivie_2p6b_p3_cx3c_bf16.pt > /dev/null 2>&1 \
  && log "cx3c đã đẩy HF snap aivie_2p6b_p3_cx3c_bf16.pt"
# ── 5) đo: bộ C7 trước (2 máy chủ chia đôi) — mục tiêu chính; tool/HE/MBPP/văn phòng chạy song song ──
wait $BASE                                                     # mốc cx3b trên bộ C7 phải xong (thường xong trước SFT)
CE=scripts/code_eval.py; PIDS=""
for pg in 8881:0 8882:0 8883:1 8884:1; do
  IFS=: read p g <<< "$pg"
  CUDA_VISIBLE_DEVICES=$g python -W ignore -u scripts/serve_hybrid_ui.py --ckpt ckpt/aivie_p3_cx3c_bf16.pt --port $p --no-rag --max-ctx 4096 > /workspace/c7_ui_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in 8881 8882 8883 8884; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI cx3c: UI $p"; kill $PIDS; exit 1; }; done
done
JOBS=""
( c7e 8881 cx3c 1/2 ) &
C1=$!
( c7e 8882 cx3c 2/2
  $EV -u $CE run --suite humaneval --url http://localhost:8882 --tag cx3c > /workspace/cx3c_humaneval.log 2>&1 \
    && log "cx3c humaneval (cx3b 13,4) · $(grep '###' /workspace/cx3c_humaneval.log | tail -1 | cut -c1-200)" || log "LỖI cx3c humaneval" ) &
C2=$!
JOBS="$C1 $C2"
( $EV -u $CE run --suite agent --url http://localhost:8883 --tag cx3c > /workspace/cx3c_agent.log 2>&1 \
    && log "cx3c agent (cx3b 55) · $(grep '###' /workspace/cx3c_agent.log | tail -1 | cut -c1-200)" || log "LỖI cx3c agent"
  $EV -u scripts/office_probe_light.py http://localhost:8883 cx3c > /workspace/cx3c_office.log 2>&1 \
    && log "cx3c văn phòng (cx3b Excel 15 · easy20 16) · $(grep '###' /workspace/cx3c_office.log)" || log "LỖI cx3c văn phòng" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite mbpp --url http://localhost:8884 --tag cx3c > /workspace/cx3c_mbpp.log 2>&1 \
    && log "cx3c mbpp (cx3b 23,3) · $(grep '###' /workspace/cx3c_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI cx3c mbpp" ) &
JOBS="$JOBS $!"
( until grep -q "^### c7 cx3c_s1" /workspace/c7_eval_cx3c_1of2.log 2>/dev/null && grep -q "^### c7 cx3c_s2" /workspace/c7_eval_cx3c_2of2.log 2>/dev/null; do
    sleep 30; kill -0 $C1 2>/dev/null || kill -0 $C2 2>/dev/null || grep -q "^### c7 cx3c_s2" /workspace/c7_eval_cx3c_2of2.log 2>/dev/null || break; done
  c7r cx3c; log "C7 KẾT QUẢ CHÍNH có rồi — so cx3b ↔ cx3c ở 2 dòng 'C7 đo' trên" ) &
JOBS="$JOBS $!"
wait $JOBS                                                     # KHÔNG 'wait' trơn — sẽ chờ cả máy chủ UI
log "cx3c lỗi máy chủ: $(cat /workspace/c7_ui_*.log | grep -c '\[lỗi /api/chat\]') lần"
kill $PIDS
tar czf /workspace/cx3c_ketqua.tgz out/code_eval/*cx3c* out/code_eval/c7_* out/office_probe_cx3c.jsonl ckpt/expert_code/*cx3c*.json \
    /workspace/cx3c_*.log /workspace/c7_*.log 2>/dev/null
up /workspace/cx3c_ketqua.tgz backup/cx3/cx3c_ketqua.tgz && log "cx3c đã đẩy HF kết quả đo + log"
log "C7 XONG — cx3c so với cx3b (tool 55 · HE 13,4 · MBPP 23,3 · Excel 15 · easy20 16) + bộ đo C7 cả hai bản (dòng 'C7 đo')"
