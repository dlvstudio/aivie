#!/bin/bash
# v5 CHẠY TIẾP từ DPO vòng r1 (lần trước hết bộ nhớ: vLLM 40GB + DPO ~39GB). Tắt vLLM khi train DPO, bật lại cho vòng sau.
cd /workspace/nn-aivie
S=/workspace/code_status.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/zenv/bin/python
P="env PYTHONPATH=scripts:src $PY -W ignore -u scripts/expert_office_1p3b.py"
B="--base ckpt/hybrid_1p3b_chat_v2.pt"
ex() { echo "--code ckpt/expert_code/$1_code.pt --office ckpt/expert_code/$1_vp.pt --hoc ckpt/expert_code/$1_hoc.pt"; }
vllm_on() {
  curl -s localhost:8000/v1/models | grep -q gemma4-31b && return
  setsid nohup /workspace/vllm_env/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
    --max-model-len 6144 --gpu-memory-utilization 0.50 --max-num-seqs 64 --port 8000 > /workspace/vllm_serve_v5.log 2>&1 < /dev/null &
  until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 10; done
}
vllm_off() { pkill -f "vllm serve"; sleep 25; }
bf16() { $PY -c "import torch
for n in ('$1_code','$1_vp','$1_hoc'):
    sd=torch.load(f'ckpt/expert_code/{n}.pt',map_location='cpu'); torch.save({k:v.to(torch.bfloat16) for k,v in sd.items()}, f'ckpt/expert_code/{n}_bf16.pt')"; }
dpo() {   # $1 = vòng
  vllm_off
  $P trainDPO $B $(ex v5$1s) --train code,vp,hoc --lr-old 5e-6 --beta 0.1 --steps 300 --bs 4 --tag v5$1 \
     > /workspace/v5_dpo_$1.log 2>&1 || { echo "$(date) LỖI V5 DPO $1 (bản tiếp)" >> $S; exit 1; }
  rm -f ckpt/expert_code/v5$1s_*.pt; bf16 v5$1
  echo "$(date) V5 vòng $1: SFT+DPO xong · $(grep -E '^\[dpo\] bước' /workspace/v5_dpo_$1.log | tail -1)" >> $S
}
echo "$(date) V5: chạy tiếp từ DPO r1 (tắt vLLM khi train DPO)" >> $S
dpo r1
# ── vòng r2 ──
vllm_on
echo "$(date) V5 vòng r2: thầy chat 1000 hội thoại với v5r1" >> $S
$PY -u scripts/v5_loop.py --n-conv 1000 --workers 3 --tag r2 --seed 2 $(ex v5r1) > /workspace/v5_loop_r2.log 2>&1
echo "$(date) V5 vòng r2 (GPU): $(grep 'điểm TB' /workspace/v5_loop_r2.log)" >> $S
$P buildV5 --v5-tags r2 > /workspace/v5_build_r2.log 2>&1
echo "$(date) V5 vòng r2: $(cat /workspace/v5_build_r2.log)" >> $S
vllm_off
$P trainGEN $B $(ex v5r1) --train code,vp,hoc --mix fix:0.5,mt:0.15,off:0.1,sgk:0.1,math:0.05,tb:0.05,calib:0.05 \
   --lr-old 2e-5 --steps 600 --tag v5r2s > /workspace/v5_sft_r2.log 2>&1 || { echo "$(date) LỖI V5 SFT r2" >> $S; exit 1; }
dpo r2
# ── vòng đo r3 (không train) ──
vllm_on
$PY -u scripts/v5_loop.py --n-conv 300 --workers 3 --tag r3 --seed 3 $(ex v5r2) > /workspace/v5_loop_r3.log 2>&1
echo "$(date) V5 đo r3: $(grep 'điểm TB' /workspace/v5_loop_r3.log)" >> $S
vllm_off
# ── bộ chấm chuẩn (giám khảo Gemma-26B) — chỉ chờ đúng tiến trình chấm ──
curl -s localhost:11434/api/tags > /dev/null || { OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_KEEP_ALIVE=24h setsid nohup ollama serve >> /workspace/ollama_serve.log 2>&1 < /dev/null & sleep 10; }
O=http://127.0.0.1:11434
V5="$B $(ex v5r2) --llm $O"
$P mbpp $V5 --n 500 --tag v5_mbpp500 > /workspace/v5_mbpp.log 2>&1 & J1=$!
sleep 45; $P humaneval $V5 --tag he_v5 > /workspace/v5_he.log 2>&1 & J2=$!
sleep 45; ( $P evaloff $V5 --ollama $O --tag office_v5 > /workspace/v5_evaloff.log 2>&1; $P easy20 $V5 --tag v5 > /workspace/v5_easy20.log 2>&1 ) & J3=$!
sleep 45; ( $P evalsgk $V5 --n 400 --tag v5 > /workspace/v5_evalsgk.log 2>&1; $P evalmath $V5 --tag v5 > /workspace/v5_evalmath.log 2>&1 ) & J4=$!
sleep 45; ( $P evalmt $V5 --tag v5 > /workspace/v5_evalmt.log 2>&1; $P evalv1 $V5 --tag v5 > /workspace/v5_evalv1.log 2>&1 ) & J5=$!
wait $J1 $J2 $J3 $J4 $J5
env PYTHONPATH=scripts:src python -u scripts/eval_clean.py judge out/evalclean_v5.jsonl > /workspace/v5_judge.log 2>&1
env PYTHONPATH=scripts:src python -u scripts/eval_clean.py report sftv2 v3 v4 v5 > /workspace/report_v5.txt 2>&1
echo "$(date) V5 XONG" >> $S
