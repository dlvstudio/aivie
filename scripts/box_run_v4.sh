#!/bin/bash
# 1.3B v4: dữ liệu (vLLM Gemma-4-31B) → code v3 → expert học thuật (SGK+toán) → bước cuối chat dài (3 expert) → chấm.
cd /workspace/nn-aivie
S=/workspace/code_status.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/zenv/bin/python
V=http://127.0.0.1:8000/v1
G="${GEN_HOURS:-5}"                          # giờ tối đa cho pha sinh dữ liệu
# ── A. sinh dữ liệu bằng vLLM ──
pkill -x ollama; sleep 3
curl -s localhost:8000/v1/models | grep -q gemma4-31b || setsid nohup /workspace/vllm_env/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
  --max-model-len 6144 --gpu-memory-utilization 0.90 --max-num-seqs 192 --port 8000 > /workspace/vllm_serve.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 10; done
echo "$(date) V4: vLLM Gemma-4-31B sẵn sàng — sinh dữ liệu (tối đa ${G} giờ)" >> $S
$PY -u scripts/gen_sgk.py topics --ollama $V > /workspace/v4_topics.log 2>&1
( $PY -u scripts/gen_sgk.py gen --workers 64 --ollama $V >> /workspace/v4_sgk.log 2>&1 ) &
( $PY -u scripts/gen_math.py gen --n 12000 --workers 48 --ollama $V >> /workspace/v4_math.log 2>&1 ) &
( $PY -u scripts/gen_multiturn.py gen --n 9000 --workers 48 --ollama $V >> /workspace/v4_mt.log 2>&1 ) &
( $PY -u scripts/gen_code_textbook.py gen --n 30000 --workers 32 --ollama $V --out data/code_tb/raw.jsonl >> /workspace/v4_tb.log 2>&1 ) &
END=$(( $(date +%s) + G * 3600 ))
while [ $(date +%s) -lt $END ] && pgrep -f "[g]en_(sgk|math|multiturn).py gen" > /dev/null; do sleep 60; done
pkill -f "[g]en_(sgk|math|multiturn|code_textbook).py gen"; sleep 5
pkill -f "vllm serve"; sleep 20
echo "$(date) V4: dừng sinh dữ liệu" >> $S
$PY scripts/gen_sgk.py build > /workspace/v4_build_sgk.log 2>&1
$PY scripts/gen_math.py build > /workspace/v4_build_math.log 2>&1
$PY scripts/gen_multiturn.py build > /workspace/v4_build_mt.log 2>&1
$PY scripts/gen_code_textbook.py build > /workspace/v4_build_tb.log 2>&1
echo "$(date) V4 dữ liệu: $(cat /workspace/v4_build_sgk.log | tail -1) | $(tail -1 /workspace/v4_build_math.log) | $(tail -1 /workspace/v4_build_mt.log) | $(head -1 /workspace/v4_build_tb.log)" >> $S
# ── B. train ──
P="env PYTHONPATH=scripts:src $PY -W ignore -u scripts/expert_office_1p3b.py"
B="--base ckpt/hybrid_1p3b_chat_v2.pt"
$P trainCODE2 $B --code ckpt/expert_code/code_v2.pt --lr 1e-4 --steps 1500 --tag code_v3 > /workspace/v4_code3.log 2>&1 || { echo "$(date) LỖI V4 code v3" >> $S; exit 1; }
echo "$(date) V4: train expert học thuật" >> $S
$P trainGEN $B --code ckpt/expert_code/code_v3.pt --office ckpt/expert_code/office_v2.pt --train hoc \
   --mix sgk:0.4,math:0.3,mt:0.1,chat:0.1,off:0.05,vi_code:0.05 --steps 3000 --tag v4a > /workspace/v4_hoc.log 2>&1 || { echo "$(date) LỖI V4 học thuật" >> $S; exit 1; }
echo "$(date) V4: bước cuối chat dài (3 expert)" >> $S
$P trainGEN $B --code ckpt/expert_code/code_v3.pt --office ckpt/expert_code/office_v2.pt --hoc ckpt/expert_code/v4a_hoc.pt \
   --train code,vp,hoc --mix mt:0.35,off:0.15,sgk:0.1,math:0.1,tb:0.1,chat:0.1,calib:0.1 --lr-old 2e-5 --steps 1000 --tag v4c \
   > /workspace/v4_chat.log 2>&1 || { echo "$(date) LỖI V4 chat dài" >> $S; exit 1; }
# ── C. chấm (giám khảo Gemma-26B qua Ollama — cùng giám khảo với các bản trước) ──
OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_KEEP_ALIVE=24h setsid nohup ollama serve >> /workspace/ollama_serve.log 2>&1 < /dev/null &
sleep 10
echo "$(date) V4: chấm" >> $S
O=http://127.0.0.1:11434
V4="$B --code ckpt/expert_code/v4c_code.pt --office ckpt/expert_code/v4c_vp.pt --hoc ckpt/expert_code/v4c_hoc.pt --llm $O"
V3="$B --code ckpt/expert_code/code_v2.pt --office ckpt/expert_code/office_v2.pt --llm $O"
$P mbpp $V4 --n 500 --tag v4_mbpp500 > /workspace/v4_mbpp.log 2>&1 &
sleep 45; $P humaneval $V4 --tag he_v4 > /workspace/v4_he.log 2>&1 &
sleep 45; ( $P evaloff $V4 --ollama $O --tag office_v4 > /workspace/v4_evaloff.log 2>&1; $P easy20 $V4 --tag v4 > /workspace/v4_easy20.log 2>&1 ) &
sleep 45; ( $P evalsgk $V4 --n 400 --tag v4 > /workspace/v4_evalsgk.log 2>&1; $P evalsgk $V3 --n 400 --tag v3 > /workspace/v3_evalsgk.log 2>&1 ) &
sleep 45; ( $P evalmath $V4 --tag v4 > /workspace/v4_evalmath.log 2>&1; $P evalmath $V3 --tag v3 > /workspace/v3_evalmath.log 2>&1 ) &
wait
( $P evalmt $V4 --tag v4 > /workspace/v4_evalmt.log 2>&1 ) &
( $P evalmt $V3 --tag v3 > /workspace/v3_evalmt.log 2>&1 ) &
$P evalv1 $V4 --tag v4 > /workspace/v4_evalv1.log 2>&1
wait
env PYTHONPATH=scripts:src python -u scripts/eval_clean.py judge out/evalclean_v4.jsonl > /workspace/v4_judge.log 2>&1
env PYTHONPATH=scripts:src python -u scripts/eval_clean.py report sftv2 codeVI office_v1 v3 v4 > /workspace/report_v4.txt 2>&1
$PY -c "import torch
for n in ('v4c_code','v4c_vp','v4c_hoc'):
    sd=torch.load(f'ckpt/expert_code/{n}.pt',map_location='cpu'); torch.save({k:v.to(torch.bfloat16) for k,v in sd.items()}, f'ckpt/expert_code/{n}_bf16.pt')"
echo "$(date) V4 XONG" >> $S
