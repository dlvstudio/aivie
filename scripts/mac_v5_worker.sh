#!/bin/bash
# M4 làm "người được hỏi" phụ cho vòng thầy v5: hội thoại i%15==0 (cùng seed với GPU), 1.3B chạy MPS, Gemma-31B gọi qua đường hầm SSH.
cd "$(dirname "$0")/.."
SSH="ssh -o ConnectTimeout=30 -i $HOME/.ssh/fpt_h100 -p 16500 root@<IP>"
RS="rsync -a --partial -e 'ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 -i $HOME/.ssh/fpt_h100 -p 16500'"
D=checkpoints/hf_hybrid/expert_code
for R in r1 r2; do
  PREV=$([ $R = r1 ] && echo v4c || echo v5r1); SEED=$([ $R = r1 ] && echo 1 || echo 2)
  until $SSH "grep -q 'V5 vòng $R: thầy chat' /workspace/code_status.log" 2>/dev/null; do sleep 120; done
  for n in code vp hoc; do
    until eval $RS root@<IP>:/workspace/nn-aivie/ckpt/expert_code/${PREV}_${n}_bf16.pt $D/; do sleep 30; done
  done
  pgrep -f "8000:localhost:8000" > /dev/null || { nohup ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -i $HOME/.ssh/fpt_h100 -p 16500 -L 8000:localhost:8000 root@<IP> > out/vllm_tunnel.log 2>&1 & sleep 5; }
  echo "$(date) M4 bắt đầu vòng $R ($PREV)"
  .venv/bin/python -W ignore scripts/v5_loop.py --n-conv 1000 --workers 1 --tag $R --seed $SEED --mod 15 --rem 0 --mode only \
     --device mps --out out/v5_${R}_mac.jsonl --base checkpoints/hf_hybrid/hybrid_1p3b_chat_v2.pt \
     --code $D/${PREV}_code_bf16.pt --office $D/${PREV}_vp_bf16.pt --hoc $D/${PREV}_hoc_bf16.pt
  until eval $RS out/v5_${R}_mac.jsonl root@<IP>:/workspace/nn-aivie/data/v5/${R}_mac.jsonl; do sleep 30; done
  $SSH "touch /workspace/nn-aivie/data/v5/${R}_mac.done"
  echo "$(date) M4 xong vòng $R: $(wc -l < out/v5_${R}_mac.jsonl) lượt"
done
