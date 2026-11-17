#!/bin/bash
# TK_108 (user đồng ý 26/10): A100 làm việc phụ trong lúc C6 chạy trên 4×H100 (H100 chỉ chạy tới bước 13 000, chiều 26/10):
#  b) sinh THÊM đề C7 (thầy Gemma-4-31B FP8, 2 vLLM): data/c7/train.jsonl 3 000 → 6 000 (c7_real.py tự bỏ đề đã thử,
#     trùng, và đề trùng bộ đo) · đẩy HF từng phần mỗi 30 phút
#  a) đo bits THEO LĨNH VỰC (c6_domain_bits.py): thân xuất phát cpt_b_stable · c6.pt @13000 (khi H100 gửi về)
#  → rồi chạy TIẾP C6 trên A100 (box_run_c6.sh + box_c6_hf_sync.sh) từ ckpt/c6.pt H100 gửi về.
# Dùng: (setsid nohup bash scripts/box_run_c6_side.sh > /workspace/c6_side.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P3=donaldluong/nn-aivie-2p6b-p3
EV=/workspace/evalenv/bin/python
up() { hf upload $P3 "$1" "$2" > /dev/null 2>&1; }
STOP_STEP=13000
# ── b) thêm đề C7 ──
cp -n data/c7/train.jsonl /workspace/c7_train_3000.bak.jsonl
LLM=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1
for g in 0 1; do
  curl -s localhost:800$g/v1/models | grep -q gemma4-31b && continue
  LD_LIBRARY_PATH=/workspace/vllm_env/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH VLLM_USE_DEEP_GEMM=0 \
  CUDA_VISIBLE_DEVICES=$g setsid nohup /workspace/vllm_env/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 128 --port 800$g > /workspace/c7_vllm_$g.log 2>&1 < /dev/null &
  sleep 60
done
OKV=1
for g in 0 1; do
  W=0; until curl -s localhost:800$g/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 120 ] && { OKV=0; break; }; done
done
if [ $OKV = 1 ]; then
  log "C7 thêm đề: thầy Gemma-4-31B FP8 (2 vLLM) · train 3 000 → 6 000"
  ( while sleep 1800; do up data/c7/train.jsonl data/c7/train.jsonl; [ -f /workspace/c7_more_done ] && break; done ) &
  rm -f /workspace/c7_more_done
  $EV -u scripts/c7_real.py gen --split train --keep 6000 --workers 200 --llm $LLM > /workspace/c7_more_gen.log 2>&1 \
    && { touch /workspace/c7_more_done; up data/c7/train.jsonl data/c7/train.jsonl
         tar czf /workspace/c7_teacher_log.tgz data/c7/teacher_log && up /workspace/c7_teacher_log.tgz data/c7/teacher_log.tgz
         log "C7 thêm đề XONG · $(grep '###' /workspace/c7_more_gen.log | cut -c1-250) · đã đẩy HF"; } \
    || { touch /workspace/c7_more_done; log "LỖI C7 thêm đề (c7_more_gen.log) — giữ phần đã sinh, đi tiếp"; up data/c7/train.jsonl data/c7/train.jsonl; }
else
  log "LỖI C7 thêm đề: vLLM không lên (c7_vllm_*.log) — bỏ qua, đi tiếp phần đo"
fi
for p in $(pgrep -f "vllm serve"); do kill $p; done; sleep 30
# ── a) đo bits theo lĩnh vực ──
mkdir -p /workspace/c6 out
[ -s /workspace/c6/cpt_b_stable.pt ] || hf download donaldluong/nn-aivie-hybrid-2p6b-stable cpt_b_stable.pt --local-dir /workspace/c6 > /workspace/c6_dl.log 2>&1
# (không đo bản giữa chừng: H100 gửi c6.pt về mỗi ~55 phút, giữ thêm 1 bản 30 GB sẽ đầy ổ)
python -u scripts/c6_domain_bits.py --ckpt /workspace/c6/cpt_b_stable.pt --out out/c6_domain_bits.json \
  > /workspace/c6_bits.log 2>&1 && log "C6 bits theo lĩnh vực · $(grep '###' /workspace/c6_bits.log | cut -c5-200 | tr '\n' ' ')" \
  || log "LỖI C6 bits theo lĩnh vực (c6_bits.log)"
# ── chờ H100 dừng ở STOP_STEP rồi đo bản đó + chạy tiếp C6 ──
until grep -q "C6 trên H100 DỪNG" $S; do sleep 60; done
python -u scripts/c6_domain_bits.py --ckpt ckpt/c6.pt --out out/c6_domain_bits.json >> /workspace/c6_bits.log 2>&1 \
  && log "C6 bits theo lĩnh vực · $(grep '###' /workspace/c6_bits.log | tail -1 | cut -c5-200)" || log "LỖI C6 bits (bản H100)"
up out/c6_domain_bits.json out/c6_domain_bits.json
rm -f /workspace/c6/cpt_b_stable.pt
(setsid nohup bash scripts/box_c6_hf_sync.sh > /workspace/c6_sync.log 2>&1 < /dev/null &)
log "C6 chạy tiếp trên A100 từ ckpt/c6.pt (H100 gửi về)"
exec bash scripts/box_run_c6.sh >> /workspace/c6_run.log 2>&1
