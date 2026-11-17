#!/bin/bash
# TK_108 CHẠY THẬT expert code LỚN (sau pilot cx/cx2): PRETRAIN RIÊNG expert 24 điểm × 8192 trên data/cx3 (~3,4 tỷ token,
# build_code_expert_data.py --profile cx3) bằng N GPU (torchrun, mặc định 4×H100). Tự chạy tiếp (--resume) khi bị ngắt, tối đa 6 lần.
# Token/bước = N × bs 4 × accum 4 × 2048 = 131K (N=4) → CX3_STEPS mặc định 26 000 bước ≈ 3,4 tỷ token.
# Dùng: (setsid nohup bash scripts/box_run_cx3_pt.sh > /workspace/cx3_pt_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/vlmenv/bin/python; [ -x $PY ] || PY=python   # box mới (NGC + setup_ngc_box.sh) dùng python hệ thống
N=${CX3_GPUS:-$(nvidia-smi -L | wc -l)}
STEPS=${CX3_STEPS:-26000}
log "cx3 pretrain bắt đầu: $N GPU · $STEPS bước × $((N*4*4*2048/1000))K token · $(tr -d '\n ' < data/cx3/manifest.json | cut -c1-200)"
for k in 1 2 3 4 5 6; do
  $PY -m torch.distributed.run --standalone --nproc_per_node $N scripts/expert_office_1p3b.py trainCPTX \
     --base ckpt/p3_sft.pt --code "" --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt \
     --pos every4 --raw-hid 8192 --cx-dir data/cx3 --cx-mix "opc_algo:0.23,opc_snip:0.15,opc_files:0.13,gh:0.26,web:0.08,vi:0.15" \
     --seq 2048 --bs 4 --accum 4 --lr 3e-4 --steps $STEPS --eval-every 1000 --resume --tag cx3pt >> /workspace/cx3_pt.log 2>&1 \
     && break
  log "LỖI cx3 pretrain lần $k (cx3_pt.log) — chạy tiếp từ trạng thái đã lưu sau 60 giây"
  sleep 60
done
grep -q "### cptx cx3pt" /workspace/cx3_pt.log || { log "LỖI cx3 pretrain: hết 6 lần thử"; exit 1; }
log "cx3 pretrain xong · $(grep '### cptx cx3pt' /workspace/cx3_pt.log | tail -1 | sed 's/### //' | cut -c1-300)"
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/cx3pt_code.pt expert_code/cx3pt_code.pt > /dev/null 2>&1 \
  && log "cx3 đã đẩy HF expert_code/cx3pt_code.pt"
log "cx3 pretrain XONG — chờ SFT"
