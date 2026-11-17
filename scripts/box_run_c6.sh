#!/bin/bash
# TK_108 C6 (user đồng ý 23/10): HỌC TIẾP CẢ THÂN 2,49B, 10 tỷ token, nặng code — trên 2×A100 (torchrun, DDP).
# Vì sao: C7/C8 cho thấy trần ở thân đóng băng (bộ đo thực tế: expert 2,5% · + LoRA khối trộn 3,4–3,7%; r128 ≈ r32).
# Xuất phát cpt_b_stable.pt (pretrain TRƯỚC đoạn hạ lr, chưa SFT; HF công khai nn-aivie-hybrid-2p6b-stable) → WSD:
#   ấm 1000 bước → lr 1,5e-4 (= đỉnh cũ) giữ phẳng → 10% cuối hạ về 1,5e-5 trên dữ liệu CHỐT chất lượng cao.
# Trộn chính (≈ 10 tỷ token; mỗi token code dùng ~1 lần): code 45% · tài liệu kỹ thuật + Anh giáo dục 12% · ÔN tiếng Việt 30% · toán 8%.
# 2 GPU × bs 2 × accum 8 × ctx 8192 = 262 144 token/bước → ~38 150 bước. Lưu mỗi 500 bước (~2–3 giờ) → ckpt/c6.pt (có optimizer,
# chạy lại tự tiếp đúng bước). Bản sao lưu HF: box_c6_hf_sync.sh (trọng số bf16, repo riêng tư, gộp lịch sử).
# Tự chờ " C8 đo lại XONG" (box đang đo C8). Dùng: (setsid nohup bash scripts/box_run_c6.sh > /workspace/c6_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CPT=donaldluong/nn-aivie-cpt-data
# ── 0) dữ liệu (làm NGAY, song song lúc C8 còn đo) — chỉ tải phần cần, ổ box có hạn ──
mkdir -p data/cpt /workspace/c6
hf download $CPT --repo-type dataset --local-dir data/cpt --include "code/shard_code_*.npy" "math/*" "vi_wiki/*" "long/*" \
   "en_edu/shard_en_edu_0000[0-4].npy" "vi_web/shard_vi_fineweb2_0000[0-9].npy" "vi_web/shard_vi_hplt_0000[0-4].npy" \
   > /workspace/c6_dl.log 2>&1 || { log "LỖI C6: tải dữ liệu cpt"; exit 1; }
for d in data/cx3/gh data/cx3/opc_algo data/cx3/opc_snip data/cx3/opc_files data/cx3/web data/cx3/vi data/cpt/code data/cpt/math data/cpt/vi_wiki data/cpt/long data/cpt/en_edu data/cpt/vi_web; do [ -n "$(ls $d/shard_*.npy 2>/dev/null)" ] || { log "LỖI C6: thiếu $d (mẫu --include "code/*" từng bị bỏ qua, 23/10)"; exit 1; }; done
[ -s ckpt/c6.pt ] || [ -s /workspace/c6/cpt_b_stable.pt ] || \
  hf download donaldluong/nn-aivie-hybrid-2p6b-stable cpt_b_stable.pt --local-dir /workspace/c6 >> /workspace/c6_dl.log 2>&1 \
  || { log "LỖI C6: tải cpt_b_stable.pt"; exit 1; }
log "C6 chuẩn bị xong: dữ liệu $(du -sh data/cpt | cut -f1) cpt + data/cx3 · thân xuất phát $(ls -la /workspace/c6/cpt_b_stable.pt 2>/dev/null | awk '{print $5/1e9 " GB"}')"
until grep -q " C8 đo lại XONG" $S; do sleep 60; done
# ── 1) học tiếp thân ──
MIX="data/cx3/gh:0.12,data/cx3/opc_algo:0.09,data/cx3/opc_snip:0.06,data/cx3/opc_files:0.05,data/cpt/code:0.13"
MIX="$MIX,data/cx3/web:0.03,data/cpt/en_edu:0.09"
MIX="$MIX,data/cpt/vi_web:0.18,data/cpt/long:0.07,data/cpt/vi_wiki:0.02,data/cx3/vi:0.03"
MIX="$MIX,data/cpt/math:0.08"
# 28/10 (user đồng ý): đo theo lĩnh vực @20500 tiếng Việt 2,61→2,84 bit (vi_wiki, long tệ nhất) · opc_algo đã 0,26 bit → anneal tiếng Việt 35%→45%
ANNEAL="data/cx3/opc_algo:0.12,data/cx3/opc_files:0.15,data/cx3/gh:0.10,data/cpt/math:0.13,data/cpt/vi_wiki:0.15,data/cpt/long:0.15,data/cpt/vi_web:0.15,data/cpt/en_edu:0.05"
for try in 1 2 3 4 5 6; do
  if [ -s ckpt/c6.pt ]; then INIT="--init-from ckpt/c6.pt"; else
    INIT="--init-from /workspace/c6/cpt_b_stable.pt --reset-step"
    [ -s /workspace/c6/cpt_b_stable.pt ] || hf download donaldluong/nn-aivie-hybrid-2p6b-stable cpt_b_stable.pt --local-dir /workspace/c6 >> /workspace/c6_dl.log 2>&1
    # ổ box có hạn (thân xuất phát 30 GB + ckpt/c6.pt 30 GB): đã nạp vào RAM thì xoá bản tải về (tải lại được từ HF bất cứ lúc nào)
    ( N0=$(grep -c "bits TRƯỚC" /workspace/c6_train.log 2>/dev/null); until [ "$(grep -c 'bits TRƯỚC' /workspace/c6_train.log 2>/dev/null)" -gt "${N0:-0}" ]; do sleep 30; done
      rm -f /workspace/c6/cpt_b_stable.pt && log "C6: thân đã nạp → xoá bản tải cpt_b_stable.pt trên box (còn trên HF)" ) &
  fi
  log "C6 học thân (lần $try) · $INIT · 10 tỷ token · 2×A100"
  torchrun --standalone --nproc_per_node 2 scripts/train_hybrid_scale.py --data "$MIX" --anneal-data "$ANNEAL" \
     --ctx 8192 --bs 2 --grad-accum 8 --lr 1.5e-4 --min-lr 1.5e-5 --warmup 1000 --tokens 10e9 --sched wsd --decay-frac 0.1 \
     --grad-ckpt --eval-every 100 --save-every 500 $INIT --out ckpt/c6.pt --hf-repo "" >> /workspace/c6_train.log 2>&1 && break
  log "LỖI C6 học thân lần $try (c6_train.log: $(grep -E 'Error|error' /workspace/c6_train.log | tail -1 | cut -c1-150)) — chạy tiếp sau 2 phút"
  sleep 120
done
grep -q " XONG · LƯU ckpt/c6.pt" /workspace/c6_train.log || { log "LỖI C6: học thân không xong sau 6 lần"; exit 1; }
log "C6 học thân XONG · $(grep ' XONG · LƯU' /workspace/c6_train.log | tail -1 | cut -c1-200)"
