#!/bin/bash
# TK_106 V1 trên box: dữ liệu cho model đọc ảnh. Trạng thái → /workspace/vlm_status.log
# Bài học: container chỉ cho 1792 luồng (pids.max) → đừng chạy hàng chục Chrome cùng vLLM (vLLM sập "Resource temporarily unavailable").
cd /workspace/nn-aivie
S=/workspace/vlm_status.log
PY=/workspace/zenv/bin/python
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4
log() { echo "$(date '+%F %T') $*" >> $S; }

until grep -q "ảnh màn hình XONG" $S; do sleep 30; done
# 1. mắt SigLIP2 (lần trước hỏng vì hết luồng)
[ -f ckpt/vision/siglip2_so400m_384_bf16.safetensors ] || $PY scripts/fetch_vision_tower.py --out ckpt/vision/siglip2_so400m_384_bf16.safetensors > /workspace/vlm_tower.log 2>&1
log "mắt: $(ls -la ckpt/vision/ 2>/dev/null | tail -1 | awk '{print $5}') byte"
# 2. thầy Gemma-31B (nhìn ảnh)
/workspace/vllm_vlm.sh
# 3. tải ảnh (song song lúc vLLM nạp) — nguồn nhỏ trước
$PY scripts/gen_vlm_data.py fetch --only vichart,openvivqa,ktvic,coco > /workspace/vlm_fetch_a.log 2>&1
log "tải ảnh A: $(grep '###' /workspace/vlm_fetch_a.log | tr '\n' ' ')"
( $PY scripts/gen_vlm_data.py fetch --only cauldron > /workspace/vlm_fetch_b.log 2>&1
  log "tải ảnh B (cauldron): $(grep '###' /workspace/vlm_fetch_b.log | tr '\n' ' ')" ) & FB=$!
W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 80 ] && { log "LỖI V1: vLLM không lên (xem vllm_vlm.log)"; exit 1; }; done
log "vLLM sẵn sàng"
# 4. thử nhỏ thầy nhìn ảnh — bắt lỗi sớm
$PY scripts/gen_vlm_data.py vision --limit 24 --workers 24 > /workspace/vlm_vision_smoke.log 2>&1
N=$(wc -l < data/vlm/nat/vision.jsonl 2>/dev/null || echo 0)
[ "$N" -ge 12 ] || { log "LỖI V1: thầy nhìn ảnh thử chỉ được $N/24 (xem vlm_vision_smoke.log)"; exit 1; }
log "thầy nhìn ảnh thử: $N/24 OK"
# 5. thầy: nhìn ảnh (≈36K) + điền câu cần hiểu của ảnh màn hình
$PY scripts/gen_screens.py teach --out data/vlm/screens --split train --workers 32 > /workspace/vlm_teach.log 2>&1 &
TP=$!
$PY scripts/gen_vlm_data.py vision --workers 96 > /workspace/vlm_vision.log 2>&1
log "thầy nhìn ảnh: $(grep '###' /workspace/vlm_vision.log)"
wait $TP; log "ảnh màn hình (thầy điền): $(tail -1 /workspace/vlm_teach.log)"
$PY scripts/gen_screens.py teach --out data/vlm/screens --split test --workers 32 >> /workspace/vlm_teach.log 2>&1
# 6. dịch cauldron
wait $FB
$PY scripts/gen_vlm_data.py translate --workers 96 > /workspace/vlm_trans.log 2>&1
log "dịch cauldron: $(grep '###' /workspace/vlm_trans.log)"
# 7. gộp
$PY scripts/gen_vlm_data.py build > /workspace/vlm_build.log 2>&1
log "gộp: $(grep -E '###|nguồn' /workspace/vlm_build.log | tr '\n' ' ')"
log "V1 XONG · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}') · dữ liệu $(du -sh data/vlm | cut -f1)"
