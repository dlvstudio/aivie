#!/bin/bash
# TK_106 V1 (sắp xếp lại): HF chỉ ~1,5MB/s → tải các nguồn SONG SONG, thầy làm ngay phần không cần chờ ảnh
# (điền câu cần hiểu của ảnh màn hình), rồi nhìn ảnh dần theo lượng ảnh đã tải. Trạng thái → /workspace/vlm_status.log
cd /workspace/nn-aivie
S=/workspace/vlm_status.log
PY=/workspace/zenv/bin/python
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4
log() { echo "$(date '+%F %T') $*" >> $S; }
G="$PY scripts/gen_vlm_data.py"
$G fetch --only openvivqa > /workspace/vlm_fetch_ovqa.log 2>&1 & F1=$!
$G fetch --only ktvic --n-ktvic 2500 > /workspace/vlm_fetch_ktvic.log 2>&1 & F2=$!
$G fetch --only coco --n-coco 30000 > /workspace/vlm_fetch_coco.log 2>&1 & F3=$!
$G fetch --only cauldron > /workspace/vlm_fetch_caul.log 2>&1 & F4=$!
log "V1b: tải song song openvivqa/ktvic/coco/cauldron; thầy điền ảnh màn hình"
$PY scripts/gen_screens.py teach --out data/vlm/screens --split train --workers 64 > /workspace/vlm_teach.log 2>&1
$PY scripts/gen_screens.py teach --out data/vlm/screens --split test --workers 64 >> /workspace/vlm_teach.log 2>&1
log "ảnh màn hình (thầy điền): $(grep '###' /workspace/vlm_teach.log | tr '\n' ' ')"
# thử nhỏ thầy nhìn ảnh
until [ -s data/vlm/nat/raw.jsonl ]; do sleep 30; done
$G vision --limit 24 --workers 24 > /workspace/vlm_vision_smoke.log 2>&1
N=$(wc -l < data/vlm/nat/vision.jsonl 2>/dev/null || echo 0)
[ "$N" -ge 12 ] || { log "LỖI V1: thầy nhìn ảnh thử chỉ được $N/24 (xem vlm_vision_smoke.log)"; exit 1; }
log "thầy nhìn ảnh thử: $N/24 OK"
# nhìn ảnh theo từng đợt cho tới khi tải xong ảnh tự nhiên
while kill -0 $F1 2>/dev/null || kill -0 $F2 2>/dev/null || kill -0 $F3 2>/dev/null; do
  $G vision --workers 96 >> /workspace/vlm_vision.log 2>&1; sleep 60
done
$G vision --workers 96 >> /workspace/vlm_vision.log 2>&1
log "tải ảnh: $(grep -h '###' /workspace/vlm_fetch_ovqa.log /workspace/vlm_fetch_ktvic.log /workspace/vlm_fetch_coco.log | tr '\n' ' ')"
log "thầy nhìn ảnh: $(wc -l < data/vlm/nat/vision.jsonl)/$(wc -l < data/vlm/nat/raw.jsonl)"
wait $F4
log "tải cauldron: $(grep '###' /workspace/vlm_fetch_caul.log)"
$G translate --workers 128 > /workspace/vlm_trans.log 2>&1
log "dịch cauldron: $(grep '###' /workspace/vlm_trans.log)"
$G build > /workspace/vlm_build.log 2>&1
log "gộp: $(grep -E '###|nguồn' /workspace/vlm_build.log | tr '\n' ' ')"
log "V1 XONG · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}') · dữ liệu $(du -sh data/vlm | cut -f1)"
