#!/bin/bash
# chờ C2b B XONG + vLLM của C3-R đã lên → dịch đề Hán/Nhật/Hàn trong verified_* sang tiếng Việt (chạy song song C3-R,
# chia GPU) → xong trước khi C4 lần 3 gộp dữ liệu (C4 lần 3 chờ " C3-R XONG — ", ~2 giờ sau).
S=/workspace/p3_status.log; log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q " C2b B XONG — " $S; do sleep 60; done
W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 20; W=$((W+1)); [ $W -gt 90 ] && { log "LỖI dịch đề Hán: không có vLLM"; exit 1; }; done
/workspace/evalenv/bin/python -u scripts/c2b_cjk_vi.py --workers 24 > /workspace/c2b_cjk.log 2>&1 || { log "LỖI dịch đề Hán (c2b_cjk.log)"; exit 1; }
for f in data/c2b/verified_*.jsonl; do hf upload donaldluong/nn-aivie-2p6b-p3 $f $f > /dev/null 2>&1; done
log "C2b: $(grep '###' /workspace/c2b_cjk.log | sed 's/### //') · đã đẩy lại HF verified_*"
