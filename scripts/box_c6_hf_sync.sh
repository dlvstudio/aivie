#!/bin/bash
# TK_108 C6: ĐẨY TỪNG PHẦN lên HF trong lúc học thân (phòng box bị thu hồi). Mỗi khi ckpt/c6.pt được lưu lại (mỗi 500 bước):
# rút TRỌNG SỐ bf16 (~5 GB, bỏ optimizer 20 GB để không chạm hạn mức riêng tư) + cfg + bước → repo RIÊNG TƯ
# donaldluong/nn-aivie-c6-ckpt → gộp lịch sử (chỉ giữ bản mới nhất). Mất box: chạy tiếp từ trọng số này (optimizer khởi lại, ấm lại lr).
# Dùng: (setsid nohup bash scripts/box_c6_hf_sync.sh > /workspace/c6_sync.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export HF_HUB_DISABLE_XET=1
REPO=donaldluong/nn-aivie-c6-ckpt; CK=ckpt/c6.pt; TMP=/workspace/c6_sync
python - <<PY
from huggingface_hub import HfApi
HfApi().create_repo("$REPO", private=True, exist_ok=True)
PY
LAST=0
while true; do
  if [ -s $CK ]; then
    M=$(stat -c %Y $CK)
    if [ "$M" != "$LAST" ]; then
      sleep 120                                             # torch.save ghi đè tại chỗ → chờ ghi xong hẳn
      mkdir -p $TMP
      STEP=$(python - "$CK" "$TMP" <<'PY'
import sys, torch
o = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
torch.save({"cfg": o["cfg"], "step": o["step"], "state": {k: v.to(torch.bfloat16) for k, v in o["state"].items()}},
           f"{sys.argv[2]}/c6_w_bf16.pt")
print(o["step"])
PY
)
      cp /workspace/c6_train.log $TMP/ 2>/dev/null
      if [ -n "$STEP" ] && python - "$TMP" "$REPO" "$STEP" <<'PY'
import sys, os
from huggingface_hub import HfApi, CommitOperationAdd
tmp, repo, step = sys.argv[1:4]
api = HfApi()
api.create_commit(repo, [CommitOperationAdd(f, os.path.join(tmp, f)) for f in sorted(os.listdir(tmp))],
                  commit_message=f"C6 học thân bước {step}")
api.super_squash_history(repo, commit_message=f"C6 học thân — bản mới nhất: bước {step}")
PY
      then log "C6 đã đẩy HF bước $STEP (trọng số bf16 + log) → $REPO"; LAST=$(stat -c %Y $CK)   # đọc lại: torch.save có thể còn ghi sau 120s → mtime đổi → đẩy trùng
      else log "LỖI C6 đẩy HF bước $STEP — thử lại sau 10 phút"; sleep 600; fi
    fi
  fi
  grep -q " C6 học thân XONG" $S && [ "$(stat -c %Y $CK 2>/dev/null)" = "$LAST" ] && { log "C6 đẩy HF: dừng (học thân xong, bản cuối đã đẩy)"; exit 0; }
  sleep 120
done
