#!/bin/bash
# TK_108: ĐẨY TỪNG PHẦN một checkpoint đang học lên HF (bản chung của box_c6_hf_sync.sh). Mỗi lần $CK được lưu lại:
# rút trọng số bf16 + cfg + bước → repo RIÊNG TƯ → gộp lịch sử (chỉ giữ bản mới nhất, không tốn hạn mức).
# Dừng khi dòng "$DONE" có trong p3_status.log và bản cuối đã đẩy.
# Dùng: (setsid nohup bash scripts/box_hf_sync_bf16.sh ckpt/c6_sft_a.pt donaldluong/nn-aivie-c6-sft-a /workspace/c6sft_a.log " SFT-A XONG" > /workspace/sync_a.log 2>&1 &)
CK=$1; REPO=$2; LOGF=$3; DONE=$4
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export HF_HUB_DISABLE_XET=1
NAME=$(basename $CK .pt); TMP=/workspace/sync_$NAME
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
      STEP=$(python - "$CK" "$TMP/${NAME}_bf16.pt" <<'PY'
import sys, torch
o = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
torch.save({"cfg": o["cfg"], "step": o["step"], "state": {k: v.to(torch.bfloat16) for k, v in o["state"].items()}}, sys.argv[2])
print(o["step"])
PY
)
      cp $LOGF $TMP/ 2>/dev/null
      if [ -n "$STEP" ] && python - "$TMP" "$REPO" "$STEP" <<'PY'
import sys, os
from huggingface_hub import HfApi, CommitOperationAdd
tmp, repo, step = sys.argv[1:4]
api = HfApi()
api.create_commit(repo, [CommitOperationAdd(f, os.path.join(tmp, f)) for f in sorted(os.listdir(tmp))],
                  commit_message=f"bước {step}")
api.super_squash_history(repo, commit_message=f"bản mới nhất: bước {step}")
PY
      then log "$NAME đã đẩy HF bước $STEP (bf16 + log) → $REPO"; LAST=$(stat -c %Y $CK)
      else log "LỖI $NAME đẩy HF bước $STEP — thử lại sau 10 phút"; sleep 600; fi
    fi
  fi
  grep -q "$DONE" $S && [ "$(stat -c %Y $CK 2>/dev/null)" = "$LAST" ] && { log "$NAME đẩy HF: dừng (xong, bản cuối đã đẩy)"; exit 0; }
  sleep 120
done
