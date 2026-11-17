#!/bin/bash
# TK_108 cx3: ĐẨY TỪNG PHẦN lên HF trong lúc pretrain (phòng box bị thu hồi đột xuất). Chạy song song, không đụng tiến trình train.
# Mỗi khi trainCPTX lưu xong (cx3pt_state.pt được os.replace sau cx3pt_code.pt → state mới = cả cặp đã đủ) → chép ra
# thư mục tạm → đẩy expert + trạng thái chạy tiếp + log lên repo riêng donaldluong/nn-aivie-cx3-ckpt → GỘP LỊCH SỬ
# (super_squash) để chỉ giữ bản mới nhất (không thì mỗi lần 14,5 GB cộng dồn vào hạn mức riêng tư).
# Chạy tiếp trên box mới:
#   hf download donaldluong/nn-aivie-cx3-ckpt --local-dir ckpt/expert_code  (→ cx3pt_code.pt + cx3pt_state.pt)
#   CX3_STEPS=<như cũ> bash scripts/box_run_cx3_pt.sh   (đã có --resume; đổi số GPU thì tính lại CX3_STEPS theo token còn lại)
# Dùng: (setsid nohup bash scripts/box_cx3_hf_sync.sh > /workspace/cx3_sync.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
REPO=donaldluong/nn-aivie-cx3-ckpt
ST=ckpt/expert_code/cx3pt_state.pt; CO=ckpt/expert_code/cx3pt_code.pt; TMP=/workspace/cx3_sync
export HF_HUB_DISABLE_XET=1
python - <<PY
from huggingface_hub import HfApi
HfApi().create_repo("$REPO", private=True, exist_ok=True)
PY
LAST=0
while true; do
  if [ -s $ST ]; then
    M=$(stat -c %Y $ST)
    if [ "$M" != "$LAST" ]; then
      sleep 30                                             # chờ ghi xong hẳn (os.replace là nguyên tử, thêm biên an toàn)
      mkdir -p $TMP && cp $CO $TMP/cx3pt_code.pt && cp $ST $TMP/cx3pt_state.pt \
        && cp ckpt/expert_code/trainCPTX_cx3pt.json $TMP/ 2>/dev/null; cp /workspace/cx3_pt.log /workspace/p3_status.log $TMP/
      STEP=$(python -c "import json;print(json.load(open('$TMP/trainCPTX_cx3pt.json'))['eval'][-1]['step'])" 2>/dev/null)
      if python - "$TMP" "$REPO" "$STEP" <<'PY'
import sys, os
from huggingface_hub import HfApi, CommitOperationAdd
tmp, repo, step = sys.argv[1:4]
api = HfApi()
ops = [CommitOperationAdd(f, os.path.join(tmp, f)) for f in sorted(os.listdir(tmp))]
api.create_commit(repo, ops, commit_message=f"cx3 pretrain bước {step}")
api.super_squash_history(repo, commit_message=f"cx3 pretrain — bản mới nhất: bước {step}")
PY
      then log "cx3 đã đẩy HF bước $STEP (expert + trạng thái chạy tiếp + log) → $REPO"; LAST=$M
      else log "LỖI cx3 đẩy HF bước $STEP — thử lại sau 10 phút"; sleep 600; fi
    fi
  fi
  grep -q " cx3 pretrain XONG — " $S && [ "$(stat -c %Y $ST 2>/dev/null)" = "$LAST" ] && { log "cx3 đẩy HF: dừng (pretrain đã xong, bản cuối đã đẩy)"; exit 0; }
  sleep 60
done
