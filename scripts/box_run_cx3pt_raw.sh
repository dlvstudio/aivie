#!/bin/bash
# TK_108 cx3: tận dụng GPU 1 RẢNH trong lúc SFT cx3 chạy trên GPU 0 — đo expert CHỈ pretrain (cx3pt, chưa SFT)
# bằng HumanEval + MBPP để tách phần lợi của pretrain 1,5B token khỏi SFT. Tự DỪNG khi " cx3 SFT xong" xuất hiện
# (box_run_cx3_post.sh sẽ cần GPU 1 cho 2 máy chủ đo). Dùng: (setsid nohup bash scripts/box_run_cx3pt_raw.sh > /workspace/cx3pt_raw.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie AIVIE_SANDBOX_USER=sbx PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EV=/workspace/evalenv/bin/python; CE=scripts/code_eval.py
python - <<'PY' || { log "LỖI cx3pt thô: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/cx3pt_code.pt", map_location="cpu")
pos = sorted({int(k.split(".")[0]) for k in sd})
o["experts"]["code"] = {"state": {k: v.to(torch.bfloat16) for k, v in sd.items()}, "gated": True,
                        "hid": sd[f"{pos[0]}.w3.weight"].shape[1], "pos": pos}
torch.save(o, "/workspace/aivie_p3_cx3pt_bf16.pt"); print("ok", len(pos))
PY
P=""
for p in 8891 8892; do
  CUDA_VISIBLE_DEVICES=1 python -W ignore -u scripts/serve_hybrid_ui.py --ckpt /workspace/aivie_p3_cx3pt_bf16.pt --port $p --no-rag --max-ctx 4096 > /workspace/cx3pt_ui_$p.log 2>&1 &
  P="$P $!"
done
for p in 8891 8892; do W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI cx3pt thô: UI $p"; kill $P; exit 1; }; done; done
log "cx3pt thô (chưa SFT) đo HE + MBPP trên GPU 1 trong lúc SFT"
( $EV -u $CE run --suite humaneval --url http://localhost:8891 --tag cx3pt > /workspace/cx3pt_he.log 2>&1 \
    && log "cx3pt thô humaneval · $(grep '###' /workspace/cx3pt_he.log | tail -1 | cut -c1-200)" ) &
J1=$!
( $EV -u $CE run --suite mbpp --url http://localhost:8892 --tag cx3pt > /workspace/cx3pt_mbpp.log 2>&1 \
    && log "cx3pt thô mbpp · $(grep '###' /workspace/cx3pt_mbpp.log | tail -1 | cut -c1-200)" ) &
J2=$!
while kill -0 $J1 2>/dev/null || kill -0 $J2 2>/dev/null; do
  grep -q " cx3 SFT xong" $S && { log "cx3pt thô: SFT xong trước — dừng đo dở để nhường GPU 1"; pkill -P $J1; pkill -P $J2; kill $J1 $J2; break; }
  sleep 30
done
kill $P; sleep 5
log "cx3pt thô XONG"
