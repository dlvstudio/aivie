#!/bin/bash
# Sửa Excel sau P3.3' (Excel 17/27 → 9/27: công thức viết lẫn trong câu + vài lỗi logic). Tự chờ "C1 XONG" (GPU rảnh) rồi:
#   1) chuẩn hoá dữ liệu: công thức luôn trong khối ```excel (build_office_fix.py)
#   2) học thêm CHỈ expert văn phòng (thân + code + hoc đóng băng) trên dữ liệu đã chuẩn hoá + ôn hội thoại
#   3) gói checkpoint aivie_p3_r1xl = p3r1 với expert văn phòng mới
#   4) đo p3r1 và p3r1xl song song: Excel 27 bài chạy công thức thật + easy20 → ghi status
# Dùng: (setsid nohup bash scripts/box_run_office_fix.sh > /workspace/officefix_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
X="$PY -W ignore -u scripts/expert_office_1p3b.py"
until grep -q "C1 XONG" $S; do sleep 120; done
log "Sửa Excel bắt đầu (chuẩn hoá công thức trong khối + học thêm expert văn phòng)"
PIP_CONSTRAINT= /workspace/evalenv/bin/pip -q install openpyxl formulas > /workspace/officefix_pip.log 2>&1 || { log "LỖI sửa Excel: cài formulas"; exit 1; }
$PY scripts/build_office_fix.py data/office_v2/train.jsonl:data/office_fix/off.jsonl data/v5/sft.jsonl:data/office_fix/fix.jsonl \
  > /workspace/officefix_data.log 2>&1 || { log "LỖI sửa Excel: dữ liệu"; exit 1; }
log "Sửa Excel dữ liệu · $(tr '\n' ' ' < /workspace/officefix_data.log | sed 's#data/##g' | cut -c1-250)"
$X trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/p3r1_code.pt --office ckpt/expert_code/p3r1_vp.pt \
   --hoc ckpt/expert_code/p3r1_hoc.pt --train vp --mix offfix:0.6,fixfix:0.2,mt:0.1,calib:0.1 \
   --lr-old 2e-5 --steps 400 --tag p3r1xl > /workspace/officefix_train.log 2>&1 || { log "LỖI sửa Excel: học (officefix_train.log)"; exit 1; }
log "Sửa Excel học xong · $(grep '\[gen\] sau' /workspace/officefix_train.log | cut -c1-250)"
$PY - <<'EOF' > /workspace/officefix_pack.log 2>&1 || { log "LỖI sửa Excel: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/p3r1xl_vp.pt", map_location="cpu")
assert set(sd) == set(o["experts"]["office"]["state"]), "khác khoá expert văn phòng"
o["experts"]["office"]["state"] = {k: v.to(torch.bfloat16) for k, v in sd.items()}
o.setdefault("merged_from", []).append({"office": "p3r1xl_vp (sửa Excel: công thức trong khối)"})
torch.save(o, "ckpt/aivie_p3_r1xl_bf16.pt"); print("ok")
EOF
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
port=8821; PIDS=""
for t in r1 r1xl; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt ckpt/aivie_p3_${t}_bf16.pt --port $port --no-rag > /workspace/officefix_ui_$t.log 2>&1 &
  PIDS="$PIDS $!"; port=$((port+1))
done
for port in 8821 8822; do
  W=0; until curl -s localhost:$port > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI sửa Excel: UI $port"; kill $PIDS; exit 1; }; done
done
port=8821
for t in r1 r1xl; do
  $EV -u scripts/office_probe_light.py http://localhost:$port p3$t > /workspace/officefix_probe_$t.log 2>&1 &
  port=$((port+1))
done
while pgrep -f office_probe_light > /dev/null; do sleep 30; done
kill $PIDS
log "Sửa Excel đo xong · $(grep -h '###' /workspace/officefix_probe_r1.log /workspace/officefix_probe_r1xl.log | tr '\n' '|')"
hf upload donaldluong/nn-aivie-2p6b-p3-snap ckpt/aivie_p3_r1xl_bf16.pt aivie_2p6b_p3_r1xl_bf16.pt > /dev/null 2>&1 \
  && log "Sửa Excel đã đẩy HF nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_r1xl_bf16.pt"
log "Sửa Excel XONG"
