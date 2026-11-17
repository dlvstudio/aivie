#!/bin/bash
# TK_108 chạy thật — SFT BÙ TOOL-USE cho cx3 (user đồng ý 22/10). cx3: MBPP 22,2 (cx2 15,6) · Excel 16/27 nhưng
# tool v3 42,5 (cx2 47,5) · HE 12,2 (cx2 13,4) — cx3 SFT KHÔNG có bước bù tool-use như cx2 → làm y cx2: học tiếp từ cx3_code
# 600 bước, ~55% tool-use (C3-R + C3-R2 + C3), giữ code đã kiểm + văn phòng chống quên, LR 5e-5
# → đẩy HF NGAY → gói aivie_p3_cx3b → đo tool-use v3 · HumanEval · MBPP · Excel + easy20 (4 máy chủ = 2 / GPU, ctx 4096).
# Tự chờ " cx3 sau pretrain XONG — " (MultiPL-E cx3 còn giữ 4 máy chủ UI trên 2 GPU).
# Dùng: (setsid nohup bash scripts/box_run_cx3b.sh > /workspace/cx3b_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P3=donaldluong/nn-aivie-2p6b-p3
EV=/workspace/evalenv/bin/python
until grep -q " cx3 sau pretrain XONG — " $S; do sleep 60; done
[ -s ckpt/expert_code/cx3_code.pt ] || { log "LỖI cx3b: thiếu cx3_code.pt"; exit 1; }
MIX="c3r:0.18,c3r2:0.12,c3:0.25,c2bV:0.12,c2bF:0.06,c2bA:0.05,c2oci:0.04,c2dbg2:0.04,mt:0.06,offfix:0.04,fixfix:0.04"
log "cx3b bắt đầu: SFT bù tool-use từ cx3 · 600 bước · lr 5e-5 · $MIX"
CUDA_VISIBLE_DEVICES=0 python -W ignore -u scripts/expert_office_1p3b.py trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/cx3_code.pt \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code --mix "$MIX" \
   --lr-old 5e-5 --steps 600 --bs 1 --accum 16 --tag cx3b > /workspace/cx3b_sft.log 2>&1 \
   || { log "LỖI cx3b: SFT (cx3b_sft.log)"; exit 1; }
log "cx3b SFT xong · $(grep '\[gen\] sau' /workspace/cx3b_sft.log | cut -c1-250)"
hf upload $P3 ckpt/expert_code/cx3b_code.pt expert_code/cx3b_code.pt > /dev/null 2>&1 && log "cx3b đã đẩy HF expert_code/cx3b_code.pt"
python - <<'PY' > /workspace/cx3b_pack.log 2>&1 || { log "LỖI cx3b: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/cx3b_code.pt", map_location="cpu")
pos = sorted({int(k.split(".")[0]) for k in sd})
o["experts"]["code"] = {"state": {k: v.to(torch.bfloat16) for k, v in sd.items()}, "gated": True,
                        "hid": sd[f"{pos[0]}.w3.weight"].shape[1], "pos": pos}
o.setdefault("merged_from", []).append({"code": "cx3b_code (TK_108 chạy thật: cx3 + SFT bù tool-use 600 bước)"})
torch.save(o, "ckpt/aivie_p3_cx3b_bf16.pt"); print("ok", len(pos))
PY
hf upload donaldluong/nn-aivie-2p6b-p3-snap ckpt/aivie_p3_cx3b_bf16.pt aivie_2p6b_p3_cx3b_bf16.pt > /dev/null 2>&1 \
  && log "cx3b đã đẩy HF snap aivie_2p6b_p3_cx3b_bf16.pt"
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie AIVIE_SANDBOX_USER=sbx
CK=ckpt/aivie_p3_cx3b_bf16.pt; CE=scripts/code_eval.py
PIDS=""
for p in 8881 8882 8883 8884; do
  G=$(( (p - 8881) / 2 ))
  CUDA_VISIBLE_DEVICES=$G python -W ignore -u scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag --max-ctx 4096 > /workspace/cx3b_ui_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in 8881 8882 8883 8884; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI cx3b: UI $p"; kill $PIDS; exit 1; }; done
done
JOBS=""
( $EV -u $CE run --suite agent --url http://localhost:8881 --tag cx3b > /workspace/cx3b_agent.log 2>&1 \
    && log "cx3b agent (khung v3 · ctx 4096; cx3 42,5 · cx2 47,5) · $(grep '###' /workspace/cx3b_agent.log | tail -1 | cut -c1-200)" || log "LỖI cx3b agent" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite humaneval --url http://localhost:8882 --tag cx3b > /workspace/cx3b_humaneval.log 2>&1 \
    && log "cx3b humaneval (cx3 12,2 · cx2 13,4) · $(grep '###' /workspace/cx3b_humaneval.log | tail -1 | cut -c1-200)" || log "LỖI cx3b humaneval" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite mbpp --url http://localhost:8883 --tag cx3b > /workspace/cx3b_mbpp.log 2>&1 \
    && log "cx3b mbpp (cx3 22,2 · cx2 15,6) · $(grep '###' /workspace/cx3b_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI cx3b mbpp" ) &
JOBS="$JOBS $!"
( $EV -u scripts/office_probe_light.py http://localhost:8884 cx3b > /workspace/cx3b_office.log 2>&1 \
    && log "cx3b văn phòng (cx3 Excel 16 · easy20 16) · $(grep '###' /workspace/cx3b_office.log)" || log "LỖI cx3b văn phòng" ) &
JOBS="$JOBS $!"
wait $JOBS                                                     # KHÔNG 'wait' trơn — sẽ chờ cả máy chủ UI
log "cx3b lỗi máy chủ: $(cat /workspace/cx3b_ui_*.log | grep -c '\[lỗi /api/chat\]') lần"
kill $PIDS
tar czf /workspace/cx3b_ketqua.tgz out/code_eval/*cx3b* out/office_probe_cx3b.jsonl ckpt/expert_code/*cx3b*.json /workspace/cx3b_*.log 2>/dev/null
hf upload $P3 /workspace/cx3b_ketqua.tgz backup/cx3/cx3b_ketqua.tgz > /dev/null 2>&1 && log "cx3b đã đẩy HF kết quả đo + log"
log "cx3b XONG — so với cx3 (tool 42,5 · HE 12,2 · MBPP 22,2 · Excel 16 · easy20 16) và cx2 (47,5 · 13,4 · 15,6 · 14 · 17)"
