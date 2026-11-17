#!/bin/bash
# TK_108 chạy thật — SAU pretrain cx3 (box 2×A100): tự chờ " cx3 pretrain XONG — " →
#  0) chuẩn bị (làm NGAY khi khởi động, song song pretrain): dữ liệu SFT từ HF, dựng lại data/office_fix
#     (build_office_fix.py, tất định), model gói r1xl, môi trường đo /workspace/evalenv, user sandbox sbx
#  1) SFT expert cx3pt bằng trộn C4 lần 3 + C3-R2 (1500 bước, bs 1 × accum 16 như cx; KHÔNG bước bù tool-use — cx ≈ cx2)
#     → đẩy HF NGAY (phòng box bị thu hồi)
#  2) gói aivie_p3_cx3 → đẩy HF snap
#  3) đo: 4 máy chủ = 2 / GPU (expert lớn ~20 GB/máy chủ; 4 máy chủ / 1 GPU đã tràn 18/10), ngữ cảnh 4096:
#     tool-use khung v3 · HumanEval · MBPP · Excel + easy20 → log + kết quả đẩy HF
# Dùng: (setsid nohup bash scripts/box_run_cx3_post.sh > /workspace/cx3_post.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P3=donaldluong/nn-aivie-2p6b-p3
# ── 0) chuẩn bị ──
hf download $P3 --include "data/c2/oci_verified.jsonl" "data/c2/debug_v2.jsonl" "data/c2b/mix_*.jsonl" \
   "data/c3/train.jsonl" "data/c3/train_r.jsonl" "data/c3/train_r2.jsonl" --local-dir . > /workspace/cx3_prep.log 2>&1 \
   || { log "LỖI cx3 chuẩn bị: tải dữ liệu SFT"; exit 1; }
[ -s /workspace/p3data/p3_data.tar.zst ] || hf download donaldluong/nn-aivie-p3-data --repo-type dataset --local-dir /workspace/p3data >> /workspace/cx3_prep.log 2>&1
for t in p3_data code_sft; do tar --zstd -xf /workspace/p3data/$t.tar.zst --exclude="._*" 2>> /workspace/cx3_prep.log; done
python -c "import transformers" 2> /dev/null || PIP_CONSTRAINT= pip install -q transformers >> /workspace/cx3_prep.log 2>&1   # UI nạp Siglip (21/10: thiếu → UI chết)
for f in data/multiturn/train.jsonl data/code_tb/train.jsonl data/office_v2/train.jsonl data/v5/sft.jsonl data/office/eval_office.jsonl \
         data/eval_clean/easy20.jsonl data/code_eval/agent_tasks.jsonl data/c2/oci_verified.jsonl data/c2/debug_v2.jsonl \
         data/c3/train.jsonl data/c3/train_r.jsonl data/c3/train_r2.jsonl; do [ -s $f ] || { log "LỖI cx3 chuẩn bị: thiếu $f"; exit 1; }; done
mkdir -p out/code_eval
python scripts/build_office_fix.py data/office_v2/train.jsonl:data/office_fix/off.jsonl data/v5/sft.jsonl:data/office_fix/fix.jsonl >> /workspace/cx3_prep.log 2>&1 \
   || { log "LỖI cx3 chuẩn bị: dựng office_fix"; exit 1; }
hf download donaldluong/nn-aivie-2p6b-p3-snap aivie_2p6b_p3_r1xl_bf16.pt --local-dir /workspace/snap >> /workspace/cx3_prep.log 2>&1 \
   && ln -sf /workspace/snap/aivie_2p6b_p3_r1xl_bf16.pt ckpt/aivie_p3_r1xl_bf16.pt || { log "LỖI cx3 chuẩn bị: tải r1xl"; exit 1; }
[ -x /workspace/evalenv/bin/python ] || { python -m venv /workspace/evalenv && PIP_CONSTRAINT= /workspace/evalenv/bin/pip install -q "datasets>=3" "numpy>=2" pandas openpyxl formulas >> /workspace/cx3_prep.log 2>&1; }
# NGC đặt PIP_CONSTRAINT → venv bị ghim datasets 2.2.1 + fsspec mới → load_dataset lỗi "'**' can only be an entire path component" (21/10)
id sbx > /dev/null 2>&1 || useradd -M -s /bin/bash sbx
chmod 755 /workspace
EV=/workspace/evalenv/bin/python
$EV -c "import datasets, openpyxl, formulas" || { log "LỖI cx3 chuẩn bị: evalenv"; exit 1; }
AIVIE_SANDBOX_USER=sbx PYTHONPATH=scripts $EV -c "import code_agent as A; r=A.run_python('print(6*7)'); assert '42' in r['out'], r" >> /workspace/cx3_prep.log 2>&1 \
   || { log "LỖI cx3 chuẩn bị: sandbox sbx không chạy được code"; exit 1; }
log "cx3 chuẩn bị xong: dữ liệu SFT (c3r2 $(wc -l < data/c3/train_r2.jsonl) · office_fix dựng lại $(wc -l < data/office_fix/off.jsonl)+$(wc -l < data/office_fix/fix.jsonl)) · r1xl · evalenv · sbx"
# ── chờ pretrain ──
until grep -q " cx3 pretrain XONG — " $S; do sleep 120; done
[ -s ckpt/expert_code/cx3pt_code.pt ] || { log "LỖI cx3 sau pretrain: thiếu cx3pt_code.pt"; exit 1; }
# ── 1) SFT ──
MIX="c2bV:0.22,c2bA:0.11,c2bP:0.11,c2bF:0.08,c3r:0.08,c3r2:0.06,c3:0.07,c2oci:0.04,c2dbg2:0.04,tb:0.04,mt:0.08,fixfix:0.04,offfix:0.03"
log "cx3 SFT bắt đầu: expert cx3pt · 1500 bước · $MIX"
CUDA_VISIBLE_DEVICES=0 python -W ignore -u scripts/expert_office_1p3b.py trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/cx3pt_code.pt \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code --mix "$MIX" \
   --lr-old 1e-4 --steps 1500 --bs 1 --accum 16 --tag cx3 > /workspace/cx3_sft.log 2>&1 \
   || { log "LỖI cx3 SFT (cx3_sft.log)"; exit 1; }
log "cx3 SFT xong · $(grep '\[gen\] sau' /workspace/cx3_sft.log | cut -c1-250)"
hf upload $P3 ckpt/expert_code/cx3_code.pt expert_code/cx3_code.pt > /dev/null 2>&1 && log "cx3 đã đẩy HF expert_code/cx3_code.pt"
# ── 2) gói ──
python - <<'PY' > /workspace/cx3_pack.log 2>&1 || { log "LỖI cx3 gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/cx3_code.pt", map_location="cpu")
pos = sorted({int(k.split(".")[0]) for k in sd})
o["experts"]["code"] = {"state": {k: v.to(torch.bfloat16) for k, v in sd.items()}, "gated": True,
                        "hid": sd[f"{pos[0]}.w3.weight"].shape[1], "pos": pos}
o.setdefault("merged_from", []).append({"code": "cx3_code (TK_108 chạy thật: expert code LỚN pretrain 1,5 tỷ token cx3 + SFT C4 lần 3 + C3-R2)"})
torch.save(o, "ckpt/aivie_p3_cx3_bf16.pt"); print("ok", len(pos))
PY
hf upload donaldluong/nn-aivie-2p6b-p3-snap ckpt/aivie_p3_cx3_bf16.pt aivie_2p6b_p3_cx3_bf16.pt > /dev/null 2>&1 \
  && log "cx3 đã đẩy HF snap aivie_2p6b_p3_cx3_bf16.pt"
# ── 3) đo ──
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie AIVIE_SANDBOX_USER=sbx
CK=ckpt/aivie_p3_cx3_bf16.pt; CE=scripts/code_eval.py
PIDS=""
for p in 8881 8882 8883 8884; do
  G=$(( (p - 8881) / 2 ))                                     # 8881-8882 → GPU 0 · 8883-8884 → GPU 1
  CUDA_VISIBLE_DEVICES=$G python -W ignore -u scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag --max-ctx 4096 > /workspace/cx3_ui_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in 8881 8882 8883 8884; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI cx3: UI $p"; kill $PIDS; exit 1; }; done
done
log "cx3: $(grep '\[expert\] code' /workspace/cx3_ui_8881.log | head -1)"
JOBS=""
( $EV -u $CE run --suite agent --url http://localhost:8881 --tag cx3 > /workspace/cx3_agent.log 2>&1 \
    && log "cx3 agent (khung v3 · ctx 4096; cx2 47,5) · $(grep '###' /workspace/cx3_agent.log | tail -1 | cut -c1-200)" || log "LỖI cx3 agent" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite humaneval --url http://localhost:8882 --tag cx3 > /workspace/cx3_humaneval.log 2>&1 \
    && log "cx3 humaneval (cx2 13,4) · $(grep '###' /workspace/cx3_humaneval.log | tail -1 | cut -c1-200)" || log "LỖI cx3 humaneval" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite mbpp --url http://localhost:8883 --tag cx3 > /workspace/cx3_mbpp.log 2>&1 \
    && log "cx3 mbpp (cx2 15,6) · $(grep '###' /workspace/cx3_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI cx3 mbpp" ) &
JOBS="$JOBS $!"
( $EV -u scripts/office_probe_light.py http://localhost:8884 cx3 > /workspace/cx3_office.log 2>&1 \
    && log "cx3 văn phòng (cx2 Excel 14 · easy20 17) · $(grep '###' /workspace/cx3_office.log)" || log "LỖI cx3 văn phòng" ) &
JOBS="$JOBS $!"
wait $JOBS
ml() { $EV -u scripts/code_eval_ml.py run --url http://localhost:$2 --tag $1 --langs $3 --n 30 --workers 1 > /workspace/cx3_ml_$1.log 2>&1; }
MJ=""
for x in "cx3-1 8881 js,ts" "cx3-2 8882 java,cs" "cx3-3 8883 php,go" "cx3-4 8884 sh"; do ( ml $x ) & MJ="$MJ $!"; done
wait $MJ                                                       # KHÔNG 'wait' trơn — sẽ chờ cả máy chủ UI (không bao giờ thoát)
log "cx3 MultiPL-E (cx ~5,7) · $($EV scripts/code_eval_ml.py report --tags cx3-1+cx3-2+cx3-3+cx3-4 | cut -c1-220)"
log "cx3 lỗi máy chủ: $(cat /workspace/cx3_ui_*.log | grep -c '\[lỗi /api/chat\]') lần"
kill $PIDS
tar czf /workspace/cx3_ketqua.tgz out/code_eval/*cx3* out/office_probe_cx3.jsonl ckpt/expert_code/*cx3*.json /workspace/cx3_*.log /workspace/p3_status.log 2>/dev/null
hf upload $P3 /workspace/cx3_ketqua.tgz backup/cx3/cx3_ketqua.tgz > /dev/null 2>&1 && log "cx3 đã đẩy HF kết quả đo + log"
log "cx3 sau pretrain XONG — so với cx2: HE 13,4 · MBPP 15,6 · tool v3 47,5 · Excel 14/27 · easy20 17/20 · ML ~5,7 (cx)"
