#!/bin/bash
# TK_108 C5a — DPO trên expert code (user cho phép chạy C5 ngay sau C4 lần 3). Tự chờ " C4 lần 3 XONG — " →
# chọn gốc: c4c nếu KHÔNG tụt (Excel ≥ 12/27, easy20 ≥ 15/20, HumanEval+MBPP không kém c4b quá 2 điểm), ngược lại c4b
# → DPO: cùng đề, lời giải ĐÚNG đã chạy qua ↔ lời giải SAI của thầy (data/c2b/dpo_code.jsonl, c2b_fixmix.py)
# → gói aivie_p3_c5a → đo HumanEval / MBPP / tool-use (khung mới) / Excel + easy20 / MultiPL-E × 30 → HF.
# Dùng: (setsid nohup bash scripts/box_run_c5a.sh > /workspace/c5a_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q " C4 lần 3 đo lại XONG — " $S; do sleep 120; done   # lần đo đầu hỏng (fork bomb) → chờ lần đo lại
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
BASE=$($EV - <<'PY'
import json, os, re
def s(suite, tag, key):
    p = f"out/code_eval/{suite}_{tag}.jsonl"
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    return 100 * sum(r[key] for r in rows) / len(rows) if rows else None
def off(tag):
    t = open(f"/workspace/{tag}_office.log").read() if os.path.exists(f"/workspace/{tag}_office.log") else ""
    m1, m2 = re.search(r"Excel (\d+)/27", t), re.search(r"easy20 (\d+)/20", t)
    return (int(m1.group(1)) if m1 else 0, int(m2.group(1)) if m2 else 0)
hc, mc, hb, mb = s("humaneval", "c4c", "pass1"), s("mbpp", "c4c", "pass1"), s("humaneval", "c4b", "pass1"), s("mbpp", "c4b", "pass1")
ex, e20 = off("c4c")
ok = None not in (hc, mc, hb, mb) and ex >= 12 and e20 >= 15 and (hc + mc) >= (hb + mb) - 2
print("c4c" if ok else "c4b")
PY
)
log "C5a bắt đầu: gốc = expert $BASE · DPO trên $(wc -l < data/c2b/dpo_code.jsonl) cặp đúng ↔ sai (nhật ký thầy)"
$PY -W ignore -u scripts/expert_office_1p3b.py trainDPO --base ckpt/p3_sft.pt --code ckpt/expert_code/${BASE}_code.pt \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code \
   --dpo-data data/c2b/dpo_code.jsonl --resp-max 1024 --max-len 2048 --steps 600 --bs 4 --lr-old 5e-6 --beta 0.1 \
   --tag c5a > /workspace/c5a_train.log 2>&1 || { log "LỖI C5a: học DPO (c5a_train.log)"; exit 1; }
log "C5a học xong · $(grep '\[dpo\] bước' /workspace/c5a_train.log | tail -1 | cut -c1-200)"
$PY - <<'PY' > /workspace/c5a_pack.log 2>&1 || { log "LỖI C5a: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/c5a_code.pt", map_location="cpu")
assert set(sd) == set(o["experts"]["code"]["state"]), "khác khoá expert code"
o["experts"]["code"]["state"] = {k: v.to(torch.bfloat16) for k, v in sd.items()}
o.setdefault("merged_from", []).append({"code": "c5a_code (TK_108 C5a: DPO đúng ↔ sai trên nhật ký thầy)"})
torch.save(o, "ckpt/aivie_p3_c5a_bf16.pt"); print("ok")
PY
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
cp scripts/code_agent.py /workspace/c1/scripts/code_agent.py
CK=ckpt/aivie_p3_c5a_bf16.pt; CE=/workspace/c1/scripts/code_eval.py
PIDS=""
for p in 8861 8862 8863 8864; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag > /workspace/c5a_ui_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in 8861 8862 8863 8864; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI C5a: UI $p"; kill $PIDS; exit 1; }; done
done
ml() { $EV -u scripts/code_eval_ml.py run --url http://localhost:$2 --tag $1 --langs $3 --n 30 --workers 1 > /workspace/c5a_ml_$1.log 2>&1; }
JOBS=""
( $EV -u $CE run --suite humaneval --url http://localhost:8861 --tag c5a > /workspace/c5a_humaneval.log 2>&1 \
    && log "C5a humaneval · $(grep '###' /workspace/c5a_humaneval.log | tail -1 | cut -c1-200)" || log "LỖI C5a humaneval"
  $EV -u scripts/office_probe_light.py http://localhost:8861 c5a > /workspace/c5a_office.log 2>&1 \
    && log "C5a văn phòng · $(grep '###' /workspace/c5a_office.log)" || log "LỖI C5a văn phòng" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite mbpp --url http://localhost:8862 --tag c5a > /workspace/c5a_mbpp.log 2>&1 \
    && log "C5a mbpp · $(grep '###' /workspace/c5a_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI C5a mbpp" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite agent --url http://localhost:8863 --tag c5a > /workspace/c5a_agent.log 2>&1 \
    && log "C5a agent (khung mới) · $(grep '###' /workspace/c5a_agent.log | tail -1 | cut -c1-200)" || log "LỖI C5a agent"
  ml c5a-1 8863 js,ts,java,cs ) &
JOBS="$JOBS $!"
( ml c5a-2 8864 php,go,sh ) &
JOBS="$JOBS $!"
wait $JOBS
log "C5a MultiPL-E · $($EV scripts/code_eval_ml.py report --tags c5a-1+c5a-2 | cut -c1-220)"
kill $PIDS
hf upload donaldluong/nn-aivie-2p6b-p3-snap $CK aivie_2p6b_p3_c5a_bf16.pt > /dev/null 2>&1 \
  && log "C5a đã đẩy HF nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_c5a_bf16.pt"
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/c5a_code.pt expert_code/c5a_code.pt > /dev/null 2>&1
hf upload donaldluong/nn-aivie-2p6b-p3 data/c2b/dpo_code.jsonl data/c2b/dpo_code.jsonl > /dev/null 2>&1
log "C5a XONG — gốc $BASE (so với kết quả C4 lần 3 ở trên)"
