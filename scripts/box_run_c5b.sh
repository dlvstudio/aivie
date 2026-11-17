#!/bin/bash
# TK_108 C5b — RL THEO TEST (GRPO, c5b_grpo.py). Tự chờ " C5a XONG — " → chọn gốc: c5a nếu không tụt so với gốc của C5a
# (HumanEval+MBPP không kém quá 1 điểm, tool-use không kém quá 5 điểm, Excel ≥ 12, easy20 ≥ 15), ngược lại giữ gốc cũ
# → GRPO ${C5B_HOURS:-6} giờ (K=4 đề × G=8 lời giải, ≤512 token, 60% Python / 40% đa ngôn ngữ, thưởng = test chạy thật QUA)
# → gói aivie_p3_c5b → đo HumanEval / MBPP / tool-use / Excel + easy20 / MultiPL-E × 30 → HF.
# Dùng: (setsid nohup bash scripts/box_run_c5b.sh > /workspace/c5b_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q " C5a XONG — " $S; do sleep 120; done
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
PY=/workspace/vlmenv/bin/python; EV=/workspace/evalenv/bin/python
OLD=$(grep " C5a XONG — " $S | tail -1 | sed -n 's/.*gốc \(c4[bc]\).*/\1/p'); OLD=${OLD:-c4b}
BASE=$($EV - "$OLD" <<'PY'
import json, os, re, sys
old = sys.argv[1]
def s(suite, tag, key):
    p = f"out/code_eval/{suite}_{tag}.jsonl"
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    return 100 * sum(r[key] for r in rows) / len(rows) if rows else None
def off(tag):
    p = f"/workspace/{tag}_office.log"
    t = open(p).read() if os.path.exists(p) else ""
    m1, m2 = re.search(r"Excel (\d+)/27", t), re.search(r"easy20 (\d+)/20", t)
    return (int(m1.group(1)) if m1 else 0, int(m2.group(1)) if m2 else 0)
h5, m5, a5 = s("humaneval", "c5a", "pass1"), s("mbpp", "c5a", "pass1"), s("agent", "c5a", "ok")
ho, mo = s("humaneval", old, "pass1"), s("mbpp", old, "pass1")
ao = s("agent", old, "ok") if old == "c4c" else s("agent", "c4b_khungmoi", "ok")
ex, e20 = off("c5a")
ok = None not in (h5, m5, ho, mo) and (h5 + m5) >= (ho + mo) - 1 and (a5 or 0) >= (ao or 0) - 5 and ex >= 12 and e20 >= 15
print("c5a" if ok else old)
PY
)
log "C5b bắt đầu: GRPO trên expert $BASE · ${C5B_HOURS:-6} giờ · K=4 × G=8 · lr 1e-5 · thưởng = test chạy thật · kiểm cố định 24 đề mỗi 25 bước"
$PY -W ignore -u scripts/c5b_grpo.py --base ckpt/p3_sft.pt --code ckpt/expert_code/${BASE}_code.pt \
   --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --steps 100000 --hours ${C5B_HOURS:-6} \
   --K 4 --G 8 --max-new 512 --lr 1e-5 --kl 0.02 --py-frac 0.6 --tag c5b > /workspace/c5b_train.log 2>&1 \
   || { log "LỖI C5b: GRPO (c5b_train.log)"; [ -s ckpt/expert_code/c5b_code.pt ] || exit 1; }
log "C5b học xong · $(grep '###' /workspace/c5b_train.log | sed 's/### //') · $(grep '\[grpo\] bước' /workspace/c5b_train.log | tail -1 | cut -c1-160)"
$PY - <<'PY' > /workspace/c5b_pack.log 2>&1 || { log "LỖI C5b: gói"; exit 1; }
import torch
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load("ckpt/expert_code/c5b_code.pt", map_location="cpu")
assert set(sd) == set(o["experts"]["code"]["state"]), "khác khoá expert code"
o["experts"]["code"]["state"] = {k: v.to(torch.bfloat16) for k, v in sd.items()}
o.setdefault("merged_from", []).append({"code": "c5b_code (TK_108 C5b: GRPO thưởng theo test chạy thật)"})
torch.save(o, "ckpt/aivie_p3_c5b_bf16.pt"); print("ok")
PY
for d in ckpt checkpoints src data out; do ln -sfn /workspace/nn-aivie/$d /workspace/c1/$d; done
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
cp scripts/code_agent.py /workspace/c1/scripts/code_agent.py
CK=ckpt/aivie_p3_c5b_bf16.pt; CE=/workspace/c1/scripts/code_eval.py
PIDS=""
for p in 8871 8872 8873 8874; do
  $PY -W ignore -u /workspace/c1/scripts/serve_hybrid_ui.py --ckpt $CK --port $p --no-rag > /workspace/c5b_ui_$p.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in 8871 8872 8873 8874; do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI C5b: UI $p"; kill $PIDS; exit 1; }; done
done
ml() { $EV -u scripts/code_eval_ml.py run --url http://localhost:$2 --tag $1 --langs $3 --n 30 --workers 1 > /workspace/c5b_ml_$1.log 2>&1; }
JOBS=""
( $EV -u $CE run --suite humaneval --url http://localhost:8871 --tag c5b > /workspace/c5b_humaneval.log 2>&1 \
    && log "C5b humaneval · $(grep '###' /workspace/c5b_humaneval.log | tail -1 | cut -c1-200)" || log "LỖI C5b humaneval"
  $EV -u scripts/office_probe_light.py http://localhost:8871 c5b > /workspace/c5b_office.log 2>&1 \
    && log "C5b văn phòng · $(grep '###' /workspace/c5b_office.log)" || log "LỖI C5b văn phòng" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite mbpp --url http://localhost:8872 --tag c5b > /workspace/c5b_mbpp.log 2>&1 \
    && log "C5b mbpp · $(grep '###' /workspace/c5b_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI C5b mbpp" ) &
JOBS="$JOBS $!"
( $EV -u $CE run --suite agent --url http://localhost:8873 --tag c5b > /workspace/c5b_agent.log 2>&1 \
    && log "C5b agent (khung v3: tự khớp thụt lề) · $(grep '###' /workspace/c5b_agent.log | tail -1 | cut -c1-200)" || log "LỖI C5b agent"
  AIVIE_TOOL_GUARD=0 $EV -u $CE run --suite agent --url http://localhost:8873 --tag c5b_khungcu > /workspace/c5b_agent0.log 2>&1 \
    && log "C5b agent (khung cũ, so với c4c 35%) · $(grep '###' /workspace/c5b_agent0.log | tail -1 | cut -c1-200)"
  ml c5b-1 8873 js,ts,java,cs ) &
JOBS="$JOBS $!"
( ml c5b-2 8874 php,go,sh ) &
JOBS="$JOBS $!"
wait $JOBS
log "C5b MultiPL-E · $($EV scripts/code_eval_ml.py report --tags c5b-1+c5b-2 | cut -c1-220)"
kill $PIDS
hf upload donaldluong/nn-aivie-2p6b-p3-snap $CK aivie_2p6b_p3_c5b_bf16.pt > /dev/null 2>&1 \
  && log "C5b đã đẩy HF nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_c5b_bf16.pt"
hf upload donaldluong/nn-aivie-2p6b-p3 ckpt/expert_code/c5b_code.pt expert_code/c5b_code.pt > /dev/null 2>&1
hf upload donaldluong/nn-aivie-2p6b-p3 out/c5b/c5b_rollouts.jsonl c5b/c5b_rollouts.jsonl > /dev/null 2>&1
log "C5b XONG — gốc $BASE (so với C5a / C4 lần 3 ở trên)"
