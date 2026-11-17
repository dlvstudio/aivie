#!/bin/bash
# TK_108 (user duyệt 05/11): VÒNG THẦY CHUNG chữ + ảnh + tool trên SFT-B (thân C6, đọc ảnh c6b) — 2×A100.
#   0  dựng bài tool mới (c3_teacher build, bỏ bài đã có trong dữ liệu C3)
#   1  ~5h song song:
#        GPU 0: thầy Gemma-31B (vLLM) + vòng chat–chấm–sửa A (vlm_loop, 3 tiến trình)
#        GPU 1: vòng chat–chấm–sửa B (vlm_loop, 3 tiến trình, seed khác) + 2 máy chủ B tự giải bài tool (sandbox, chạy test thật)
#        → bài B hỏng: thầy giải lại (cùng khung công cụ) — chỉ giữ quỹ đạo test QUA
#   2  gộp: chữ (bản sửa / câu tốt) + tool (thành công) + ôn sft_b → data/sft_c6/loop1.jsonl · ảnh → data/vlm/fixmix_c6l1.jsonl
#   3  song song: GPU 1 học tiếp TOÀN THÂN (train_hybrid_sft, 1 epoch, lr nhỏ) · GPU 0 học tiếp đọc ảnh trên bản sửa có ảnh
#      (thân B — phần đọc ảnh chuyển qua thân cùng gốc gần như nguyên vẹn: c6b0 ≈ c6b)
#   4  đo: 6 làn code/tool/văn phòng (như c6sftb) + 6 bộ ảnh + giám khảo + vòng thầy nhỏ đo lại điểm chấm TB → gói checkpoint
# Repo HF riêng tư có thể đầy → checkpoint KHÔNG đẩy; Mac kéo về box_backup/. KHÔNG đẩy dữ liệu / repo công khai.
# Dùng: (setsid nohup bash scripts/box_run_c6_loop.sh > /workspace/c6_loop_run.log 2>&1 < /dev/null &)
cd /workspace/nn-aivie
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
unset PIP_CONSTRAINT
VLLM=/workspace/vllm_env; EV=/workspace/evalenv/bin/python; CE=scripts/code_eval.py
B=ckpt/c6_sft_b.pt; VB=ckpt/vlm/c6b_s2.pt
TAG=c6l1; NEW=ckpt/c6_loop1.pt
NCA=${NCA:-400}; NCB=${NCB:-600}; TOOL_H=${TOOL_H:-4.5}   # GPU 0 chung vLLM → 2 tiến trình; GPU 1 → 3
T="--tower ckpt/vision/siglip2_so400m_384_bf16.safetensors"
A4="--shuf 2 --no-thumb --max-tiles 8 --max-len 3072"
V="python -W ignore -u scripts/vlm_1p3b.py"
D=/workspace/c6loop; mkdir -p $D
step() { [ -f $D/$1.done ]; }
done_() { touch $D/$1.done; log "vòng thầy $TAG: $1 xong · $2"; }
vllm_on() {
  curl -s localhost:8000/v1/models | grep -q gemma4-31b && return
  CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
    --max-model-len 8192 --gpu-memory-utilization $1 --max-num-seqs 64 --port 8000 > /workspace/c6loop_vllm.log 2>&1 < /dev/null &
  local W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 120 ] && { log "LỖI vòng thầy: vLLM không lên"; exit 1; }; done
}
vllm_off() { for p in $(pgrep -f "vllm serve"); do kill $p; done; sleep 30; }
ui() {  # $1 GPU $2 cổng $3 ckpt → chờ lên
  CUDA_VISIBLE_DEVICES=$1 setsid nohup python -W ignore -u scripts/serve_hybrid_ui.py --ckpt $3 --port $2 --no-rag --max-ctx 4096 \
    > /workspace/${TAG}_ui_$2.log 2>&1 < /dev/null &
  local W=0; until curl -s localhost:$2 > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI vòng thầy: UI $2 không lên"; return 1; }; done
  return 0   # until trả mã của lệnh cuối trong thân ([ W -gt 150 ] = 1) → không có dòng này ui "lỗi" mỗi khi phải chờ
}
ui_off() { for p in $(pgrep -f "serve_hybrid_ui.py --ckpt"); do kill $p; done; sleep 10; }
# 05/11 lần 1 chết lặng lẽ ngay sau khi bật UI 8891 (không lỗi, không OOM) → ghi vết + bắt tín hiệu; chạy lại được không nhân đôi
exec 2>> /workspace/c6_loop_trace.log; set -x
trap 'log "LỖI vòng thầy: script nhận tín hiệu (TERM/HUP/INT) — xem c6_loop_trace.log"; exit 1' TERM HUP INT
alive() { pgrep -f "$1" > /dev/null; }
waitp() { while alive "$1"; do sleep 60; done; }   # chờ theo tên (tiến trình có thể không phải con của script này)
log "vòng thầy $TAG bắt đầu trên SFT-B: chữ + ảnh ($((NCA+NCB)) hội thoại) + tool (${TOOL_H}h B tự giải → thầy giải bài B hỏng)"

# ── 0. bài tool mới ──
if ! step tasks; then
  python scripts/c3_teacher.py build --n 1500 > /workspace/c6loop_tasks.log 2>&1 || { log "LỖI vòng thầy: dựng bài tool"; exit 1; }
  python - <<'EOF' >> /workspace/c6loop_tasks.log 2>&1
import glob, json, random
used = set()
for p in glob.glob("data/c3/train*.jsonl") + glob.glob("data/c3/traj_*.jsonl"):
    for l in open(p):
        try: used.add(json.loads(l).get("id", ""))
        except Exception: pass
t = [json.loads(l) for l in open("data/c3/tasks.jsonl")]
new = [x for x in t if x["id"] not in used]
random.Random(5).shuffle(new)
with open("data/c3/tasks_c6l.jsonl", "w") as f:
    for x in new: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"### bài tool: {len(t)} dựng · {len(t)-len(new)} đã dùng · {len(new)} mới → data/c3/tasks_c6l.jsonl")
EOF
  done_ tasks "$(grep '###' /workspace/c6loop_tasks.log | tail -1)"
fi

# ── 1. vòng thầy ──
if ! step run; then
  vllm_on 0.60
  alive "vlm_loop.py run .*--tag ${TAG}a" || CUDA_VISIBLE_DEVICES=0 setsid nohup python -u scripts/vlm_loop.py run --base $B --vision $VB $T \
     --n-conv $NCA --workers 2 --bf16 --seed 0 --tag ${TAG}a >> /workspace/c6loop_run_a.log 2>&1 < /dev/null &
  alive "vlm_loop.py run .*--tag ${TAG}b" || CUDA_VISIBLE_DEVICES=1 setsid nohup python -u scripts/vlm_loop.py run --base $B --vision $VB $T \
     --n-conv $NCB --workers 3 --bf16 --seed 1 --tag ${TAG}b >> /workspace/c6loop_run_b.log 2>&1 < /dev/null &
  for P in 8891 8892; do curl -s localhost:$P > /dev/null || ui 1 $P $B || exit 1; done
  NT=$(wc -l < data/c3/tasks_c6l.jsonl); H=$((NT / 2))
  python -u scripts/c3_teacher.py run --tasks tasks_c6l.jsonl --offset 0 --n $H --llm http://localhost:8891 --model c6b \
     --tag ${TAG}_b0 --workers 1 --hours $TOOL_H > /workspace/c6loop_tool_b0.log 2>&1 & T0=$!
  python -u scripts/c3_teacher.py run --tasks tasks_c6l.jsonl --offset $H --n $H --llm http://localhost:8892 --model c6b \
     --tag ${TAG}_b1 --workers 1 --hours $TOOL_H > /workspace/c6loop_tool_b1.log 2>&1 & T1=$!
  wait $T0 $T1
  ui_off
  python - <<EOF > /workspace/c6loop_tool_fail.log 2>&1
import json
done = {}
for t in ("${TAG}_b0", "${TAG}_b1"):
    for l in open(f"data/c3/traj_{t}.jsonl"):
        r = json.loads(l); done[r["id"]] = r["ok"] and r["valid"]
tasks = [json.loads(l) for l in open("data/c3/tasks_c6l.jsonl")]
fail = [x for x in tasks if x["id"] in done and not done[x["id"]]]
with open("data/c3/tasks_c6l_fail.jsonl", "w") as f:
    for x in fail: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"### B tự giải {len(done)} bài · thành công {sum(done.values())} ({sum(done.values())/max(len(done),1):.0%}) · {len(fail)} bài hỏng → thầy")
EOF
  python -u scripts/c3_teacher.py run --tasks tasks_c6l_fail.jsonl --model gemma4-31b --tag ${TAG}_g --workers 24 \
     > /workspace/c6loop_tool_g.log 2>&1 || log "LỖI vòng thầy: thầy giải bài tool (c6loop_tool_g.log)"
  waitp "vlm_loop.py run .*--tag ${TAG}[ab]"
  vllm_off
  for x in a b; do grep -q "điểm TB" /workspace/c6loop_run_$x.log || { log "LỖI vòng thầy: vlm_loop $x (c6loop_run_$x.log)"; exit 1; }; done
  done_ run "A $(grep 'điểm TB' /workspace/c6loop_run_a.log | cut -c1-350) · B $(grep 'điểm TB' /workspace/c6loop_run_b.log | cut -c1-350) · $(grep '###' /workspace/c6loop_tool_fail.log) · thầy $(grep '###' /workspace/c6loop_tool_g.log | cut -c1-150)"
fi

# ── 2. gộp dữ liệu ──
if ! step build; then
  python -u scripts/vlm_loop.py build --tags ${TAG}a,${TAG}b > /workspace/c6loop_build.log 2>&1 || { log "LỖI vòng thầy: tách ảnh"; exit 1; }
  python -u scripts/build_c6_loop.py --v6 ${TAG}a,${TAG}b --traj ${TAG}_b0,${TAG}_b1,${TAG}_g --out data/sft_c6/loop1.jsonl \
     >> /workspace/c6loop_build.log 2>&1 || { log "LỖI vòng thầy: gộp chữ + tool"; exit 1; }
  python - <<EOF >> /workspace/c6loop_build.log 2>&1
import json, random
r = random.Random(33)
fix = [json.loads(l) for l in open("data/vlm/fix_${TAG}a_${TAG}b.jsonl")]
old = [json.loads(l) for l in open("data/vlm/sft5.jsonl")]; r.shuffle(old)
mix = fix * 2 + old[:20000]
r.shuffle(mix)
with open("data/vlm/fixmix_${TAG}.jsonl", "w") as f:
    for x in mix: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"### ảnh {len(fix)} bản sửa/câu tốt · {len(mix)} mẫu (×2 + ôn 20K)")
EOF
  done_ build "$(grep '###' /workspace/c6loop_build.log | tr '\n' ' ' | cut -c1-500)"
fi

# ── 3. học tiếp: thân (GPU 1) ‖ đọc ảnh (GPU 0) ──
if ! step train; then
  N=$(grep -c '' data/sft_c6/loop1.jsonl)
  CUDA_VISIBLE_DEVICES=1 python -u scripts/train_hybrid_sft.py --init-from $B --data data/sft_c6/loop1.jsonl \
    --max-len 8192 --tok-budget 16384 --bs 8 --grad-accum 4 --grad-ckpt --lr 5e-6 --min-lr 1e-6 --warmup 30 --epochs 1 \
    --eval-every 200 --save-every 500 --out $NEW > /workspace/c6loop_sft.log 2>&1 & PS=$!
  NV=$(grep -c '' data/vlm/fixmix_$TAG.jsonl); STEPS=$(( NV / 32 )); [ $STEPS -gt 1000 ] && STEPS=1000; [ $STEPS -lt 200 ] && STEPS=200
  CUDA_VISIBLE_DEVICES=0 $V train --stage 2 --base $B $T $A4 --init $VB --data data/vlm/fixmix_$TAG.jsonl --steps $STEPS --bs 32 \
     --lr 5e-5 --lr-expert 1e-4 --tower-last 6 --lr-tower 1e-5 --lora 64 --lr-lora 5e-5 --warmup 50 --val-every 500 --save-every 500 \
     --tok-budget 8000 --out ckpt/vlm/${TAG}_s2.pt > /workspace/c6loop_vtrain.log 2>&1 || log "LỖI vòng thầy: học đọc ảnh (c6loop_vtrain.log)"
  wait $PS
  grep -q "XONG SFT" /workspace/c6loop_sft.log || { log "LỖI vòng thầy: học tiếp thân (c6loop_sft.log)"; exit 1; }
  [ -s ckpt/vlm/${TAG}_s2.pt ] || { log "LỖI vòng thầy: không có ckpt/vlm/${TAG}_s2.pt"; exit 1; }
  done_ train "thân $N mẫu · $(grep -h -E 'TRƯỚC SFT|XONG SFT' /workspace/c6loop_sft.log | tr '\n' ' ' | cut -c1-160) · ảnh $STEPS bước · $(grep 'val trước' /workspace/c6loop_vtrain.log | grep -o 'val trước.*') → $(grep 'val sau' /workspace/c6loop_vtrain.log)"
fi

# ── 4. đo ──
if ! step evaltext; then
  lane() {  # $1 GPU $2 cổng $3… việc
    local G=$1 P=$2; shift 2
    ui $G $P $NEW || return 1
    local U=http://localhost:$P
    for job in "$@"; do
      case $job in
        agent)     $EV -u $CE run --suite agent --url $U --tag $TAG > /workspace/${TAG}_agent.log 2>&1 ;;
        humaneval) $EV -u $CE run --suite humaneval --url $U --tag $TAG > /workspace/${TAG}_humaneval.log 2>&1 ;;
        mbpp)      $EV -u $CE run --suite mbpp --url $U --tag $TAG > /workspace/${TAG}_mbpp.log 2>&1 ;;
        c7_*)      $EV -u scripts/c7_real.py eval --url $U --tag $TAG --shard ${job#c7_}/3 --workers 1 > /workspace/${TAG}_c7_${job#c7_}.log 2>&1 ;;
        office)    $EV -u scripts/office_probe_light.py $U $TAG > /workspace/${TAG}_office.log 2>&1 ;;
      esac
    done
    for p in $(pgrep -f "serve_hybrid_ui.py --ckpt $NEW --port $P"); do kill $p; done
  }
  lane 1 8891 agent & J1=$!; lane 1 8892 humaneval & J2=$!; lane 1 8893 mbpp & J3=$!
  lane 0 8894 c7_1 office & J4=$!; lane 0 8895 c7_2 & J5=$!; lane 0 8896 c7_3 & J6=$!
  wait $J1 $J2 $J3 $J4 $J5 $J6
  log "ĐO $TAG · C7 $($EV scripts/c7_real.py report --tag $TAG 2>/dev/null | cut -c1-200) · HE $(grep '###' /workspace/${TAG}_humaneval.log | tail -1 | cut -c1-90) · MBPP $(grep '###' /workspace/${TAG}_mbpp.log | tail -1 | cut -c1-90) · tool $(grep '###' /workspace/${TAG}_agent.log | tail -1 | cut -c1-140) · văn phòng $(grep '###' /workspace/${TAG}_office.log | cut -c1-120) · lỗi máy chủ $(cat /workspace/${TAG}_ui_889[1-6].log | grep -c '\[lỗi /api/chat\]')"
  touch $D/evaltext.done
fi
if ! step evalimg; then
  ev1() { local G=$1; shift; for t in "$@"; do
    CUDA_VISIBLE_DEVICES=$G $V evalvlm --base $NEW $T --vision ckpt/vlm/${TAG}_s2.pt --device cuda --tag $TAG --max-tiles 8 \
      --test data/vlm/test_$t.jsonl --n 200 > /workspace/c6loop_veval_$t.log 2>&1 || log "LỖI vòng thầy: đo ảnh $t"; done; }
  ev1 0 screen nat chart & P0=$!; ev1 1 ovqa news code & P1=$!; wait $P0 $P1
  vllm_on 0.60
  $V judgevlm --files "ckpt/vlm/eval_${TAG}_*.json" > /workspace/c6loop_vjudge.log 2>&1
  # đo lại bằng chính vòng thầy (150 hội thoại mới, seed khác) → so điểm chấm TB với lúc đầu
  CUDA_VISIBLE_DEVICES=1 python -u scripts/vlm_loop.py run --base $NEW --vision ckpt/vlm/${TAG}_s2.pt $T --n-conv 150 --workers 3 --bf16 \
     --seed 7 --tag ${TAG}e > /workspace/c6loop_rerun_new.log 2>&1 & R1=$!
  CUDA_VISIBLE_DEVICES=0 python -u scripts/vlm_loop.py run --base $B --vision $VB $T --n-conv 150 --workers 2 --bf16 \
     --seed 7 --tag ${TAG}e0 > /workspace/c6loop_rerun_b.log 2>&1 & R2=$!
  wait $R1 $R2   # 05/11: 'wait' trơn chờ cả vLLM (con của script) → kẹt mãi
  vllm_off
  done_ evalimg "ảnh: $(grep '###' /workspace/c6loop_vjudge.log | sed 's/### //' | tr '\n' '|' | cut -c1-700) · thầy chấm lại (cùng 150 hội thoại) B: $(grep 'điểm TB' /workspace/c6loop_rerun_b.log | cut -c1-300) → mới: $(grep 'điểm TB' /workspace/c6loop_rerun_new.log | cut -c1-300)"
fi

# ── 5. MỘT checkpoint ──
if ! step pack; then
  python -c "
import torch
sd = torch.load('ckpt/vlm/${TAG}_s2.pt', map_location='cpu')
out = {'proj': {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd['proj'].items()}, 'hid': sd['hid'], 'cfg': sd['cfg'],
       'expert': {k: v.to(torch.bfloat16) for k, v in sd['expert'].items()}, 'lora': sd.get('lora', {})}
if 'tower' in sd: out['tower'] = sd['tower']
torch.save(out, 'ckpt/vlm/${TAG}_vision_bf16.pt')"
  $V pack --base $NEW --vision ckpt/vlm/${TAG}_vision_bf16.pt $T --max-tiles 8 --out ckpt/${TAG}_vlm.pt > /workspace/c6loop_pack.log 2>&1 \
    || { log "LỖI vòng thầy: gói checkpoint"; exit 1; }
  python - <<EOF >> /workspace/c6loop_pack.log 2>&1
import torch
o = torch.load("ckpt/${TAG}_vlm.pt", map_location="cpu", weights_only=False)
o["state"] = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in o["state"].items()}
o.pop("opt", None)
torch.save(o, "ckpt/${TAG}_vlm_bf16.pt")
EOF
  rm -f ckpt/${TAG}_vlm.pt
  tar czf /workspace/${TAG}_ketqua.tgz out/code_eval/*${TAG}* out/office_probe_${TAG}.jsonl ckpt/vlm/eval_${TAG}_*.json /workspace/${TAG}_*.log /workspace/c6loop_*.log 2>/dev/null
  hf upload donaldluong/nn-aivie-c6-sft-b /workspace/${TAG}_ketqua.tgz results/${TAG}_ketqua.tgz > /dev/null 2>&1 && log "$TAG đã đẩy HF kết quả đo"
  done_ pack "$(du -h ckpt/${TAG}_vlm_bf16.pt | cut -f1) · Mac kéo về box_backup/c6sft/"
fi
log "vòng thầy $TAG XONG — so 'ĐO $TAG' với 'ĐO c6sftb' + ảnh $TAG với c6b + điểm thầy chấm lại → chọn bản / vòng 2"
