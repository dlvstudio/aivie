#!/bin/bash
# TK_108 C3-R2 (18/10) — đợt 2 quỹ đạo phục hồi, dồn vào kiểu lỗi cx2 còn mắc (10/15 bài viết hàm hỏng = sửa sai rồi gọi lại
# Y HỆT dù khung v3 nhắc): tỉ lệ tab 0,15 · bad 0,30 · repeat 0,45 · none 0,10, bài CHƯA dùng ở C3-R đầu → train_r2.jsonl.
# Chạy ngay (không chờ). Dùng: (setsid nohup bash scripts/box_run_c3r2.sh > /workspace/c3r2_run.log 2>&1 &)
# --- phần dưới giống box_run_c3r.sh ---
# TK_108 C3-R — quỹ đạo PHỤC HỒI (sau báo cáo C4 lần 2: 27/40 bài hỏng vì lặp lại khi gặp lỗi; quỹ đạo C3 cũ gần như
# không có cảnh "sai → đổi cách": 24/5.869). Tự chờ " C2b B XONG — " → vLLM Gemma → dựng bài có lượt SAI chèn sẵn
# (TAB, sửa sai, lặp lại; lượt chèn đánh nolearn) → thầy phục hồi tới khi test qua (≤15 bước, khung công cụ v2)
# → data/c3/train_r2.jsonl → HF. LUÔN ghi " C3-R XONG — " ở cuối (kể cả lỗi) để C4 lần 3 không chờ mãi.
# Dùng: (setsid nohup bash scripts/box_run_c3r.sh > /workspace/c3r_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
fin() { log "C3-R2 XONG — $*"; exit 0; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 AIVIE_SANDBOX_USER=sbx AIVIE_TOOL_GUARD=1
EV=/workspace/evalenv/bin/python; VLLM=/workspace/vllm_env/bin/vllm
$EV scripts/c3_teacher.py build_r --n 9000 --offset 8000 --weights 0.15,0.3,0.45,0.1 --out tasks_r2.jsonl > /workspace/c3r2_build.log 2>&1 || fin "LỖI dựng bài (c3r2_build.log) · không có dữ liệu"
log "C3-R2: $(grep '###' /workspace/c3r2_build.log | sed 's/### //')"
pkill -f "vllm serve"; sleep 20
VLLM_USE_DEEP_GEMM=0 CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 128 --port 8000 > /workspace/c3r2_vllm.log 2>&1 < /dev/null &
W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 160 ] && fin "LỖI vLLM · không có dữ liệu"; done
$EV -u scripts/c3_teacher.py run --tasks tasks_r2.jsonl --model gemma4-31b --tag r2 --offset 0 --workers 64 --steps 15 \
  --hours ${C3R_HOURS:-3} > /workspace/c3r2_main.log 2>&1 || log "LỖI C3-R2 sinh quỹ đạo (c3r2_main.log)"
pkill -f "vllm serve"
python3 - <<'PY' > /workspace/c3r2_data.log 2>&1
import json, collections
n, k = 0, collections.Counter()
with open("data/c3/train_r2.jsonl", "w", encoding="utf-8") as f:
    for l in open("data/c3/traj_r2.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["ok"] and r["valid"]:
            f.write(json.dumps({"id": r["id"], "kind": r["kind"], "rkind": r.get("rkind"), "messages": r["messages"]},
                               ensure_ascii=False) + "\n")
            n += 1; k[r.get("rkind")] += 1
print(n, dict(k))
PY
for f in tasks_r2.jsonl traj_r2.jsonl train_r2.jsonl; do [ -s data/c3/$f ] && hf upload donaldluong/nn-aivie-2p6b-p3 data/c3/$f data/c3/$f > /dev/null 2>&1; done
fin "$(grep '###' /workspace/c3r2_main.log | sed 's/### c3 //') · giữ (thành công + đúng giao thức) $(cat /workspace/c3r2_data.log) · đã đẩy HF"
