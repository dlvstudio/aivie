#!/bin/bash
# TK_108 C3 qua đêm: tự chờ C4 sớm xong → user sandbox → dựng bài dự án nhỏ → thử nhỏ chọn thầy (Gemma-4-31B vs
# Qwen3-Coder-30B-A3B, Apache-2.0, cùng 50 bài) → thầy thắng sinh quỹ đạo dùng công cụ tới khoảng 07:30 giờ VN → đẩy HF.
# Trạng thái → /workspace/p3_status.log. Dùng: (setsid nohup bash scripts/box_run_c3.sh > /workspace/c3_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
until grep -q "C4 sớm XONG" $S; do sleep 120; done
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1
EV=/workspace/evalenv/bin/python; VLLM=/workspace/vllm_env/bin/vllm
QWEN=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8

# ── 0. cách ly: công cụ của thầy (bash, chạy test) chạy dưới user sbx, KHÔNG ghi được /workspace ──
id sbx > /dev/null 2>&1 || useradd -M -s /usr/sbin/nologin sbx
chmod o-w /workspace; chmod -R o-w /workspace/nn-aivie /workspace/c1 2>/dev/null
export AIVIE_SANDBOX_USER=sbx
CHK=$($EV - <<'PY'
import sys; sys.path.insert(0, "scripts"); import code_agent as A
r1 = A.run_python("open('/workspace/nn-aivie/_c3_probe','w').write('x')"); r2 = A.run_python("print(1+1)")
print("OK" if r1["rc"] != 0 and r2["out"].strip() == "2" else f"FAIL {r1} {r2}")
PY
)
[ "$CHK" = "OK" ] || { log "LỖI C3: sandbox chưa chặn ghi /workspace ($CHK)"; exit 1; }
rm -f /workspace/nn-aivie/_c3_probe
log "C3 bắt đầu: sandbox user sbx đã kiểm (không ghi được /workspace) · dựng bài + tải Qwen3-Coder FP8"

# ── 1. dựng bài + tải Qwen song song ──
hf download $QWEN > /workspace/c3_qwen_dl.log 2>&1 &
DL=$!
$EV -u scripts/c3_teacher.py build --n 30000 --workers 48 > /workspace/c3_build.log 2>&1 || { log "LỖI C3: dựng bài"; exit 1; }
log "C3 dựng bài · $(grep '###' /workspace/c3_build.log | sed 's/### c3 build: //')"

vllm_on() {   # $1 = repo, $2 = tên phục vụ
  pkill -f "vllm serve"; sleep 20
  VLLM_USE_DEEP_GEMM=0 CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve $1 --served-model-name $2 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --max-num-seqs 128 --port 8000 > /workspace/c3_vllm_$2.log 2>&1 < /dev/null &
  W=0; until curl -s localhost:8000/v1/models | grep -q "$2"; do sleep 15; W=$((W+1)); [ $W -gt 160 ] && return 1; done
  return 0      # until trả mã của lệnh cuối trong thân vòng ([ … ] sai = 1) → phải trả 0 rõ ràng
}
score() { grep '###' /workspace/c3_pilot_$1.log | sed -n 's/.*thành công \([0-9]*\).*/\1/p'; }

# ── 2. thử nhỏ chọn thầy: cùng 50 bài đầu ──
vllm_on RedHatAI/gemma-4-31B-it-FP8-dynamic gemma4-31b || { log "LỖI C3: vLLM Gemma"; exit 1; }
$EV -u scripts/c3_teacher.py run --model gemma4-31b --tag pilot_gemma --n 50 --offset 0 --workers 50 > /workspace/c3_pilot_gemma.log 2>&1
wait $DL
G=$(score gemma); Q=-1
if vllm_on $QWEN qwen3-coder; then
  $EV -u scripts/c3_teacher.py run --model qwen3-coder --tag pilot_qwen --n 50 --offset 0 --workers 50 > /workspace/c3_pilot_qwen.log 2>&1
  Q=$(score qwen)
else
  log "C3: Qwen không lên được (c3_vllm_qwen3-coder.log) → dùng Gemma"
fi
G=${G:-0}; Q=${Q:-0}
if [ "$Q" -gt "$G" ]; then WIN=qwen3-coder; REPO=$QWEN; else WIN=gemma4-31b; REPO=RedHatAI/gemma-4-31B-it-FP8-dynamic; fi
log "C3 thử nhỏ 50 bài · Gemma thành công $G · Qwen3-Coder $Q → thầy: $WIN"

# ── 3. thầy thắng sinh quỹ đạo tới khoảng 07:30 giờ VN (00:30 UTC) ──
curl -s localhost:8000/v1/models | grep -q "$WIN" || vllm_on $REPO $WIN || { log "LỖI C3: vLLM $WIN"; exit 1; }
END=$(date -u -d "tomorrow 00:30" +%s); [ $(date -u +%H) -lt 12 ] && END=$(date -u -d "today 00:30" +%s)
H=$(python3 -c "print(max(1.0, ($END - $(date +%s)) / 3600 - 0.3))")
log "C3 sinh quỹ đạo: thầy $WIN · ngân sách ${H} giờ"
$EV -u scripts/c3_teacher.py run --model $WIN --tag main --offset 100 --workers 64 --hours $H > /workspace/c3_main.log 2>&1 \
  || log "LỖI C3 sinh quỹ đạo (c3_main.log)"
pkill -f "vllm serve"
for f in tasks.jsonl traj_pilot_gemma.jsonl traj_pilot_qwen.jsonl traj_main.jsonl; do
  [ -s data/c3/$f ] && hf upload donaldluong/nn-aivie-2p6b-p3 data/c3/$f data/c3/$f > /dev/null 2>&1
done
log "C3 XONG · $(grep '###' /workspace/c3_main.log | sed 's/### c3 //') · đã đẩy HF nn-aivie-2p6b-p3/data/c3/"
