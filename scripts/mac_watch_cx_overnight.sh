#!/bin/bash
# Overnight watch TK_108 CX pilot. Log: out/cx_overnight_watch.log
# - Không restart nếu trainCPTX / SFT / eval đang chạy
# - Chỉ báo LỖI mới (sau dòng "Pilot expert lớn bắt đầu" cuối)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/out/cx_overnight_watch.log"
SSH=(ssh -o ConnectTimeout=25 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -i "$HOME/.ssh/vast_p3" -p 43856 root@<IP>)
mkdir -p "$ROOT/out"
ts() { date '+%F %T'; }
log() { echo "$(ts) $*" | tee -a "$LOG"; }

R=$(
  "${SSH[@]}" 'bash -s' <<'EOS' 2>/dev/null | grep -v -E 'Welcome|Have fun|authentication'
PHASE=none
pgrep -f 'trainCPTX' >/dev/null && PHASE=pretrain
pgrep -f 'expert_office_1p3b.py trainGEN' >/dev/null && PHASE=sft
pgrep -af 'serve_hybrid_ui.py' | grep -q 'aivie_p3_cx\|8881\|8882\|8883\|8884' && PHASE=eval
pgrep -f 'code_eval' >/dev/null && PHASE=eval
pgrep -f 'box_run_cx_pilot' >/dev/null && WRAP=1 || WRAP=0
GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' %')
DONE=$(grep -c 'Pilot expert lớn XONG' /workspace/p3_status.log 2>/dev/null || echo 0)
# LỖI sau lần bắt đầu THÀNH CÔNG cuối (bỏ LỖI 08:50 cũ trước restart 08:51)
LAST_START=$(grep -n 'Pilot expert lớn bắt đầu' /workspace/p3_status.log | tail -1 | cut -d: -f1)
NEW_ERR=$(tail -n +$((LAST_START+1)) /workspace/p3_status.log 2>/dev/null | grep 'LỖI pilot' | tail -1)
PT=$(grep '\[cptx\] bước' /workspace/cx_pt.log 2>/dev/null | tail -1)
SF=$(grep -E '\[gen\]|bước|XONG|LỖI' /workspace/cx_sft.log 2>/dev/null | tail -1)
STAT=$(tail -n 1 /workspace/p3_status.log)
DISK=$(df -h / | awk 'NR==2{print $4" free "$5}')
echo "PHASE=$PHASE|WRAP=$WRAP|GPU=${GPU:-0}|DONE=$DONE|DISK=$DISK"
echo "STAT=$STAT"
echo "PT=$PT"
echo "SF=$SF"
echo "NEW_ERR=$NEW_ERR"
EOS
)
RC=$?
if [[ $RC -ne 0 || -z "$R" ]]; then
  log "SSH fail rc=$RC — thử lại tick sau"
  echo 'AGENT_LOOP_WAKE_cx {"prompt":"CX overnight: SSH box thất bại. Thử SSH lại, kiểm tra Vast còn sống không, sửa kết nối nếu cần."}'
  exit 0
fi

echo "$R" >> "$LOG"
PHASE=$(echo "$R" | head -1 | sed -n 's/^PHASE=\([^|]*\).*/\1/p')
WRAP=$(echo "$R" | head -1 | sed -n 's/.*WRAP=\([01]\).*/\1/p')
GPU=$(echo "$R" | head -1 | sed -n 's/.*GPU=\([0-9]*\).*/\1/p')
DONE=$(echo "$R" | head -1 | sed -n 's/.*DONE=\([0-9]*\).*/\1/p')
NEW_ERR=$(echo "$R" | grep '^NEW_ERR=' | sed 's/^NEW_ERR=//')
PT=$(echo "$R" | grep '^PT=' | sed 's/^PT=//' | cut -c1-140)

if [[ "${DONE:-0}" -ge 1 ]]; then
  log "DONE — pilot XONG"
  echo 'AGENT_LOOP_WAKE_cx {"prompt":"CX pilot XONG. Đọc p3_status + kết quả đo, tóm tắt cho user, dừng loop theo dõi overnight."}'
  exit 0
fi

# Có job GPU đang chạy → chỉ ghi nhận
if [[ "$PHASE" != "none" ]]; then
  log "ok $PHASE gpu=${GPU:-?} wrap=$WRAP | $PT"
  # LỖI mới trong khi vẫn còn process? ghi nhận nhưng không restart
  if [[ -n "${NEW_ERR:-}" ]]; then
    log "note NEW_ERR while running: $NEW_ERR"
  fi
  exit 0
fi

# Không còn train/eval
if [[ -n "${NEW_ERR:-}" ]]; then
  log "FAIL NEW_ERR=$NEW_ERR — cần agent sửa"
  echo "AGENT_LOOP_WAKE_cx {\"prompt\":\"CX overnight LỖI pilot mới khi không còn train: $NEW_ERR. Đọc log, sửa và resume (pretrain/SFT/đo). User đang ngủ.\"}"
  exit 0
fi

if [[ "$WRAP" == "1" && "${GPU:-0}" -lt 8 ]]; then
  log "WARN wrapper sống, PHASE=none, GPU idle — pipeline kẹt?"
  echo 'AGENT_LOOP_WAKE_cx {"prompt":"CX overnight: box_run_cx_pilot còn sống nhưng không thấy train/SFT/eval và GPU idle. Kiểm tra script đang chờ gì (status/log) và đẩy tiếp hoặc sửa."}'
  exit 0
fi

if [[ "$WRAP" == "0" ]]; then
  # An toàn: chỉ restart wrapper nếu KHÔNG có train. Script sẽ thấy C5b XONG rồi chạy lại từ đầu pretrain —
  # CHỈ làm khi pretrain chưa xong (chưa có dòng Pilot pretrain xong). Nếu đã pretrain xong thì agent phải resume SFT/eval tay.
  HAS_PT_DONE=$(echo "$R" | grep -c 'Pilot pretrain xong' || true)
  # Check remote for pretrain complete marker
  PT_DONE=$("${SSH[@]}" 'grep -c "Pilot pretrain xong" /workspace/p3_status.log 2>/dev/null' 2>/dev/null | grep -v -E 'Welcome|Have fun|authentication' | tail -1)
  if [[ "${PT_DONE:-0}" -ge 1 ]]; then
    log "wrapper chết sau pretrain — KHÔNG auto restart full script (sẽ train lại từ đầu). Gọi agent resume SFT/eval."
    echo 'AGENT_LOOP_WAKE_cx {"prompt":"CX overnight: wrapper chết SAU khi pretrain xong. ĐỪNG chạy lại box_run_cx_pilot từ đầu. Resume từ SFT (trainGEN) hoặc đo theo box_run_cx_pilot.sh phần còn lại. User đang ngủ."}'
    exit 0
  fi
  log "FIX: wrapper+train đều chết, pretrain chưa xong — restart box_run_cx_pilot"
  "${SSH[@]}" 'cd /workspace/nn-aivie && (setsid nohup bash scripts/box_run_cx_pilot.sh > /workspace/cx_run.log 2>&1 &) && sleep 3 && pgrep -af box_run_cx_pilot | head -3' 2>/dev/null | grep -v -E 'Welcome|Have fun|authentication' | tee -a "$LOG"
  echo 'AGENT_LOOP_WAKE_cx {"prompt":"CX overnight: đã restart box_run_cx_pilot (pretrain chưa xong, mọi process chết). Xác nhận trainCPTX đã lên lại; nếu script train lại từ đầu trong khi đã có cxpt_code.pt thì cân nhắc --hours còn lại / resume."}'
  exit 0
fi

log "ok idle? phase=none wrap=$WRAP gpu=${GPU:-?} | $PT"
exit 0
