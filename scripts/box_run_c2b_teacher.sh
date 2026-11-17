#!/bin/bash
# TK_108 C2b phần B + chạy lại C2 debug — CÓ THẦY, CHẠY THẬT, LƯU NHẬT KÝ ĐẦY ĐỦ.
#  0. (CPU, chạy ngay) cài node+tsc · JDK 17 · .NET 8 · PHP 8.3 · Go 1.22 · Chromium (playwright) → tự kiểm từng ngôn ngữ
#     trong sandbox (lời giải đúng phải QUA, lời giải sai phải HỎNG)
#  1. chờ dòng " C4 lần 2 XONG — " (GPU rảnh) → bật vLLM Gemma-4-31B
#  2. C2 debug chạy lại (--tag _v2, thầy được sửa thêm 1 lượt) → data/c2/debug_v2.jsonl + data/c2/teacher_log/debug_v2.jsonl
#  3. C2b B: 7 ngôn ngữ + SQL + HTML, mỗi thứ tới 4000 bài, ngân sách C2B_HOURS giờ → data/c2b/verified_* + teacher_log/*
#  4. đẩy HF · trạng thái → /workspace/p3_status.log (tiền tố "C2b B")
# Dùng: (setsid nohup bash scripts/box_run_c2b_teacher.sh > /workspace/c2b_teacher_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 AIVIE_SANDBOX_USER=sbx DEBIAN_FRONTEND=noninteractive
EV=/workspace/evalenv/bin/python; VLLM=/workspace/vllm_env/bin/vllm; T=/workspace/c2b_tool.log

# ── 0. công cụ ──
command -v dotnet > /dev/null && command -v tsc > /dev/null && $EV -c 'import playwright' 2>/dev/null || { apt-get update -qq && apt-get install -y -qq nodejs npm openjdk-17-jdk-headless php-cli golang-go dotnet-sdk-8.0 \
  && npm install -g typescript@5.6.3 @types/node@20.16.10 \
  && PIP_CONSTRAINT= $EV -m pip -q install playwright==1.48.0 && $EV -m playwright install --with-deps chromium; } > $T 2>&1 \
  || { log "LỖI C2b B: cài công cụ ($T)"; exit 1; }
# SQL Server 2022 Express (user đồng ý EULA 16/10/2026) — chỉ nghe 127.0.0.1; mật khẩu sa + login c2b ngẫu nhiên, chỉ root đọc
PIP_CONSTRAINT= $EV -m pip -q install pymssql pillow >> $T 2>&1
if ! pgrep -x sqlservr > /dev/null; then
  (ACCEPT_EULA=Y MSSQL_PID=Express MSSQL_SA_PASSWORD="$(cat /root/.c2b_mssql_pw)" setsid nohup runuser -u mssql -- \
     /opt/mssql/bin/sqlservr > /workspace/mssql.log 2>&1 < /dev/null &)
  for i in $(seq 1 60); do grep -q "ready for client connections" /workspace/mssql.log && break; sleep 3; done
fi
( umask 077; [ -s /root/.c2b_mssql_login ] || python3 -c "import secrets,string; a=string.ascii_letters+string.digits; print('Bb2!'+''.join(secrets.choice(a) for _ in range(24)))" > /root/.c2b_mssql_login )
$EV scripts/c2b_teacher.py --mssql-login >> $T 2>&1 || { log "LỖI C2b B: tạo login SQL Server"; exit 1; }
mkdir -p /opt/c2b/gocache /opt/c2b/work && chmod 1777 /opt/c2b/gocache /opt/c2b/work
$EV -u scripts/c2b_teacher.py --selftest > /workspace/c2b_selftest.log 2>&1 \
  || { log "LỖI C2b B: tự kiểm khung chạy ($(grep HỎNG /workspace/c2b_selftest.log | cut -c1-200 | tr '\n' ' '))"; exit 1; }
log "C2b B: cài + tự kiểm OK (cả T-SQL trên SQL Server Express) — node $(node -v) · javac $(javac -version 2>&1 | cut -d' ' -f2) · dotnet $(dotnet --version) · php $(php -r 'echo PHP_VERSION;') · $(go version | cut -d' ' -f3) · Chromium · chờ GPU rảnh"

# ── 1. chờ GPU ──
until grep -q " C4 lần 2 XONG — " $S; do sleep 120; done   # chỉ chờ dòng XONG (lỗi lẻ một bộ đo không có nghĩa GPU đã rảnh)
pkill -f "vllm serve"; sleep 20
VLLM_USE_DEEP_GEMM=0 CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 128 --port 8000 \
  --limit-mm-per-prompt '{"image": 2, "audio": 0, "video": 0}' > /workspace/c2b_vllm.log 2>&1 < /dev/null &
W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 160 ] && { log "LỖI C2b B: vLLM"; exit 1; }; done

# ── 2. C2 debug chạy lại, lưu nhật ký đầy đủ ──
log "C2b B: C2 debug chạy lại (lưu nhật ký đầy đủ, thầy sửa thêm 1 lượt)"
$EV -u scripts/c2_teacher.py debug --n 20000 --tag _v2 --workers 96 > /workspace/c2_debug_v2.log 2>&1 || log "LỖI C2 debug _v2"
log "C2b B: $(grep '###' /workspace/c2_debug_v2.log | sed 's/### //')"

# ── 3. C2b B ──
log "C2b B: $(grep 'thầy nhìn ảnh' /workspace/c2b_teacher.log 2>/dev/null | head -1)"
log "C2b B: thầy viết test + chạy thật 7 ngôn ngữ + SQL + HTML · ngân sách ${C2B_HOURS:-6} giờ"
$EV -u scripts/c2b_teacher.py --per-lang 4000 --hours ${C2B_HOURS:-6} --workers 64 > /workspace/c2b_teacher.log 2>&1 \
  || log "LỖI C2b B (c2b_teacher.log)"
pkill -f "vllm serve"
log "C2b B: $(grep '###' /workspace/c2b_teacher.log | sed 's/### //')"

# ── 4. HF ── (ảnh chụp HTML gói thành 1 file tar)
[ -d data/c2b/html_shots ] && tar -cf /workspace/c2b_html_shots.tar -C data/c2b html_shots
for f in data/c2/debug_v2.jsonl data/c2/teacher_log/debug_v2.jsonl data/c2b/verified_*.jsonl data/c2b/teacher_log/*.jsonl; do
  [ -s $f ] && hf upload donaldluong/nn-aivie-2p6b-p3 $f $f > /dev/null 2>&1
done
[ -s /workspace/c2b_html_shots.tar ] && hf upload donaldluong/nn-aivie-2p6b-p3 /workspace/c2b_html_shots.tar data/c2b/html_shots.tar > /dev/null 2>&1
log "C2b B XONG — đã đẩy HF nn-aivie-2p6b-p3 (data/c2b/verified_*, teacher_log/*, data/c2/debug_v2 + nhật ký)"
