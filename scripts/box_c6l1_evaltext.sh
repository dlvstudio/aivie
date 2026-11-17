#!/bin/bash
# TK_108 05/11: đo lại chữ c6l1 — bước evaltext của box_run_c6_loop.sh thoát sau 40s (ui() trả mã 1 vì until) nên không đo gì.
# Dùng 3 máy chủ c6_loop1.pt đang chạy ở GPU 1 (8891-8893); tắt 3 cái ở GPU 0 để chừa chỗ vLLM + vòng thầy chấm lại.
cd /workspace/nn-aivie
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8
EV=/workspace/evalenv/bin/python; CE=scripts/code_eval.py; TAG=c6l1; NEW=ckpt/c6_loop1.pt
for P in 8894 8895 8896; do for p in $(pgrep -f "serve_hybrid_ui.py --ckpt $NEW --port $P"); do kill $p; done; done
for P in 8891 8892 8893; do curl -s localhost:$P > /dev/null || { log "LỖI đo lại c6l1: máy chủ $P không sống"; exit 1; }; done
run() {  # $1 cổng $2… việc
  local U=http://localhost:$1; shift
  for job in "$@"; do
    case $job in
      agent)     $EV -u $CE run --suite agent --url $U --tag $TAG > /workspace/${TAG}_agent.log 2>&1 ;;
      humaneval) $EV -u $CE run --suite humaneval --url $U --tag $TAG > /workspace/${TAG}_humaneval.log 2>&1 ;;
      mbpp)      $EV -u $CE run --suite mbpp --url $U --tag $TAG > /workspace/${TAG}_mbpp.log 2>&1 ;;
      c7_*)      $EV -u scripts/c7_real.py eval --url $U --tag $TAG --shard ${job#c7_}/3 --workers 1 > /workspace/${TAG}_c7_${job#c7_}.log 2>&1 ;;
      office)    $EV -u scripts/office_probe_light.py $U $TAG > /workspace/${TAG}_office.log 2>&1 ;;
    esac
  done
}
log "c6l1 đo lại chữ bắt đầu: 3 máy chủ GPU 1 · tool+C7/1 · HE+C7/2 · MBPP+văn phòng+C7/3"
run 8891 agent c7_1 & J1=$!; run 8892 humaneval c7_2 & J2=$!; run 8893 mbpp office c7_3 & J3=$!
wait $J1 $J2 $J3
log "ĐO c6l1 lại · C7 $($EV scripts/c7_real.py report --tag $TAG 2>/dev/null | cut -c1-200) · HE $(grep '###' /workspace/${TAG}_humaneval.log | tail -1 | cut -c1-90) · MBPP $(grep '###' /workspace/${TAG}_mbpp.log | tail -1 | cut -c1-90) · tool $(grep '###' /workspace/${TAG}_agent.log | tail -1 | cut -c1-140) · văn phòng $(grep '###' /workspace/${TAG}_office.log | cut -c1-120) · lỗi máy chủ $(cat /workspace/${TAG}_ui_889[1-3].log | grep -c '\[lỗi /api/chat\]')"
for P in 8891 8892 8893; do for p in $(pgrep -f "serve_hybrid_ui.py --ckpt $NEW --port $P"); do kill $p; done; done
