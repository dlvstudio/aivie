#!/usr/bin/env bash
# Train tuần tự curriculum lớp 3–7: teach → G1 → G2 → eval (G3 tuỳ chọn)
# Dùng khi cần train lại toàn bộ hoặc từ một lớp cụ thể.
#
#   ./scripts/train_curriculum.sh              # lớp 3→7
#   ./scripts/train_curriculum.sh --from 4     # chỉ lớp 4→7
#   ./scripts/train_curriculum.sh --grade 5    # chỉ lớp 5
#   ./scripts/train_curriculum.sh --g3         # bật G3 dopamine (chậm)

set -euo pipefail
cd "$(dirname "$0")/.."

FROM=3
TO=7
ONLY=""
G3_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --to)   TO="$2"; shift 2 ;;
    --grade) ONLY="$2"; shift 2 ;;
    --g3)   G3_FLAG="--g3"; shift ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$ONLY" ]]; then
  FROM="$ONLY"
  TO="$ONLY"
fi

# Chờ train lớp 3 đang chạy (nếu --from >= 4)
if [[ "$FROM" -ge 4 ]] && pgrep -f "run_train_eval.py --grade 3" >/dev/null 2>&1; then
  echo "==> Đang chờ train lớp 3 hoàn thành..."
  while pgrep -f "run_train_eval.py --grade 3" >/dev/null 2>&1; do sleep 30; done
  echo "==> Lớp 3 xong."
fi

for grade in $(seq "$FROM" "$TO"); do
  out="checkpoints/grade_$(printf '%02d' "$grade").pkl"
  log="/tmp/nn_train_grade$(printf '%02d' "$grade").log"
  pack="lessons/grade_$(printf '%02d' "$grade")_pack.yaml"

  if [[ ! -f "$pack" ]]; then
    echo "SKIP lớp $grade: chưa có $pack" >&2
    continue
  fi

  echo ""
  echo "============================================================"
  echo " TRAIN LỚP $grade — $(date)"
  echo " checkpoint: $out"
  echo " log: $log"
  echo "============================================================"

  rm -f "$out"
  PYTHONUNBUFFERED=1 python scripts/run_train_eval.py \
    --grade "$grade" \
    --full \
    $G3_FLAG \
    --out "$out" \
    2>&1 | tee "$log"

  echo "==> Lớp $grade xong: $out"
done

echo ""
echo "Done. Checkpoints: checkpoints/grade_*.pkl"
