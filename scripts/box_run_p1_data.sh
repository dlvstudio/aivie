#!/bin/bash
# TK_107 P1 · DỮ LIỆU train tiếp (CPT) cho thân 2,6B — box 1×H100, ổ 120–150GB (python hệ thống NGC + vLLM venv).
#   1) Gemma-31B chấm "giá trị giáo dục" 40K đoạn web Việt → train bộ chấm nhỏ (edu_classifier.py) → XOÁ Gemma (đỡ 32GB)
#   2) dựng shard theo nhóm, MỖI NGUỒN 1 TIẾN TRÌNH song song; mỗi shard ghi xong → ĐẨY lên HF dataset riêng tư → xoá trên ổ
#      (cần token WRITE: user tự `hf auth login` trên box; kịch bản CHỜ tới khi đăng nhập xong)
# Trạng thái → /workspace/cpt_status.log
cd /workspace/nn-aivie
S=/workspace/cpt_status.log
PY=/workspace/vllm_env/bin/python        # Python hệ thống NGC ghim datasets 2.2.1 (vỡ với fsspec mới) → dùng venv vLLM
export PATH=/workspace/vllm_env/bin:$PATH
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=2
log() { echo "$(date '+%F %T') $*" >> $S; }
HF_DATA=${HF_DATA:-donaldluong/nn-aivie-cpt-data}
NC=$(nproc --all)                       # KHÔNG dùng nproc trơn: nó tôn trọng OMP_NUM_THREADS=2 ở trên
until grep -q SETUP_XONG /workspace/setup_p1.log 2>/dev/null; do sleep 30; done
log "P1: bắt đầu · $NC nhân · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}') · $(grep vllm /workspace/setup_p1.log | tail -1)"

# 1. bộ chấm giáo dục
mkdir -p data/cpt
if [ ! -s data/cpt/edu_clf.pt ]; then
  [ -s data/cpt/edu_sample.jsonl ] || $PY scripts/edu_classifier.py sample --sources fineweb2,hplt --n 40000 --out data/cpt/edu_sample.jsonl > /workspace/cpt_sample.log 2>&1
  log "P1 lấy mẫu: $(tail -1 /workspace/cpt_sample.log)"
  curl -s localhost:8000/v1/models | grep -q gemma4-31b || setsid nohup /workspace/vllm_env/bin/vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic --served-model-name gemma4-31b \
    --max-model-len 6144 --gpu-memory-utilization 0.90 --max-num-seqs 128 --port 8000 > /workspace/vllm_p1.log 2>&1 < /dev/null &
  W=0; until curl -s localhost:8000/v1/models | grep -q gemma4-31b; do sleep 15; W=$((W+1)); [ $W -gt 80 ] && { log "LỖI P1: vLLM không lên (xem vllm_p1.log)"; exit 1; }; done
  $PY scripts/edu_classifier.py label --inp data/cpt/edu_sample.jsonl --out data/cpt/edu_labels.jsonl --workers 160 > /workspace/cpt_label.log 2>&1
  pkill -f "vllm serve"; sleep 10
  $PY scripts/edu_classifier.py train --inp data/cpt/edu_labels.jsonl --out data/cpt/edu_clf.pt --epochs 4 > /workspace/cpt_clf.log 2>&1
  log "P1 bộ chấm: $(tail -1 /workspace/cpt_label.log) · $(grep -E 'train\]|###' /workspace/cpt_clf.log | tail -2 | tr '\n' ' ')"
fi
[ -s data/cpt/edu_clf.pt ] || { log "LỖI P1: chưa có bộ chấm"; exit 1; }
rm -rf /root/.cache/huggingface/hub/models--RedHatAI--gemma-4-31B-it-FP8-dynamic      # đỡ ~32GB (tải lại được)
log "P1: đã xoá Gemma · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}')"

# 2. chờ user đăng nhập HF (token WRITE) rồi dựng shard + đẩy dần
until hf auth whoami >/dev/null 2>&1 || huggingface-cli whoami >/dev/null 2>&1; do
  [ -f /workspace/.p1_wait_logged ] || { log "P1: CHỜ user 'hf auth login' (token WRITE) trên box…"; touch /workspace/.p1_wait_logged; }
  sleep 60
done
log "P1: HF đã đăng nhập ($(hf auth whoami 2>/dev/null | head -1)) → dựng shard, đẩy lên $HF_DATA"
B="$PY -W ignore scripts/build_cpt_data.py --out data/cpt --edu-clf data/cpt/edu_clf.pt --edu-min 2.5 --long-min 4096 --hf-upload $HF_DATA"
W1=$(( NC * 3 / 10 )); W2=$(( NC * 2 / 10 )); W3=$(( NC / 10 )); [ $W3 -lt 2 ] && W3=2
$B --sources vi_fineweb2 --budget vi_web:6e9,long:1.2e9 --workers $W1 > /workspace/cpt_b_fw2.log 2>&1 &
$B --sources vi_hplt --budget vi_web:3e9,long:0.8e9 --workers $W2 > /workspace/cpt_b_hplt.log 2>&1 &
$B --sources vi_wiki,vi_wikisource --budget vi_wiki:0.8e9,long:0.4e9 --workers $W3 > /workspace/cpt_b_wiki.log 2>&1 &
$B --sources en_edu --budget en_edu:3e9 --workers $W3 > /workspace/cpt_b_en.log 2>&1 &
$B --sources math --budget math:1e9 --workers $W3 > /workspace/cpt_b_math.log 2>&1 &
$B --sources code --budget code:1.5e9 --workers $W3 > /workspace/cpt_b_code.log 2>&1 &
wait
hf upload $HF_DATA data/cpt/edu_clf.pt edu_clf.pt --repo-type dataset > /dev/null 2>&1
log "P1 shard: $(grep -h '###' /workspace/cpt_b_*.log | tr '\n' ' ')"
log "P1 đẩy HF: $(grep -h '\[up\] đã đẩy' /workspace/cpt_b_*.log | tr '\n' ' ') · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}')"
log "P1 XONG → box P2 tải: hf download $HF_DATA --repo-type dataset --local-dir data/cpt"
