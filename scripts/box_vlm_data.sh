#!/bin/bash
# TK_107 P3.4 chuẩn bị · DỰNG LẠI DỮ LIỆU ĐỌC ẢNH (công thức V4 + dữ liệu V5) trên box P2 (4×H100) — bản cũ mất cùng box TK_106.
#   A  (CPU/mạng — chạy được NGAY trong lúc pha B train, không đụng GPU):
#      cài vLLM venv + playwright + phông Việt · tải gói p3-data (nguồn chữ) · tải ảnh tự nhiên / biểu đồ / the_cauldron
#      · vẽ 45K ảnh màn hình (V1) + 40K ảnh màn hình đọc-trước (V5) + 40K ảnh chữ (V3) + 80K ảnh chữ nhiều số (V5)
#   B  (sau khi P2 XONG): thầy Gemma-4-31B trên GPU 1–3 (GPU 0 để đánh giá mốc 20 tỷ), 3 việc song song:
#        GPU1 nhìn 32K ảnh tự nhiên · GPU2 dịch 24K hỏi–đáp the_cauldron · GPU3 điền câu cần hiểu cho ảnh màn hình
#      → vẽ 40K trang báo (cần ảnh đã được thầy nhìn) → thầy điền câu cần hiểu cho báo + ảnh màn hình V5
#      → gộp: align / sft / sft_ocr (V4) / sft5 (V5) / test_* → ĐẨY HF (bài học: lần trước không sao lưu → mất)
# HF (công khai, user duyệt 2026-10-11): phần TA TỰ TẠO + chữ thầy viết. KHÔNG đăng lại ảnh gốc của bộ dữ liệu ngoài
# (nat/img, caul/img, chart/img) → box P3 tải lại từ nguồn (`gen_vlm_data.py fetch`, thứ tự luồng cố định → cùng tên file).
# Trang báo nhúng ảnh COCO/OpenViVQA → chỉ đẩy khi UP_NEWS=1.
# Dùng:  bash scripts/box_vlm_data.sh A      (ngay)        bash scripts/box_vlm_data.sh B      (tự chờ "P2 XONG")
#        CLEAN_OK=1 → được xoá ckpt A1/A2 cũ (~88GB) khi ổ không đủ cho Gemma (33GB)
cd /workspace/nn-aivie
S=/workspace/vlm_data_status.log
VE=/workspace/vllm_env; PY=$VE/bin/python
export PATH=$VE/bin:$PATH PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4
unset PIP_CONSTRAINT                              # NGC ghim torch hệ thống cho MỌI pip → venv vLLM cài không được
log() { echo "$(date '+%F %T') $*" >> $S; }
D=/workspace/vlmdata; mkdir -p $D data/vlm
HF_VLM=${HF_VLM:-donaldluong/nn-aivie-vlm-data}
GEMMA=RedHatAI/gemma-4-31B-it-FP8-dynamic
free_gb() { df -BG /workspace | tail -1 | awk '{print $4}' | tr -d G; }
step() { [ -f $D/$1.done ]; }                       # bước đã xong → bỏ qua khi chạy lại
done_() { touch $D/$1.done; log "$1 xong · $2 · ổ còn $(free_gb)G"; }

stage_A() {
  log "A: bắt đầu · ổ còn $(free_gb)G"
  if ! step install; then
    apt-get update -q > /workspace/vlmd_apt.log 2>&1
    apt-get install -y -q fonts-noto-core fonts-liberation fonts-dejavu-core zstd >> /workspace/vlmd_apt.log 2>&1
    [ -x $PY ] || python3 -m venv $VE
    $PY -m pip install -q -U pip > /workspace/vlmd_pip.log 2>&1
    $PY -m pip install -q vllm datasets playwright pillow pyarrow "huggingface_hub[cli]" >> /workspace/vlmd_pip.log 2>&1 \
      || { log "LỖI A: pip (xem vlmd_pip.log)"; exit 1; }
    $PY -m playwright install --with-deps chromium >> /workspace/vlmd_pip.log 2>&1 || { log "LỖI A: playwright"; exit 1; }
    $PY -c "import vllm, datasets, playwright; print('vllm', vllm.__version__)" >> /workspace/vlmd_pip.log 2>&1 || { log "LỖI A: import"; exit 1; }
    done_ install "$(tail -1 /workspace/vlmd_pip.log)"
  fi
  if ! step p3data; then                               # nguồn chữ cho ảnh màn hình / ảnh chữ (văn phòng, SGK, code…)
    hf download donaldluong/nn-aivie-p3-data p3_data.tar.zst --repo-type dataset --local-dir /workspace/dl > /workspace/vlmd_dl.log 2>&1 \
      && tar -I zstd -xf /workspace/dl/p3_data.tar.zst -C . && rm -f /workspace/dl/p3_data.tar.zst || { log "LỖI A: p3-data"; exit 1; }
    done_ p3data "$(ls data | tr '\n' ' ')"
  fi
  if ! step fetch; then                                # ảnh của bộ dữ liệu ngoài (song song 2 luồng tải)
    $PY scripts/gen_vlm_data.py fetch --only vichart,openvivqa,ktvic,coco > /workspace/vlmd_fetch_a.log 2>&1 & J1=$!
    $PY scripts/gen_vlm_data.py fetch --only cauldron > /workspace/vlmd_fetch_b.log 2>&1 & J2=$!
    wait $J1 $J2
    grep -q "### coco" /workspace/vlmd_fetch_a.log && grep -q "### cauldron" /workspace/vlmd_fetch_b.log || { log "LỖI A: tải ảnh"; exit 1; }
    done_ fetch "$(grep -h '###' /workspace/vlmd_fetch_*.log | tr '\n' ' ')"
  fi
  if ! step render; then                               # ≤9 trình duyệt cùng lúc · ảnh chữ (PIL) chạy song song
    ( $PY scripts/gen_ocr.py --n 40000 --out data/vlm/ocr --workers 24 --seed 0 > /workspace/vlmd_ocr.log 2>&1
      $PY scripts/gen_ocr.py --n 80000 --out data/vlm/ocr5 --workers 24 --seed 5 --num-frac 0.4 --photo-frac 0.3 > /workspace/vlmd_ocr5.log 2>&1 ) & JO=$!
    for k in 0 1 2 3 4 5 6 7 8; do
      $PY scripts/gen_screens.py render --n 5000 --out data/vlm/screens --split train_p$k --seed $((100+k)) > /workspace/vlmd_scr_$k.log 2>&1 &
    done
    wait $(jobs -p | grep -v $JO)
    cat data/vlm/screens/train_p*.jsonl > data/vlm/screens/train.jsonl && rm -f data/vlm/screens/train_p*.jsonl
    $PY scripts/gen_screens.py render --n 600 --out data/vlm/screens --split test --seed 999 > /workspace/vlmd_scr_test.log 2>&1
    for k in 0 1 2 3 4 5 6 7; do
      $PY scripts/gen_screens.py render --n 5000 --out data/vlm/screens5 --split train_p$k --seed $((700+k)) --read-first \
        --weights excel:4,chart:3,email:1,doc:1,code:1.5,chat:0.5 > /workspace/vlmd_scr5_$k.log 2>&1 &
    done
    wait
    cat data/vlm/screens5/train_p*.jsonl > data/vlm/screens5/train.jsonl && rm -f data/vlm/screens5/train_p*.jsonl
    [ -s data/vlm/screens/train.jsonl ] && [ -s data/vlm/screens5/train.jsonl ] && [ -s data/vlm/ocr5/train.jsonl ] || { log "LỖI A: vẽ ảnh"; exit 1; }
    done_ render "màn hình $(wc -l < data/vlm/screens/train.jsonl)+$(wc -l < data/vlm/screens/test.jsonl) · đọc-trước $(wc -l < data/vlm/screens5/train.jsonl) · chữ $(wc -l < data/vlm/ocr/train.jsonl)+$(wc -l < data/vlm/ocr5/train.jsonl) · $(du -sh data/vlm | cut -f1)"
  fi
  [ -s data/vlm/news/docs.jsonl ] || $PY scripts/gen_news.py --dump --out data/vlm/news --fineweb "" --docs 60000 --seed 1 > /workspace/vlmd_news_dump.log 2>&1
  log "A XONG · chờ P2 XONG để chạy B"
}

serve() {  # $1 GPU · $2 cổng (đã chạy thì thôi — chạy lại B không bật thầy thứ hai)
  up_ $2 && return
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $VE/bin/vllm serve $GEMMA --served-model-name gemma4-31b --max-model-len 6144 \
    --gpu-memory-utilization 0.90 --max-num-seqs 128 --port $2 > /workspace/vlmd_vllm_$2.log 2>&1 < /dev/null &
}
up_() { curl -s localhost:$1/v1/models | grep -q gemma4-31b; }

stage_B() {
  until grep -q "P2 XONG" /workspace/cpt_status.log; do
    grep -q "LỖI P2" <(tail -n 3 /workspace/cpt_status.log) && { log "B: P2 báo LỖI → không chạy"; exit 1; }
    sleep 120
  done
  for s in install p3data fetch render; do step $s || { log "LỖI B: bước A '$s' chưa xong"; exit 1; }; done
  if [ $(free_gb) -lt 50 ]; then                     # Gemma 33GB + trang báo + chỗ thở
    if [ "$CLEAN_OK" = 1 ] && [ -f ckpt/cpt_b.done ] && [ -f ckpt/cpt_b_stable.pt ]; then
      rm -f ckpt/cpt_a1.pt ckpt/cpt_a1_stable.pt ckpt/cpt_a2.pt ckpt/cpt_a2_stable.pt ckpt/hybrid_2p6b_x.pt
      log "B: xoá ckpt A1/A2/chèn khối cũ (CLEAN_OK=1) → ổ còn $(free_gb)G"
    else log "LỖI B: ổ còn $(free_gb)G < 50G — chạy lại với CLEAN_OK=1 (xoá ckpt A1/A2 cũ)"; exit 1; fi
  fi
  hf download $GEMMA > /workspace/vlmd_gemma_dl.log 2>&1 || { log "LỖI B: tải Gemma"; exit 1; }
  serve 1 8001; serve 2 8002; serve 3 8003
  W=0; until up_ 8001 && up_ 8002 && up_ 8003; do sleep 15; W=$((W+1)); [ $W -gt 100 ] && { log "LỖI B: vLLM không lên (vlmd_vllm_*.log)"; exit 1; }; done
  log "B: 3 thầy Gemma sẵn sàng (GPU 1–3)"
  # thử nhanh thầy nhìn ảnh trước khi chạy hàng loạt (bỏ qua khi đã có kết quả nhìn ảnh từ lần chạy trước)
  if ! step vision && [ ! -s data/vlm/nat/vision.jsonl ]; then
    $PY scripts/gen_vlm_data.py vision --llm http://127.0.0.1:8001/v1 --limit 24 --workers 24 > /workspace/vlmd_vision_smoke.log 2>&1
    N=$(grep -o 'vision: [0-9]*' /workspace/vlmd_vision_smoke.log | grep -o '[0-9]*$'); [ "${N:-0}" -ge 12 ] || { log "LỖI B: thầy nhìn ảnh thử chỉ $N/24"; exit 1; }
  fi
  # nhìn ảnh (việc dài nhất) CHIA ĐỀU cho cả 3 thầy theo crc32(id) — mỗi phần ghi file riêng rồi gộp
  ( step vision || {
      for i in 0 1 2; do
        $PY scripts/gen_vlm_data.py vision --llm http://127.0.0.1:800$((i+1))/v1 --workers 64 --shard $i/3 > /workspace/vlmd_vision_s$i.log 2>&1 &
      done
      wait
      $PY -c "
import sys; sys.path.insert(0, 'scripts'); import glob, json, gen_vlm_data as G
p = 'data/vlm/nat/vision.jsonl'; seen = set(); rows = []
for f in [p] + sorted(glob.glob('data/vlm/nat/vision_s*.jsonl')):
    for r in G.rows_of(f):
        if r['id'] not in seen: seen.add(r['id']); rows.append(r)
with open(p + '.tmp', 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\\n')
print(len(rows))" > /workspace/vlmd_vision_merge.log 2>&1 && mv data/vlm/nat/vision.jsonl.tmp data/vlm/nat/vision.jsonl \
        && rm -f data/vlm/nat/vision_s*.jsonl && done_ vision "$(cat /workspace/vlmd_vision_merge.log) ảnh có lời thầy"; } ) & JV=$!
  # dịch / điền câu: lần chạy trước còn đang chạy (tiến trình mồ côi) → CHỜ dấu xong, không chạy trùng
  waitfor() { local miss=0; while ! step $1 && [ $miss -lt 2 ]; do pgrep -f "$2" > /dev/null && miss=0 || miss=$((miss+1)); sleep 30; done; }
  ( waitfor translate "gen_vlm_data.py translate"; step translate || { $PY scripts/gen_vlm_data.py translate --llm http://127.0.0.1:8002/v1 --workers 96 > /workspace/vlmd_trans.log 2>&1 \
      && done_ translate "$(grep '###' /workspace/vlmd_trans.log)"; } ) & JT=$!
  ( waitfor teach1 "gen_screens.py teach --out data/vlm/screens"; step teach1 || { for sp in train test; do $PY scripts/gen_screens.py teach --out data/vlm/screens --split $sp --llm http://127.0.0.1:8003/v1 --workers 96; done > /workspace/vlmd_teach1.log 2>&1
      $PY scripts/gen_screens.py teach --out data/vlm/screens5 --split train --llm http://127.0.0.1:8003/v1 --workers 96 >> /workspace/vlmd_teach1.log 2>&1
      done_ teach1 "$(grep '###' /workspace/vlmd_teach1.log | tr '\n' ' ')"; } ) & JS=$!
  wait $JV                                             # trang báo cần ảnh đã được thầy nhìn (nat/vision.jsonl)
  step vision || { log "LỖI B: thầy nhìn ảnh (vlmd_vision.log)"; exit 1; }
  if ! step news; then
    for k in 0 1 2 3 4 5 6 7; do
      $PY scripts/gen_news.py --n 5000 --out data/vlm/news --split train_p$k --seed $((500+k)) --fineweb data/vlm/news/docs.jsonl \
        --docs 100000 > /workspace/vlmd_news_$k.log 2>&1 &
    done
    wait $(jobs -p | grep -v -E "^($JT|$JS)$")
    cat data/vlm/news/train_p*.jsonl > data/vlm/news/train.jsonl && rm -f data/vlm/news/train_p*.jsonl
    $PY scripts/gen_news.py --n 600 --out data/vlm/news --split test --seed 999 --fineweb data/vlm/news/docs_test.jsonl --docs 100000 \
      > /workspace/vlmd_news_test.log 2>&1
    done_ news "$(wc -l < data/vlm/news/train.jsonl) train · $(wc -l < data/vlm/news/test.jsonl) test"
  fi
  if ! step teach2; then                               # GPU1 rảnh sau khi nhìn ảnh xong → điền câu cho báo
    for sp in train test; do $PY scripts/gen_screens.py teach --out data/vlm/news --split $sp --llm http://127.0.0.1:8001/v1 --workers 96; done \
      > /workspace/vlmd_teach2.log 2>&1
    done_ teach2 "$(grep '###' /workspace/vlmd_teach2.log | tr '\n' ' ')"
  fi
  wait $JT $JS
  for s in translate teach1; do step $s || { log "LỖI B: '$s' (xem log vlmd_*)"; exit 1; }; done
  pkill -f "vllm serve"; sleep 10
  # gộp: V1 (align/sft/test_*) → V4 (sft + ảnh chữ) → V5 (báo + ảnh chữ nhiều số + đọc-trước + ôn 60K V4)
  $PY scripts/gen_vlm_data.py build > /workspace/vlmd_build.log 2>&1 || { log "LỖI B: build"; exit 1; }
  $PY - > /workspace/vlmd_mix.log 2>&1 <<'EOF'
import json, random
def rows(p, pre, n=None, seed=5):
    xs = [json.loads(l) for l in open(p)]
    xs = [x for x in xs if all(m['content'] for m in x['messages'])]
    for x in xs: x['images'] = [pre + q for q in x['images']]
    random.Random(seed).shuffle(xs); return xs[:n] if n else xs
def dump(p, xs):
    with open(p, 'w') as f:
        for x in xs: f.write(json.dumps(x, ensure_ascii=False) + '\n')
v4 = [json.loads(l) for l in open('data/vlm/sft.jsonl')] + rows('data/vlm/ocr/train.jsonl', 'ocr/')
random.Random(4).shuffle(v4); dump('data/vlm/sft_ocr.jsonl', v4)
v5 = rows('data/vlm/news/train.jsonl', 'news/') + rows('data/vlm/ocr5/train.jsonl', 'ocr5/') + \
     rows('data/vlm/screens5/train.jsonl', 'screens5/') + rows('data/vlm/sft_ocr.jsonl', '', 60000)
random.Random(5).shuffle(v5); dump('data/vlm/sft5.jsonl', v5)
dump('data/vlm/test_news.jsonl', rows('data/vlm/news/test.jsonl', 'news/'))
print(f"sft_ocr {len(v4)} · sft5 {len(v5)}")
EOF
  log "B gộp: $(grep -E '###|nguồn' /workspace/vlmd_build.log | tr '\n' ' ' | cut -c1-400) · $(cat /workspace/vlmd_mix.log)"
  # đẩy HF: mỗi phần 1 file tar (ảnh đã nén sẵn → tar thường) + 1 file chữ (jsonl) nén zstd
  mkdir -p /workspace/vlm_up
  tar -I 'zstd -10 -T0' -cf /workspace/vlm_up/meta.tar.zst --exclude='*/img/*' --exclude='*.jpg' --exclude='*.png' data/vlm
  for p in screens screens5 ocr ocr5; do tar -cf /workspace/vlm_up/$p.tar data/vlm/$p; done
  [ "$UP_NEWS" = 1 ] && tar -cf /workspace/vlm_up/news.tar data/vlm/news
  hf repo create $HF_VLM --repo-type dataset > /dev/null 2>&1
  hf upload-large-folder $HF_VLM /workspace/vlm_up --repo-type dataset > /workspace/vlmd_up.log 2>&1 \
    && log "B: đã đẩy HF $HF_VLM ($(du -sh /workspace/vlm_up | cut -f1))" || log "LỖI B: đẩy HF (vlmd_up.log) — dữ liệu vẫn trên ổ"
  log "VLM DATA XONG · $(du -sh data/vlm | cut -f1)"
}

case "$1" in
  A) stage_A ;;
  B) stage_B ;;
  *) echo "dùng: $0 A|B"; exit 1 ;;
esac
