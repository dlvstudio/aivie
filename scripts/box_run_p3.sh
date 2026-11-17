#!/bin/bash
# TK_107 P3 · biến thân nền 2,5B (cpt_b.pt, mốc 20 tỷ) thành aivie biết chat — 1×H100 80GB, ~1,5–2 ngày.
#   P3.0  thử nhanh code trên thân mới (smoke_p3.py) — sai là DỪNG, không đốt giờ GPU
#   P3.1  SFT toàn bộ tham số: đủ mọi đợt SFT của 1.3B (125K mẫu, ~85M token, 5% mẫu dài 8K) × 2 epoch → ckpt/p3_sft.pt
#   P3.2  3 bộ expert MỚI (code / văn phòng / học thuật), thân đóng băng, 12 điểm cắm × nửa cỡ → ckpt/expert_code/p3_*.pt
#   P3.5  bộ chấm: eval_v1 (giám khảo) + easy20 + văn phòng + SGK/toán + ngữ cảnh dài 8K — so với 1.3B v5
#   (P3.3 vòng thầy và P3.4 đọc ảnh chạy SAU khi P3.5 đạt — xem box_run_v5.sh / box_run_vlm_v4.sh, đổi --base)
# Mỗi bước có dấu ckpt/<bước>.done → chạy lại bỏ qua bước đã xong. Kết quả + checkpoint đẩy HF sau MỖI bước.
# Trạng thái → /workspace/p3_status.log.  Dùng: bash scripts/box_run_p3.sh   (sau `hf auth login` token WRITE)
cd /workspace/nn-aivie
S=/workspace/p3_status.log
PY=${PY:-$( [ -x /workspace/zenv/bin/python ] && echo /workspace/zenv/bin/python || echo /usr/bin/python )}   # box NGC: python hệ thống
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PIP_CONSTRAINT                                  # NGC ghim torch cho mọi pip → cài bitsandbytes/ollama-client không được
log() { echo "$(date '+%F %T') $*" >> $S; }
HF_CKPT=${HF_CKPT:-donaldluong/nn-aivie-hybrid-2p6b-ckpt-b}      # nơi P2 sao lưu cpt_b.pt
HF_P3=${HF_P3:-donaldluong/nn-aivie-2p6b-p3}                      # nơi P3 sao lưu (model riêng tư)
HF_DATA=${HF_DATA:-donaldluong/nn-aivie-p3-data}                  # gói dữ liệu P3 từ Mac (pack_p3_data.sh)
up() { hf upload $HF_P3 "$1" "$2" > /dev/null 2>&1 && log "HF ↑ $2" || log "LỖI sao lưu HF $2 (chạy tiếp)"; }
mkdir -p ckpt/expert_code out

until hf auth whoami >/dev/null 2>&1; do
  [ -f /workspace/.p3_wait ] || { log "P3: CHỜ user 'hf auth login' (token WRITE) trên box…"; touch /workspace/.p3_wait; }
  sleep 60
done
hf repo create $HF_P3 --private > /dev/null 2>&1

# ── tải thân nền + dữ liệu ─────────────────────────────────────────────────────────────────────
[ -f ckpt/cpt_b.pt ] || hf download $HF_CKPT cpt_b.pt --local-dir ckpt > /workspace/p3_dl.log 2>&1
[ -f ckpt/cpt_b.pt ] || { log "LỖI P3: không tải được ckpt/cpt_b.pt từ $HF_CKPT"; exit 1; }
if [ ! -f data/sft_p3/sft_p3.jsonl ]; then
  hf download $HF_DATA p3_data.tar.zst --repo-type dataset --local-dir /workspace/dl >> /workspace/p3_dl.log 2>&1 \
    && tar -I zstd -xf /workspace/dl/p3_data.tar.zst -C . && rm -f /workspace/dl/p3_data.tar.zst
fi
[ -f data/sft_p3/sft_p3.jsonl ] || { log "LỖI P3: thiếu dữ liệu (data/sft_p3) — chạy scripts/pack_p3_data.sh trên Mac"; exit 1; }
[ -f data/code_sft/train.jsonl ] || $PY -u scripts/expert_ffn_1p3b.py prep > /workspace/p3_prep.log 2>&1   # Magicoder + chat
log "P3: thân $(du -h ckpt/cpt_b.pt | cut -f1) · dữ liệu SFT $(wc -l < data/sft_p3/sft_p3.jsonl) mẫu · ổ còn $(df -h /workspace | tail -1 | awk '{print $4}')"

# ── P3.0 thử nhanh ────────────────────────────────────────────────────────────────────────────
if [ ! -f ckpt/p3_smoke.done ]; then
  $PY -W ignore -u scripts/smoke_p3.py --base ckpt/cpt_b.pt --device cuda > /workspace/p3_smoke.log 2>&1 \
    || { log "LỖI P3.0 smoke: $(tail -3 /workspace/p3_smoke.log | tr '\n' ' ')"; exit 1; }
  log "P3.0 smoke OK: $(tail -1 /workspace/p3_smoke.log)"; touch ckpt/p3_smoke.done
fi

# ── P3.1 SFT toàn bộ tham số (AdamW fp32 ~54GB trạng thái; OOM → tự chuyển AdamW 8-bit) ──────────
if [ ! -f ckpt/p3_sft.done ]; then
  # box chết giữa chừng → lấy bản sao lưu mới nhất trên HF (con trỏ latest.json → slot) rồi CHẠY TIẾP
  [ -f ckpt/p3_sft.pt ] || $PY - >> /workspace/p3_dl.log 2>&1 <<EOF
import json
from huggingface_hub import hf_hub_download
try:
    m = json.load(open(hf_hub_download("$HF_P3", "latest.json")))
    hf_hub_download(m.get("slot_repo", "$HF_P3"), "p3_sft.pt", local_dir="ckpt")
    print("SFT chạy tiếp từ HF step", m.get("step"))
except Exception as e:
    print("chưa có bản SFT trên HF:", type(e).__name__)
EOF
  [ -f ckpt/p3_sft.pt ] && log "P3.1: chạy tiếp SFT từ bản sao lưu ($(grep 'chạy tiếp' /workspace/p3_dl.log | tail -1))"
  sft() {  # $1 = adamw | adamw8bit
    $PY -u scripts/train_hybrid_sft.py --resume --init-from ckpt/cpt_b.pt --data data/sft_p3/sft_p3.jsonl \
      --max-len 8192 --tok-budget 16384 --bs 8 --grad-accum 4 --grad-ckpt --optim $1 \
      --lr 1e-5 --min-lr 1e-6 --warmup 50 --epochs 2 --eval-every 200 --save-every 500 \
      --out ckpt/p3_sft.pt --hf-repo $HF_P3 --hf-every 500 --hf-name p3_sft.pt > /workspace/p3_sft_$1.log 2>&1
  }
  sft adamw || {
    grep -q "OutOfMemory\|out of memory" /workspace/p3_sft_adamw.log || { log "LỖI P3.1 SFT: $(tail -2 /workspace/p3_sft_adamw.log | tr '\n' ' ')"; exit 1; }
    log "P3.1: AdamW fp32 hết bộ nhớ → AdamW 8-bit"
    $PY -c "import bitsandbytes" 2>/dev/null || $PY -m pip install -q bitsandbytes >> /workspace/p3_sft_adamw8bit.log 2>&1
    sft adamw8bit || { log "LỖI P3.1 SFT 8-bit: $(tail -2 /workspace/p3_sft_adamw8bit.log | tr '\n' ' ')"; exit 1; }
  }
  log "P3.1 SFT xong: $(grep -h -E 'TRƯỚC SFT|XONG SFT' /workspace/p3_sft_*.log | tr '\n' ' ')"
  touch ckpt/p3_sft.done
fi

# ── P3.2 3 bộ expert mới trên thân SFT (đóng băng) ─────────────────────────────────────────────
X="$PY -W ignore -u scripts/expert_office_1p3b.py"
B="--base ckpt/p3_sft.pt"
EX="--code ckpt/expert_code/p3_code.pt --office ckpt/expert_code/p3_vp.pt --hoc ckpt/expert_code/p3_hoc.pt"
if [ ! -f ckpt/p3_expert.done ]; then
  $X trainGEN $B --code "" --office "" --hoc "" --train code,vp,hoc \
     --mix code:0.15,tb:0.15,vi_code:0.05,off:0.15,sgk:0.15,math:0.15,mt:0.15,calib:0.05 \
     --lr 3e-4 --steps 3000 --bs 4 --accum 4 --tag p3 > /workspace/p3_expert.log 2>&1 \
     || { log "LỖI P3.2 expert: $(tail -2 /workspace/p3_expert.log | tr '\n' ' ')"; exit 1; }
  for n in code vp hoc; do up ckpt/expert_code/p3_$n.pt expert_code/p3_$n.pt; done
  log "P3.2 expert xong: $(grep -E '^\[gen\] (sau|cổng)' /workspace/p3_expert.log | tr '\n' ' ' | cut -c1-600)"
  touch ckpt/p3_expert.done
fi

# ── P3.5 bộ chấm (giám khảo Ollama như v5) ─────────────────────────────────────────────────────
if [ ! -f ckpt/p3_eval.done ]; then
  command -v ollama > /dev/null || curl -fsSL https://ollama.com/install.sh | sh >> /workspace/p3_ollama.log 2>&1
  curl -s localhost:11434/api/tags > /dev/null || { OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_KEEP_ALIVE=24h \
    setsid nohup ollama serve >> /workspace/ollama_serve.log 2>&1 < /dev/null & sleep 10; }
  O=http://127.0.0.1:11434
  curl -s $O/api/tags | grep -q "gemma4:26b-a4b-it-qat" || ollama pull gemma4:26b-a4b-it-qat >> /workspace/p3_ollama.log 2>&1 \
    || { log "LỖI P3.5: không tải được giám khảo gemma4:26b-a4b-it-qat"; exit 1; }
  V="$B $EX --llm $O"
  $X evalv1 $V --tag p3 > /workspace/p3_evalv1.log 2>&1 & J1=$!                        # chỉ chờ ĐÚNG các PID này
  sleep 45; ( $X easy20 $V --tag p3 > /workspace/p3_easy20.log 2>&1; $X evaloff $V --ollama $O --tag office_p3 > /workspace/p3_evaloff.log 2>&1 ) & J2=$!
  sleep 45; ( $X evalsgk $V --n 400 --tag p3 > /workspace/p3_evalsgk.log 2>&1; $X evalmath $V --tag p3 > /workspace/p3_evalmath.log 2>&1 ) & J3=$!
  wait $J1 $J2 $J3
  $PY -u scripts/eval_longctx.py --ckpt ckpt/p3_sft.pt --texts data/sft_p3/longctx_texts.jsonl > /workspace/p3_longctx.log 2>&1
  $PY -u scripts/eval_clean.py judge out/evalclean_p3.jsonl > /workspace/p3_judge.log 2>&1
  $PY -u scripts/eval_clean.py report v5 p3 > /workspace/p3_report.txt 2>&1
  tar -czf /workspace/p3_results.tgz out/*p3* /workspace/p3_*.log /workspace/p3_report.txt 2>/dev/null
  up /workspace/p3_results.tgz results/p3_results.tgz
  log "P3.5 chấm xong: $(tail -5 /workspace/p3_report.txt | tr '\n' ' ' | cut -c1-500) · ngữ cảnh dài: $(grep -h '###' /workspace/p3_longctx.log | cut -c1-300 | tr '\n' ' ')"
  touch ckpt/p3_eval.done
fi
log "P3 XONG (SFT + expert + chấm) — chờ user quyết P3.3 vòng thầy / P3.4 đọc ảnh"
