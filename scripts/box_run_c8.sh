#!/bin/bash
# TK_108 C8 (user đồng ý 22/10): thử LoRA KHỐI TRỘN của thân — giả thuyết "trần ở thân đóng băng" (C7: bộ đo thực tế cx3b 0,6% →
# cx3c ~3%, Python/JS nghiệp vụ vẫn 0% dù loss C7 giảm một nửa; expert FFN không chạm tới phần đọc/nối đề của Mamba + attention).
# So sánh SẠCH với cx3c: cùng gốc cx3b, cùng trộn, cùng 700 bước, cùng LR expert — CHỈ thêm CodeLoRA (wq/wk/wv/wo + in_proj/out_proj
# Mamba, từ khối 3) học LR 2e-4. Hai biến thể song song, mỗi GPU một bản: cx3d r=32 · cx3e r=128.
# Tự chờ " C7 XONG — " → SFT → đẩy HF expert + LoRA → gói (KHÔNG đẩy snap: hạn mức HF) → đo bộ C7 355 đề (mỗi bản 2 máy chủ chia đôi)
# → tool-use · MBPP · Excel/easy20 → so với cx3c. Dùng: (setsid nohup bash scripts/box_run_c8.sh > /workspace/c8_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AIVIE_SANDBOX_USER=sbx
P3=donaldluong/nn-aivie-2p6b-p3
EV=/workspace/evalenv/bin/python
up() { hf upload $P3 "$1" "$2" > /dev/null 2>&1; }
until grep -q " C7 XONG — " $S; do sleep 60; done
rm -f out/code_eval/c7_cx3d_* out/code_eval/c7_cx3e_*        # bỏ kết quả đo hỏng (máy chủ chưa lên) của lần trước
[ -s data/c7/train.jsonl ] && [ -s ckpt/expert_code/cx3b_code.pt ] || { log "LỖI C8: thiếu data/c7/train.jsonl hoặc cx3b_code.pt"; exit 1; }
MIX="c7:0.35,c3r:0.12,c3r2:0.08,c3:0.10,c2bV:0.10,c2bF:0.05,c2bA:0.04,c2oci:0.03,mt:0.06,offfix:0.04,fixfix:0.03"
log "C8 bắt đầu: LoRA khối trộn + expert code từ cx3b · 700 bước · trộn như cx3c · GPU 0 cx3d r32 · GPU 1 cx3e r128"
sft() {   # $1 tag  $2 GPU  $3 r
  CUDA_VISIBLE_DEVICES=$2 python -W ignore -u scripts/expert_office_1p3b.py trainGEN --base ckpt/p3_sft.pt --code ckpt/expert_code/cx3b_code.pt \
     --office ckpt/expert_code/p3r1xl_vp.pt --hoc ckpt/expert_code/p3r1_hoc.pt --train code --mix "$MIX" \
     --lr-old 5e-5 --lora-r $3 --lora-lr 2e-4 --steps 700 --bs 1 --accum 16 --tag $1 > /workspace/$1_sft.log 2>&1 \
     || { log "LỖI C8: SFT $1 ($1_sft.log)"; return 1; }
  log "$1 SFT xong (LoRA r$3) · $(grep '\[gen\] sau' /workspace/$1_sft.log | cut -c1-230)"
  up ckpt/expert_code/$1_code.pt expert_code/$1_code.pt && up ckpt/expert_code/$1_lora.pt expert_code/$1_lora.pt \
     && log "$1 đã đẩy HF expert_code/$1_code.pt + $1_lora.pt"
  python - $1 $3 <<'PY' > /workspace/$1_pack.log 2>&1 || { log "LỖI C8: gói $1"; return 1; }
import sys, torch
tag, r = sys.argv[1], int(sys.argv[2])
o = torch.load("ckpt/aivie_p3_r1xl_bf16.pt", map_location="cpu", weights_only=False)
sd = torch.load(f"ckpt/expert_code/{tag}_code.pt", map_location="cpu")
pos = sorted({int(k.split(".")[0]) for k in sd})
o["experts"]["code"] = {"state": {k: v.to(torch.bfloat16) for k, v in sd.items()}, "gated": True,
                        "hid": sd[f"{pos[0]}.w3.weight"].shape[1], "pos": pos}
o["code_lora"] = {"r": r, "state": torch.load(f"ckpt/expert_code/{tag}_lora.pt", map_location="cpu")}
o.setdefault("merged_from", []).append({"code": f"{tag}_code + CodeLoRA r{r} (TK_108 C8: cx3b + C7 700 bước + LoRA khối trộn)"})
torch.save(o, f"ckpt/aivie_p3_{tag}_bf16.pt"); print("ok", len(pos), len(o["code_lora"]["state"]))
PY
}
if [ -s ckpt/aivie_p3_cx3d_bf16.pt ] && [ -s ckpt/aivie_p3_cx3e_bf16.pt ]; then
  log "C8: đã có gói cx3d + cx3e → bỏ qua SFT, chỉ đo lại"
else
  sft cx3d 0 32 & J1=$!
  sft cx3e 1 128 & J2=$!
  wait $J1; wait $J2
fi
[ -s ckpt/aivie_p3_cx3d_bf16.pt ] || [ -s ckpt/aivie_p3_cx3e_bf16.pt ] || { log "LỖI C8: không bản nào gói được"; exit 1; }
# ── đo ──
export PYTHONPATH=/workspace/nn-aivie/scripts:/workspace/nn-aivie/src:/workspace/nn-aivie
CE=scripts/code_eval.py; PIDS=""; TAGS=""
for tg in cx3d:0 cx3e:1; do
  IFS=: read t g <<< "$tg"
  [ -s ckpt/aivie_p3_${t}_bf16.pt ] || continue
  TAGS="$TAGS $t"
  for k in 1 2; do
    p=$((8880 + 2 * g + k))
    CUDA_VISIBLE_DEVICES=$g python -W ignore -u scripts/serve_hybrid_ui.py --ckpt ckpt/aivie_p3_${t}_bf16.pt --port $p --no-rag --max-ctx 4096 > /workspace/c8_ui_$p.log 2>&1 &
    PIDS="$PIDS $!"
  done
done
for t in $TAGS; do                                             # 22/10: bản cũ bỏ qua bước chờ khi log còn rỗng → đo vào máy chủ chưa lên = 0%
 g=$([ $t = cx3d ] && echo 0 || echo 1)
 for p in $((8881 + 2 * g)) $((8882 + 2 * g)); do
  W=0; until curl -s localhost:$p > /dev/null; do sleep 10; W=$((W+1)); [ $W -gt 150 ] && { log "LỖI C8: UI $p"; kill $PIDS; exit 1; }; done
 done
done
log "C8: $(grep -h '\[lora\]' /workspace/c8_ui_*.log | head -1)"
c7e() { $EV -u scripts/c7_real.py eval --url http://localhost:$1 --tag $2 --shard $3 --workers 1 > /workspace/c7_eval_$2_${3/\//of}.log 2>&1 \
        || log "LỖI C8 đo C7 $2 phần $3"; }
JOBS=""
for t in $TAGS; do
  g=$([ $t = cx3d ] && echo 0 || echo 1); p1=$((8881 + 2 * g)); p2=$((p1 + 1))
  ( c7e $p1 $t 1/2
    $EV -u $CE run --suite agent --url http://localhost:$p1 --tag $t > /workspace/${t}_agent.log 2>&1 \
      && log "$t agent (cx3c 50 · cx3b 55) · $(grep '###' /workspace/${t}_agent.log | tail -1 | cut -c1-200)" || log "LỖI C8 $t agent"
    $EV -u scripts/office_probe_light.py http://localhost:$p1 $t > /workspace/${t}_office.log 2>&1 \
      && log "$t văn phòng (cx3b Excel 15 · easy20 16) · $(grep '###' /workspace/${t}_office.log)" || log "LỖI C8 $t văn phòng" ) &
  JOBS="$JOBS $!"
  ( c7e $p2 $t 2/2
    until grep -q "^### c7 ${t}_s1" /workspace/c7_eval_${t}_1of2.log 2>/dev/null; do sleep 30; done
    log "C7 đo $t (cx3b 0,6 · cx3c xem dòng 'C7 đo cx3c') · $($EV scripts/c7_real.py report --tag $t | cut -c1-260)"
    $EV -u $CE run --suite mbpp --url http://localhost:$p2 --tag $t > /workspace/${t}_mbpp.log 2>&1 \
      && log "$t mbpp (cx3c ~22,5 · cx3b 23,3) · $(grep '###' /workspace/${t}_mbpp.log | tail -1 | cut -c1-200)" || log "LỖI C8 $t mbpp" ) &
  JOBS="$JOBS $!"
done
wait $JOBS                                                     # KHÔNG 'wait' trơn — sẽ chờ cả máy chủ UI
log "C8 lỗi máy chủ: $(cat /workspace/c8_ui_*.log | grep -c '\[lỗi /api/chat\]') lần"
kill $PIDS
tar czf /workspace/c8_ketqua.tgz out/code_eval/*cx3d* out/code_eval/*cx3e* out/office_probe_cx3d.jsonl out/office_probe_cx3e.jsonl \
    ckpt/expert_code/*cx3d*.json ckpt/expert_code/*cx3e*.json /workspace/cx3d_*.log /workspace/cx3e_*.log /workspace/c7_eval_cx3[de]_* 2>/dev/null
up /workspace/c8_ketqua.tgz backup/cx3/c8_ketqua.tgz && log "C8 đã đẩy HF kết quả đo + log"
log "C8 đo lại XONG — LoRA khối trộn cx3d (r32) · cx3e (r128) so với cx3c (cùng dữ liệu, không LoRA): xem các dòng 'C7 đo'"
