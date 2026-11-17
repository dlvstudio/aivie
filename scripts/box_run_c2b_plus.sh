#!/bin/bash
# TK_108 C2b+ (CPU): tải 9 bộ code hiện đại (Next.js/React, Tailwind, Svelte 5, Kotlin Compose, Flutter, shadcn, glaive…)
# → kiểm cú pháp tree-sitter / áp lại bản vá → data/c2b/plus_*.jsonl → HF. Trạng thái → p3_status.log (tiền tố "C2b+").
# Dùng: (setsid nohup bash scripts/box_run_c2b_plus.sh > /workspace/c2b_plus_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1
EV=/workspace/evalenv/bin/python; R=/workspace/c2b_raw/plus
mkdir -p $R
log "C2b+ bắt đầu: tải Next.js · Tailwind · Svelte 5 · Kotlin Compose · Flutter · FrontendCookbook · glaive v3 · shadcn · WebSight (chỉ HTML)"
dl() { hf download $1 --repo-type dataset --local-dir $R/$2 > /dev/null 2>> /workspace/c2b_plus_dl.log || log "LỖI C2b+: tải $1"; }
dl Tesslate/Next.js-Dataset nextjs &
dl kiiwee/tailwind_component_llama3_dataset tailwind &
dl Mungus451/svelte-5-expert-sft svelte &
dl giggiovpg/android-kotlin-compose-compiler-verified kotlin &
dl bbidpa/flutter-full-examples-v1 flutter_full &
dl bbidpa/flutter-diff-steps-v1 flutter_diff &
dl Tensoic/FrontendCookbook fecook &
dl glaiveai/glaive-code-assistant-v3 glaive &
dl valentin-marquez/react-shadcn-codex shadcn &
wait
log "C2b+ tải xong · $(du -sh $R | cut -f1)"
$EV -u scripts/c2b_plus.py --raw $R --workers 24 > /workspace/c2b_plus.log 2>&1 || { log "LỖI C2b+: lọc (c2b_plus.log)"; exit 1; }
log "$(grep '###' /workspace/c2b_plus.log | sed 's/### //')"
for f in data/c2b/plus_*; do hf upload donaldluong/nn-aivie-2p6b-p3 $f $f > /dev/null 2>&1; done
log "C2b+ xong — đã đẩy HF nn-aivie-2p6b-p3/data/c2b/plus_*"
