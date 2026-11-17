#!/bin/bash
# TK_107 · user duyệt 2026-10-12: dữ liệu train để CÔNG KHAI HẾT → đẩy nốt phần box_vlm_data.sh / box_vlm_code.sh chưa đẩy:
#   trang báo (news) + ảnh gốc bộ dữ liệu ngoài (nat/img: COCO·KTVIC·OpenViVQA · caul/img: the_cauldron · chart/img: ViChart)
#   + README ghi nguồn & giấy phép gốc. Chạy SAU "VLM CODE XONG"; mỗi phần 1 file tar, đẩy xong xoá tar để đỡ ổ.
cd /workspace/nn-aivie
S=/workspace/vlm_data_status.log
export PATH=/workspace/vllm_env/bin:$PATH HF_HUB_DISABLE_XET=1
log() { echo "$(date '+%F %T') $*" >> $S; }
HF_VLM=${HF_VLM:-donaldluong/nn-aivie-vlm-data}
until grep -q "VLM CODE XONG" $S; do sleep 120; done
U=/workspace/vlm_up2; mkdir -p $U
cat > $U/README.md <<'EOF'
---
license: other
language: [vi, en]
task_categories: [visual-question-answering, image-to-text]
---
# nn-aivie VLM data (TK_107)

Dữ liệu dạy model **aivie** đọc ảnh (tiếng Việt). Mỗi phần là một file tar; giải nén tại thư mục gốc repo `nn-aivie` → `data/vlm/…`.

| File | Nội dung | Nguồn / giấy phép |
|---|---|---|
| `meta.tar.zst` | mọi file jsonl: align / sft / sft_ocr / sft5 / test_* · lời thầy (mô tả ảnh, hỏi–đáp dịch, đáp án câu cần hiểu) | tự tạo (thầy Gemma-4-31B) |
| `screens.tar`, `screens5.tar`, `screens_code.tar` | ảnh chụp màn hình tự dựng (Excel, email, văn bản, biểu đồ, chat, code, terminal) | tự tạo; nội dung chữ từ dữ liệu của dự án |
| `ocr.tar`, `ocr5.tar` | ảnh chữ tổng hợp | tự tạo |
| `news.tar` | trang báo giả (tên báo giả) | chữ: FineWeb-2 (ODC-By); ảnh minh hoạ lấy từ các bộ ở dưới |
| `nat_img.tar` | ảnh đời thường | COCO (CC BY 4.0, ảnh Flickr theo giấy phép riêng) · KTVIC · OpenViVQA (MIT) |
| `caul_img.tar` | ảnh biểu đồ / tài liệu / khoa học | HuggingFaceM4/the_cauldron — mỗi tập con theo giấy phép GỐC của nó |
| `chart_img.tar` | biểu đồ tiếng Việt | Zenng2812/vqa-vietnamese-charts |

Ảnh của các bộ dữ liệu bên ngoài vẫn thuộc giấy phép gốc của bộ đó — người dùng lại phải tuân thủ giấy phép gốc.
EOF
for p in news nat/img caul/img chart/img; do
  n=$(echo $p | tr / _)
  [ -d data/vlm/$p ] || { log "LỖI đẩy-hết: thiếu data/vlm/$p"; continue; }
  tar -cf $U/$n.tar data/vlm/$p
done
hf upload-large-folder $HF_VLM $U --repo-type dataset > /workspace/vlmd_upall.log 2>&1 \
  && { log "đẩy-hết: trang báo + ảnh gốc ($(du -sh $U | cut -f1)) + README → HF $HF_VLM"; rm -f $U/*.tar; } \
  || log "LỖI đẩy-hết (vlmd_upall.log) — tar vẫn ở $U"
log "VLM UPLOAD ALL XONG"
