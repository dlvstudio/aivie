# Thí nghiệm C — Distillation (gemma4:26b dạy mạng 54M)

- **Thầy**: gemma4:26b-a4b-it-qat (Ollama, think=false — BẮT BUỘC, không thì content rỗng). Tiếng Việt xuất sắc, ~1-2s/đoạn.
- **Data**: 972 đoạn văn sạch (~95k token) do gemma sinh về nhiều chủ đề (cắt sớm vì thời gian; nhiều hơn sẽ mạnh hơn).
- **Trò**: d512mix (54M) warm-start + fine-tune 100 bước (600 bước OVERFIT nặng — data quá nhỏ).

## Hiệu ứng (LM thuần, distill vs d512mix)
| Prompt | d512mix | distill (sau chưng cất) |
|---|---|---|
| biến đổi khí hậu | "...Tham khảo. Liên kết ngoài. Đô thị bang São Paulo... Cựu sinh" (rác template) | "...khu vực nhiệt đới... vùng đất ngập mặn... kinh tế và văn hóa..." (văn mượt) |
| âm nhạc là | "...bát nhạc * hát theo điệu nhạc . nghĩa là từ dùng..." (mảnh từ điển) | "...loại hình nghệ thuật biểu diễn... nhạc sĩ để lại dấu ấn sâu sắc..." (essay) |
| khoa học là | "một trường đại học, thành lập 1950" (cụt/lệch) | "ngành khoa học nghiên cứu về sinh học và sinh thái... vai trò quan trọng..." (bám chủ đề) |

## Kết luận (đúng dự đoán)
- ✅ **Phong cách SẠCH hơn rõ**: học từ đích-sạch-của-gemma → bỏ rác template Wikipedia + mảnh từ điển, chuyển sang văn xuôi essay mượt.
- ❌ **KHÔNG truyền được suy-luận**: vẫn trôi chủ đề (biến đổi khí hậu → "châu Á"). 54M không hấp thụ nổi khả năng của 26B (chênh ~130×).
- ⚠️ Data 95k token quá nhỏ → dễ overfit + lặp. Cần ≥vài triệu token gemma để hiệu ứng ổn định.

**Distillation = "làm trò sạch sẽ/chuyên nghiệp hơn trong trần của nó", KHÔNG làm trò thông minh.**
