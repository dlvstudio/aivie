# Kiểm chứng cổng expert — P3 text (aivie_2p6b_p3_text_bf16)

Ngày 14/10/2026, chạy trên Mac M4. Script: `scripts/run_gate_ablation.sh`, `scripts/run_gate_ablation_off.sh`, `scripts/gate_probe_office.py`.

## Câu hỏi
Cổng expert lúc học đóng gần hết (code ≈0,003, vp ≈0,01–0,05, hoc ≈0,03–0,07). Có phải vì cổng đóng mà lập trình chỉ 1,0/5?

## Ba chế độ
- **normal**: cổng như đã học
- **open**: bỏ cổng, 3 expert chữ mở hẳn (cổng = 1)
- **off**: tắt 3 expert chữ, chỉ còn thân

## Kết quả

| Chế độ | HumanEval pass@1 (40 bài, không sửa) | Excel đúng (27 bài, chạy công thức thật) |
|---|---|---|
| normal | **4/40 = 10%** (HumanEval/7, 22, 23, 29) | **17/27** |
| open | 0/29, ra rác → dừng sớm | không chạy |
| off | **0/40** | **3/27** |

## Ví dụ thật

**HumanEval/3 `below_zero`**
- normal: viết code Python đúng khuôn (`def below_zero(operations: List[int]) -> bool: ...`). Logic sai, nhưng là code.
- off: `Hàm của bạn trả về True nếu số tiền trong tài khoản của bạn giảm xuống dưới 0 ...` → chỉ giải thích, không viết code.
- open: `mở_____ năng hiện lớn,,,,,,,,,,,,,,,,,,,,,,,, ...` → rác.

**Excel, trung bình lương phòng "Nhân sự"**
- normal: `=AVERAGEIF(C2:C6,"Nhân sự",D2:D6)`, chạy ra đúng.
- off: `=AVERAGE(C:C, A:A)`, sai hàm và sai vùng.

## Kết luận
1. **Cổng đóng KHÔNG phải nguyên nhân yếu.** Cổng nhỏ nhưng FFN expert học đầu ra rất lớn để bù, nên tích cổng × FFN vẫn là một tín hiệu đáng kể. Bằng chứng: tắt expert thì Excel tụt từ 17 xuống 3, code từ 10% xuống 0%.
2. **Không được mở cổng.** Cổng và FFN dính nhau vì học cùng lúc; mở cổng thì đầu ra quá lớn phá thân.
3. **Trần nằm ở thân và dữ liệu.** Expert chỉ lái thân, không thêm kiến thức. Thân mới học 20B token, ít code, nên ngay cả khi có expert, logic code vẫn sai.

## Đề xuất
- **Không cần học lại expert trước P3.3'.** Box cứ chạy tiếp như kế hoạch.
- Học lại 2 giai đoạn (cổng mở trước, học cổng sau) chỉ làm "sạch" hơn, nên gộp vào TK_108.
- Đòn bẩy thật cho code là TK_108:
  - **C1:** cho thân học tiếp nhiều code (cập nhật thân, trộn dữ liệu cũ).
  - Sau đó cố định thân mới, học lại expert theo 2 giai đoạn.
  - Dạy chạy–sửa / tool use.
