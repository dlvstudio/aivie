# Robot đá bóng — mạng GRU tự học trí-nhớ (luyện trên M4)

Mạng hồi-quy **thật** thay cho bộ-nhớ-tri-giác viết tay (tracker) trong demo `robotball`
(artifact `38b4c92c-1fa0-4e8b-b421-6a4fde07840c`). Đã SHIP làm lựa-chọn **🧠 GRU (tự học nhớ)**
bên cạnh **🧠 Mạng (tracker)** và **⚙ Luật**.

## Kiến-trúc
- `nn.GRU(IN=55, H=64)` → `Linear(64→48)` relu → `Linear(48→2)` tanh = [steer, thr*2-1].
- Vào (55): tốc-độ + la-bàn sin/cos-θ + **hướng gôn-mình theo trí-nhớ camera** (seen,sin,cos,dist — BỀN) + camera 120° thô (3×16 bins: bóng|đối-thủ|gôn).
- GRU **tự học giữ trí-nhớ bóng** trong trạng-thái ẩn (phần khó). Hướng-gôn-mình phải cấp sẵn (xem Bài-học).

## Luyện (PyTorch/MPS trên Apple M4)
```
# 1. dump dữ-liệu thầy (rule vs rule, cân 2 phe)
node dump_seqs.mjs                     # → seqs.json  (820 chunk × T=120 × 55)
# 2. GRU gốc (BPTT imitation)
python ../scripts/train_gru_bptt.py --device mps --epochs 140 --seqs seqs.json --out gruw.json
# 3. DAgger 3 vòng (cân phe): chạy GRU tự lái, thầy gán nhãn, gộp, luyện lại
cp seqs.json seqs_agg.json
for R in 1 2 3; do node dagger_dump.mjs; \
  python ../scripts/train_gru_bptt.py --device mps --epochs 120 --seqs seqs_agg.json --out gruw.json; done
# 4. đo & chọn checkpoint
node _cmp.mjs gruw.json 8              # so với: node _cmp.mjs tracker 8
```
`gruStep` trong `robotball.html` KHỚP GRU PyTorch (parity Δ≈1e-6). Nhúng: thay `/*__GRUW__*/ null` bằng JSON.

## Kết-quả (8 ván/cặp)
| | net ghi / thủng vs ⚙ | net-vs-net |
|---|---|---|
| 🧠 GRU (g1, đã ship) | ~82 / ~225 (Xanh & Đỏ đều nhau) | ~19-17 |
| 🧠 tracker (MLP) | ~118 / ~213 | ~20-22 |

GRU đạt **~70% số bàn của tracker**, đá **cả hai phe**, sống-động ở net-vs-net. **Chưa vượt** tracker về ghi bàn.

## Bài-học (trung-thực)
1. **SSM-chéo** (Mamba nhỏ) quá yếu: đá 1-81 → phải dùng **GRU có cổng**.
2. **BPTT imitation** thuần: loss thấp (0.056) nhưng đá 2-157 (lỗi tích-luỹ open-loop) → **DAgger** cứu (chạy on-policy).
3. **Thiên-lệch phe** (đá Xanh được, Đỏ sụp 0 bàn): camera thuần KHÔNG cho tín-hiệu bền "gôn nào của mình".
   Cân dữ-liệu 2 phe KHÔNG sửa được (giới-hạn cấu-trúc) → phải **thêm hướng-gôn-mình (trí-nhớ camera) vào input** (IN 51→55).
   Sau đó Đỏ ghi 11-25 bàn, net-vs-net cân → mạng học phần khó (nhớ bóng), chỉ cần cấp phần nó không suy được (phe mình).
