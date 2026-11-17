# Nhận-diện đồ vật — CNN tự-viết, luyện trên M4

Demo thứ 4 của nn-aivie3. CNN **viết tay từ-đầu** phân-loại **CIFAR-10** (10 loại vật, ảnh 32×32),
luyện PyTorch/MPS trên Apple M4, suy-luận **ngay trong trình-duyệt** bằng JS.
Artifact: `https://claude.ai/artifact/TdwxbToW8XdAS7rnKhTWgB`.

## Kiến-trúc (~90k tham-số)
```
Conv(3→16,3x3,pad1)+BN+ReLU + MaxPool2   # 32→16
Conv(16→32,3x3,pad1)+BN+ReLU + MaxPool2  # 16→8
Conv(32→64,3x3,pad1)+BN+ReLU + MaxPool2  # 8→4
Flatten(1024) → Linear(1024→64)+ReLU → Linear(64→10)
```
- **KHÔNG cần torchvision**: nạp CIFAR-10 pickle bằng numpy, augment (crop pad4 + flip) tự viết.
- **BatchNorm GẬP vào conv** lúc xuất (`fold_bn`) → forward JS chỉ cần conv+relu+maxpool+fc+softmax.

## Chạy
```
# 1. tải CIFAR-10 (1 lần) vào ../../data/
curl -sL https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz -o ../../data/cifar-10-python.tar.gz
tar xzf ../../data/cifar-10-python.tar.gz -C ../../data/
# 2. luyện (MPS) → cnnw.json + samples.npz
python train_cnn.py --epochs 24 --device mps
# 3. (tùy) làm gọn trọng-số: round 5 chữ-số → ~0.8MB
# 4. đối-chiếu forward numpy(=JS) vs PyTorch
python check_parity.py
# 5. nhúng trọng-số + ảnh mẫu → vision_pub.html
python build_pub.py
```
`vision.html` có forward JS KHỚP PyTorch (đã kiểm parity: JS cho ra đúng số numpy, vd mèo 96.8% / máy bay 70.3%).

## Kết-quả THẬT (24 epoch, ~1.8s/epoch trên M4)
- **Test tổng: 83.1%** trên 10 000 ảnh CIFAR-10 chưa từng thấy.
- Theo lớp: ô tô 93 · tàu 91 · ếch 90 · xe tải 88 · ngựa 86 · máy bay 85 · hươu 83 · chó 77 · **chim 73 · mèo 66**.
- Yếu nhất mèo/chim — kinh-điển ở 32×32 (dễ lẫn). Báo thẳng, không giấu.

## BỘ NÃO THỊ-GIÁC (gộp vật + mặt) — `brain.html`
Gộp 2 mạng vào 1 trang: thả ảnh → mạng-vật (cnnw_hd, 160px, 11 lớp) đoán "đây là gì"; nếu **người** (top-1 hoặc
P(người)≥0.3) → chạy mạng-mặt (facew, ArcFace) so **sổ mặt** (localStorage) → "và là AI / người lạ" + ghi-danh.
Artifact: `https://claude.ai/artifact/Jf66CmQkfJoRBozp8Yo3ey`. Build: `python build_pub_brain.py` (~8.4MB).
- **Bẫy 2 crop:** mạng-vật học người từ LFW **full-frame**; mạng-mặt học **crop sát 110/250**. 1 ảnh mẫu không hầu được cả hai.
  Giải: lưu mẫu **full-frame 160** (mạng-vật 15/16 ra người) + `prep()` cho mạng-mặt **tự crop giữa 0.44** (khớp train).
  Đo: mặt→người 15/16, face-ID one-shot 6/8 (mẫu). Ảnh tải lên nên là mặt căn-giữa kiểu LFW.

## Trung-thực / giới-hạn (bản CIFAR 32×32)
- Ảnh vào ép 32×32 (như dữ-liệu luyện). Ảnh ngoài 10 loại → mạng vẫn buộc chọn 1 trong 10, độ tự-tin thấp.
- Suy-luận thuần JS, không máy-chủ; trọng-số nhúng thẳng (~0.8MB sau round 5 chữ-số, accuracy không đổi).
- `train_cnn11.py` = biến-thể 11 lớp (thêm "người" từ siêu-lớp people của CIFAR-100) — sẵn nếu cần bản có người.

## Bản ĐỘ-PHÂN-GIẢI CAO — 128px, 11 lớp có "người" (`*_hd`)
Ảnh **nét thật** + có lớp người. Artifact riêng: `https://claude.ai/artifact/Ks62mJjCgDC8DRXoaP1Z5F`.
- **Dữ-liệu:** Imagenette (10 lớp ImageNet ~160px, ~99MB) + **người** từ **LFW** (khuôn-mặt thật ~250px, ~173MB
  qua figshare) — người nét cao, KHÔNG dùng CIFAR-100 32px (sẽ thành "lớp mờ" mạng ăn-gian).
- **Kiến-trúc:** 4 conv+BN + **Global-Average-Pool** + 1 FC (~165k tham-số). Input **128×128** (16× điểm-ảnh so 32×32).
  Forward JS tổng-quát theo **danh-sách conv**, input_size đọc từ meta (`vision_hd.html`). Parity Node OK.
- **VAL: 96px 84.4% → 128px 87.0% → 160px 86.7%**. Người ~98% (mặt rất khác 10 đồ-vật → dễ tách; LFW là miền hẹp
  chân-dung, không phải phát-hiện người mọi cảnh). Yếu nhất: cưa máy ~72% / trụ xăng ~74%.
- **Phát-hiện thật:** nâng 128→160px **KHÔNG cải-thiện** accuracy (còn nhỉnh thấp hơn tí) — nút thắt là **sức-chứa
  mạng (~165k tham-số)**, không phải độ nét. 160px chỉ *nhìn rõ hơn* chứ không *đoán giỏi hơn*, lại chậm hơn.
- Chạy: `python train_hd11.py --size 160 --load 176 --epochs 34 --device mps` → `python build_pub_hd.py`. ~10s/epoch @160.
- Lịch-sử: `train_imagenette.py` = bản gốc 10 lớp @96px (val 84.4%) trước khi thêm người + nâng 128px.
- Giới-hạn thật: 128px vẫn nhỏ so mắt người; tăng 224px nét hơn nhưng forward JS trong tab ~1-2s/ảnh + trọng-số nặng.
