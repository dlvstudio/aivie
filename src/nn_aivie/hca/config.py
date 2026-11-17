"""HCA configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HCAConfig:
    k_max: int = 24
    lambda_frame: float = 0.3
    lambda_bind: float = 0.7
    # Composition #2 — multi-hop spreading activation (general, graph-only).
    spread_hops: int = 2
    spread_gamma: float = 0.45
    spread_weight: float = 0.6
    spread_max_frontier: int = 48
    spread_threshold: float = 1e-3
    spread_use_csr: bool = True  # scipy CSR matvec thay BFS dict mỗi hop (inference read-only)
    # Composition #4 — G4 verify (ngoại suy có kiểm soát; scope: phiên QA).
    verify_enabled: bool = True
    verify_min: float = 0.04
    # Copy/pointer — đáp tham chiếu lại thực thể câu hỏi (chống bias tần suất).
    copy_weight: float = 2.5

    # --- Lõi sinh hybrid v3 (generate_hebbian) — mỗi term ablate được bằng cách =0 ---
    hybrid_macro_weight: float = 1.0   # top-down concept field (lan ngược)
    hybrid_macro_floor: float = 0.05   # sàn macro: token off-agenda bị nén (gate Hadamard thật)
    hybrid_micro_weight: float = 1.0   # W_seq bigram (kênh fluency)
    hybrid_micro_floor: float = 0.15   # sàn micro để macro candidate không bị Hadamard giết
    # SỔ NỢ (đổi luật 2026-07-02 — không tắt cơ chế): fuzzy BẬT lại. Giá đã đo: big_a2
    # Syn 33→50 KHÔNG hại QA/Para; grade_v3 (corpus nhỏ) Para 75→62 — nghi thiếu data đồng
    # hiện, fix = scale corpus/E-episode cụm (Trụ 7 T3), KHÔNG phải bug kiến trúc.
    hybrid_fuzzy_topk: int = 3
    hybrid_fuzzy_weight: float = 0.35  # fuzzy_scale
    hybrid_fuzzy_min_sim: float = 0.35
    hybrid_parent_min_score: float = 0.18  # dưới ngưỡng → coi như không có cha khớp
    hybrid_bottomup_hops: int = 2      # số hop lan xuôi token→concept
    hybrid_temperature: float = 0.0    # 0 = argmax (xác định); >0 = multinomial sampling
    hybrid_top_k: int = 12             # giới hạn ứng viên khi sampling
    hybrid_top_p: float = 0.0          # 0 = tắt; (0,1) = nucleus trên top-k
    hybrid_eos_threshold: float = 0.05 # cạnh w0→EOS đã học vượt ngưỡng → đóng câu
    # dội verbatim (mềm trật tự từ). Mặc định TẮT (verbatim giữ fact chính xác nhất).
    # Nơ-ron mẫu chung (slot hub) — guard chống hallucination chéo filler: câu hỏi thuộc họ
    # mẫu với filler LẠ (chưa nối) mà đáp lại nói về filler ĐỐI THỦ (Lý khi hỏi Trần) → không
    # bịa; ưu tiên ứng viên slot-consistent (đáp nhắc đúng thực thể hỏi, kể cả từ statement).
    hybrid_hub_guard: bool = True
    # Hướng 1 (WH/frame cluster activation) — cụm phổ biến (df > phrase_max_df: "ở đâu", "là gì")
    # KHÔNG còn bị loại khỏi đơn vị match: nạp làm unit, cân bằng bằng IDF (thấp) + phạt
    # distinct-ratio² đã có → không nhấn chìm thực thể. Nhờ đó nơ-ron WH được KÍCH THÍCH (hiện,
    # gom nhóm qua anchor). ĐO A/B (2026-07-04): 0 flip top-1 trên QA grade_v3 (21 ca) + ISG (7).
    # =False → khôi phục hard-gate df cũ (0.5.9/0.5.10).
    match_common_clusters: bool = True

    # --- A3: sinh AUTOREGRESSIVE token-level (docx §4: a_final = a_macro ⊙ a_micro) ---
    # KHÁC dội verbatim: mỗi bước tính điểm toàn vocab, chọn 1 token, lặp → dựng chuỗi token,
    # có thể sinh câu chưa lưu (đảo ngữ/ghép). a_macro = spreading từ agenda (ý cần phủ, do A1
    # cấp), a_micro = W_seq bigram (mượt).
    ar_micro_floor: float = 0.15   # sàn a_micro (W_seq) để a_macro không triệt ứng viên mượt
    ar_macro_floor: float = 0.10   # sàn a_macro để token bắc-cầu off-agenda vẫn có cơ hội
    ar_spread_weight: float = 1.0  # trọng số trường spreading từ agenda
    ar_agenda_boost: float = 0.6   # ưu tiên token agenda còn thiếu, đạt được từ w0
    ar_temperature: float = 0.0    # 0 = argmax; >0 = multinomial
    # --- TK_046 §22: nối CHUỖI BA BƯỚC vào đường sinh ---
    # `hub_spread` dựng xong từ TK_046 §18 nhưng CHƯA có caller nào trong `src/`
    # — suốt chín vòng nó được đo bằng thước TRUY HỒI (top-k + chấm nhị phân)
    # trong khi thiết kế của tác giả là cơ chế SINH (prompt → phân bố token kế,
    # `khỉ` thấp hơn `chó`/`mèo` nhưng KHÁC 0).
    #
    # Đo trên ca thử vàng + nhiễu, prompt «nghịch ngợm như loài», hạng của `khỉ`:
    #     cha=895    h479 → h8    (60×)
    #     cha=3069   h1560 → h28  (56×)
    #     cha=10287  không đổi — bước ① đã trôi sang bạn ghép từ (§21.4)
    #
    # 0 = TẮT (mặc định — không đổi hành vi bất kỳ caller nào).
    # BẬT MẶC ĐỊNH (§79). Đo đầu-cuối trên ca sạch (`thuoc_sinh_dau_cuoi`),
    # chấm PHÂN BỐ CUỐI sau khi bơm:
    #     180 ca      boost 0 → top-5 76,7% · top-10 78,9%
    #                 boost 2 → top-5 78,3% · top-10 80,0%
    # Đo lại qua ĐƯỜNG PRODUCTION (`generate()` → `_w_skip_primary_loop`) sau
    # khi §79 nối chuỗi vào đó — 55 ca lớp CỤ THỂ, token kế ĐÚNG LỚP:
    #     boost 0 → 74,5%   ·   boost 1 → 76,4%   ·   boost 2 → 76,4%
    # Chọn 1.0: cùng mức 76,4% mà KHÔNG rối đuôi câu như 2.0 («chó là gia súc
    # thuộc nhóm ăn thịt nuôi nhà để, giữ, để nuôi»).
    # Và chỗ nó thật sự đóng góp là prompt LOẠI SUY, nơi mô hình nền không có
    # gì để nói: «nghịch ngợm như loài ___» chỉ ra `khỉ` KHI BẬT (§78.4).
    hub_chain_boost: float = 1.0
    hub_chain_d_max: int = 6
    hub_chain_min_count: float = 0.5   # xem memory `w-skip-min-count-is-weight`
    # Ghép CẢ HAI vế + lọc bậc-vào: phủ cạnh lớp 28,6% → 68,4% trên classrich
    # (TK_046 §24). indeg_q=0,8 giữ nguyên phủ mà bỏ 26% cạnh rác.
    # Lưới ĐẦY ĐỦ 4×4 trên classrich (TK_046 §29) — quét một vế bỏ sót tương tác:
    #   trai_du_dac_thu | phai_nhat        phủ 69,5%  ②top1 23,5%  top5 39,7%  ← chọn
    #   dau_lop         | phai_nhat        phủ 67,4%  ②top1 20,0%  top5 54,3%  ← dư địa lớn nhất
    #   trai_nhat       | phai_nhat        phủ 71,8%  ②top1 18,1%  top5 50,0%
    #   trai_du_dac_thu | phai_du_dac_thu  phủ 68,4%  ②top1 17,5%  top5 36,5%
    #   max_gidf        | max_gidf (gốc)   phủ 17,9%  ②top1  0,0%  top5  4,8%
    # `phai_nhat` — luật tôi khai tử ở §21 — thắng ở MỌI cặp.
    hub_chain_head_rule: str = "trai_du_dac_thu"
    hub_chain_head_rule_trai: str = "phai_nhat"
    # ⚠️ HAI BỘ LỌC PHÂN VỊ TẮT — QUYẾT ĐỊNH CỦA TÁC GIẢ (§57): ƯU TIÊN NĂNG LỰC.
    # §56 đo được xung đột THẬT giữa ba thước, không cấu hình nào thắng cả ba:
    #     phai_nhat + lọc cả hai      ngưỡng 3 069     ②top1 15,2%   sinh 0,290
    #     phai_nhat_danh_tu + tắt     ngưỡng ≥20 316   ②top1 19,4%   sinh 0,995
    # Tác giả chọn NĂNG LỰC (chuỗi chạy được ở quy mô lớn) thay vì chất lượng
    # sinh trung bình. Đổi hai dòng dưới là quay về hướng kia.
    # Thêm ở §24 để tăng độ chính xác, nhưng
    # khi corpus lớn lên phân bố DỊCH và cùng một phân vị cắt vào CẠNH VÀNG:
    #     cha=10287  gidf_q=0,2 indeg_q=0,8 → via_pivot[khỉ] = RỖNG
    #                gidf_q=0   indeg_q=0   → via_pivot[khỉ] = [(động, 0,95)]
    # Tắt cả hai: ngưỡng nhiễu 3 069 → ≥20 316 cha (6,6×) và classrich ②top1
    # 15,2% → 19,4%. Phân vị vốn được dùng để TRÁNH bẫy ngưỡng-hằng-không-chuyển-
    # quy-mô, nhưng nó mắc đúng bẫy đó ở dạng khác.
    hub_chain_indeg_q: float = 0.0
    hub_chain_gidf_q: float = 0.0
    # NÚM RIÊNG cho việc CHỌN ĐẦU ngữ đoạn (§64). `hub_chain_gidf_q` điều
    # khiển ba việc; §55 vặn nó về 0 để tắt bộ LỌC CẠNH và vô tình tắt luôn
    # ngưỡng «đủ đặc thù» của luật chọn đầu ⇒ nhãn thành ĐỊNH TỪ (1 142 cạnh
    # trỏ vào `một` · 58 vào `các` · 45 vào `những`). Tách ra thì giữ được cả
    # hai: lọc cạnh TẮT (năng lực) mà đầu ngữ đoạn vẫn phải là từ đặc thù.
    # Quét trên `classrich` với chuẩn chân lý v2 — nhãn ĐÚNG chuẩn:
    #     0,00 → 321   ·   0,10 → 403   ·   0,20 → 403   ·   0,40 → 246
    hub_chain_head_gidf_q: float = 0.20
    hub_chain_pos_dict: str = "data/dict/vi_dict_raw.jsonl"
    # Xếp nhãn ở bước ②. Đo trên classrich, 119 ca có sẵn điều kiện cần —
    # ② chọn đúng lớp (TK_046 §28):
    #     diem         top-1  0,8%  · top-5  5,0%   ← xếp theo điểm: gần như câm
    #     phieu        top-1 21,8%  · top-5 43,7%   ← ĐỒNG THUẬN, tốt nhất
    #     phieu_chuan  top-1  7,0%  · top-5 14,0%   ← chia cho bậc-vào: TỆ HƠN
    # `phieu_chuan` hỏng vì LỚP NHIỀU THÀNH VIÊN thì bậc-vào CAO theo định
    # nghĩa — chia cho nó là phạt đúng đáp án. Giữ lại làm đối chứng âm.
    # `phieu_x_ctx` — nhân điểm với phần NGỮ CẢNH khớp CHỮ KÝ của nhãn (§43).
    # Tín hiệu đầu tiên nhích được ② một cách nhất quán:
    #     n=157: 38,2% → 41,4%   ·   n=230: 43,5% → 45,7%
    # Tám tín hiệu trước (gidf · bậc-vào · khứ hồi · E · từ loại · đồng thuận
    # thuộc tính · chuẩn hoá phiếu · biến thể id) đều ~0 — chúng suy ra được từ
    # đồ thị đồng-hiện; ngữ cảnh câu là thứ `via_pivot` VỨT ĐI, nên nó mới thêm
    # được thông tin (§42.2).
    hub_chain_rank: str = "phieu_x_ctx"
    # Nhân trọng số theo SỐ THUỘC TÍNH trong ngữ cảnh cùng đề cử một thành viên
    # (0 = tắt). Tín hiệu ĐỘC LẬP duy nhất còn lại sau khi §35 loại gidf /
    # bậc-vào / khứ hồi — cả ba đều suy ra từ chính cạnh đang xét.
    hub_chain_dong_thuan: float = 0.0
    # TÁCH BỂ (§37): ② dùng `via_pivot` đầy đủ (cần bể PHIẾU rộng), ③ dùng
    # `via_pivot_sach` đã loại CHỦ NGỮ GIẢ do trạng ngữ («Đặc biệt là ngành…»).
    # §36.4 đo: lọc chung một bể thì ③ sạch hơn nhưng ②top1 tụt 35,8% → 29,4%.
    hub_chain_chu_gia_q: float = 0.90
    # KỀ NGỮ CẢNH ở bước ③ (§60). Ở quy mô lớn MỌI thành viên typed của một
    # nhãn đều nặng đúng 0,95 (mỗi câu định nghĩa xuất hiện một lần) ⇒ thứ tự
    # rơi về ABC và `khỉ` (K) thua `Am`/`An`/`Biên`. Đo trên toy 99 362 cha:
    #     chỉ KỀ sau `loài`, bỏ chuỗi ②③ : bể 179 · khỉ hạng 131  ← đối chứng
    #     chỉ chuỗi ②③                   : bể  62 · khỉ hạng  35
    #     chuỗi ②③ × KỀ                  : bể  62 · khỉ hạng   4
    # Không tín hiệu nào tự đứng được: chuỗi thu bể bằng TƯ CÁCH THÀNH VIÊN,
    # ngữ cảnh xếp thứ tự TRONG bể. 0 = tắt.
    hub_chain_ke_ctx: float = 2.0
    # HẠT theo ĐỒNG THUẬN LỚP ở ① (§66). Trên `classrich2`, ① top-4 theo điểm
    # cho ≥2 đồng-thành-viên ở 0,0% ca trong khi top-60 có ở 11,8% ⇒ bước ②
    # xưa nay chạy với ĐẦU VÀO RỖNG. Chọn NHÓM ứng viên cùng trỏ về một nhãn:
    # ≥2 đồng-thành-viên 0,0% → 0,9%, ca ĐẠT 2 → 5 / 570.
    # ⚠️ MẶC ĐỊNH TẮT sau §67 D. Thước ca cho +2, nhưng ĐỌC ĐẦU RA thì nó làm
    # đường sinh TỆ ĐI: «cây bàng là ___» đang cho `Chó·Dê·Sư·Báo·Cá·Gấu·Hổ`
    # (một lớp sạch) thì bật vào thành `họ·Thủ·Patron·Campuchia`. Nhóm «lớn
    # nhất» hội tụ về nhãn khổng lồ chung cho mọi prompt. Giữ cờ để A/B.
    hub_chain_hat_dong_thuan: bool = False
    # LOẠI CẠNH THỨ BA (§69): thuộc tính theo VỊ NGỮ («chó rất trung thành»).
    # §68.2: kênh nuôi ① chỉ có 3 383 cạnh (pivot so sánh, 5,5% câu) trong khi
    # kênh nuôi ②③ có 50 686 — mỏng gấp 15 lần. Khuôn DANH TỪ … TÍNH TỪ dùng
    # từ loại từ điển đã bake.
    hub_chain_kenh_vi_ngu: bool = False
    # topk của bước ① (TK_052 vòng 2). §66 nới 10 → 60 để `chon_hat_dong_thuan`
    # có bể quét; nhưng luật hạt-chỉ-typed (§59) lấy 4 hit typed ĐẦU TIÊN, nên
    # bể rộng chỉ độn thêm hit typed YẾU vào `hat` và làm loãng ②.
    hub_chain_topk_1: int = 10
    # GỘP MẢNH BPE thành TỪ trước khi lọc nội dung (TK_052 vòng 8–9).
    # `_content_positions` loại mảnh không `is_surface` (`s` của `sóc`), nên
    # span WS không khớp được và cạnh lớp bị khoá dưới bề mặt MẢNH (`óc`).
    # Vá xong: `via_pivot[sóc]` có `thú` — ca đích của vòng 7–9.
    #     55 ca lớp CỤ THỂ  top-1 54,5% → 56,4%
    #     TK_052 tổng hợp   84,8% → 84,1%      (−0,7, trong nhiễu)
    #     ca vàng toy · thước sinh 300 ca      KHÔNG đổi
    # Vòng 10: phép gộp nay VÔ HIỆU khi từ vựng không đánh dấu ranh giới từ
    # (fixture), nên ca thử vàng không còn bị đụng ⇒ bật được mặc định.
    # §93 — lọc ĐÍCH của cạnh `A là B` không thể là LỚP (số/dấu câu/EOS, hoặc
    # từ điển nói không phải danh từ). Đích NGOÀI từ điển thì giữ. Đo §93.1:
    # 223 988 → 162 314 cạnh, danh từ 39,1% → 53,9%, cạnh vàng không mất.
    # §99 — cụm chủ ngữ không có mục định nghĩa ⇒ lùi về TỪ CHÍNH đặc trưng
    # nhất của cụm (`con chó` → `chó`, `chim sẻ` → `sẻ`). §98 đo tương quan 10/10
    # giữa «có mục định nghĩa» và «sinh đúng».
    lui_ve_tu_chinh: bool = False
    # §101 — kênh bắc cầu ở ①②: typed_first (mặc định) · typed · cooc · mixed.
    # `cooc` = bắc cầu bằng ĐỒNG HIỆN thuần, KHÔNG cần nhãn lớp — nếu chạy thì
    # toàn bộ nút thắt «nhãn lớp» (§89–§100) không còn liên quan.
    hub_chain_channel: str = "typed_first"
    # §103 — ghi đè kênh cho TỪNG bước ("" = theo `hub_chain_channel`).
    # §105 — ép cầu nối ② phải ĐẶC TRƯNG (§104: gidf≥6 cho 3,4σ, kiểm chia đôi
    # 3,0σ). 0 = mặc định cũ, hư danh nuốt sạch.
    # §106 — ngưỡng trọng số cạnh `via_attr` ở bước ①. Đo: rác (mảnh từ ghép
    # kiểu `lực`+`sĩ` từ `khoẻ như lực sĩ`) có max 2,40; vật mang chứng thực có
    # trung vị 3,40 ⇒ ngưỡng 2,5 cho chính xác 100%, phủ 67%.
    hub_chain_attr_w_min: float = 0.0
    hub_chain_gidf_min_2: float = 0.0
    # §105 — phát HUB ② thẳng làm đề cử (§104: hub tự chứa bạn cùng lớp).
    hub_chain_hub_lam_de_cu: bool = False
    hub_chain_hub_top: int = 0
    hub_chain_channel_1: str = ""
    hub_chain_channel_2: str = ""
    hub_chain_loc_dich: bool = False
    # §128 — bảng NHÃN LỚP bake từ khuôn định nghĩa (`nhan_lop.json`, 9 222 từ).
    # §127: nhãn từ `via_pivot` đúng 4,1%; bảng này 84,9% top-1 trên cùng chuẩn.
    # Trọng số 0 = TẮT. Xem `scripts/dung_bang_nhan_lop.py`.
    hub_chain_bang_nhan_lop: str = "data/dict/nhan_lop.json"
    # §134 — ĐÃ CHẤM trên bộ khoá SẠCH (3 026 từ có lớp ≥5 thành viên, mẫu
    # 400, đối chứng hoán vị): bật cờ KHÔNG đổi gì — top-5 27,5 → 27,3, khoảng
    # cách thật−hoán vị 3,6 → 3,4. Giữ ở mức 0 (TẮT). Mã để lại vì nó đúng và
    # là chỗ dùng duy nhất của bảng `nhan_lop.json` đã thẩm định (§128).
    hub_chain_bang_nhan_w: float = 0.0
    # §136 — gieo ③ thẳng từ ĐẢO của bảng (lớp → thành viên, hạt 2 từ) thay vì
    # qua `via_rev` của corpus. §135: 64,4% số từ có lớp ≥2 (mốc 38,5%). 0 = TẮT.
    hub_chain_bang_tv_w: float = 0.0
    # §138 — thang HỒI SINH cho token chuỗi chưa có trong phiếu: 0 = TRUNG VỊ
    # (giữ §12.144), >0 = tỉ lệ của ĐỈNH. Đo giá trước khi bàn đổi.
    ar_w_skip_chain_hoi_sinh: float = 0.0
    # §139 — chỉ cho đề cử chuỗi tác động nếu có CHỨNG CỨ bigram đứng sau token
    # vừa phát (`_seq_bi`). Chữa «biết từ, không biết chỗ» đo được ở §138.
    ar_w_skip_chain_vi_tri: bool = False
    # §140 — mỗi từ do chuỗi đề cử chỉ được phát MỘT LẦN mỗi câu. Chữa phần
    # thoái hoá còn lại sau cổng vị trí (§139): lỗi là TÁI XUẤT, không phải sai
    # vị trí cục bộ.
    ar_w_skip_chain_mot_lan: bool = False
    hub_chain_gop_manh_tu: bool = True
    hub_chain_n_bien_the: int = 4   # số id cùng bề mặt được gieo ở ①
    # GOM TOKEN PROMPT THÀNH TỪ ở ① (TK_052 vòng 3). `encode_prompt` tách 89%
    # tính từ thành mảnh BPE (`ngặt` → `ng`+`ặt`), nên vòng lặp theo TOKEN gieo
    # ① bằng MẢNH. Gom thành từ đưa thu hồi của TK_052 từ 44,1% → 70,3%.
    # ⚠️ MẶC ĐỊNH TẮT: trên `classrich2` nó làm ca SUY tụt 32,5% → 2,5%, vì hai
    # corpus mã hoá cùng một từ khác nhau — chỗ thì cạnh bám vào id CẢ TỪ, chỗ
    # thì bám vào MẢNH. Hợp cả hai cũng không cứu được (TK_052 48,3% ·
    # TK_046 SUY 14,2%): trần `n_bien_the` bị lấp bởi phía sai.
    #     0 = chỉ MẢNH (hành vi cũ) · 1 = HỢP · 2 = chỉ id CẢ TỪ
    #     TK_052 thu hồi top-10 :  0 → 44,1%  ·  1 → 48,3%  ·  2 → 70,3%
    #     TK_046 ca SUY top-1   :  0 → 32,5%  ·  1 → 14,2%  ·  2 →  2,5%
    hub_chain_gom_tu: int = 0
    hub_chain_bo_manh: bool = True     # §38 — bỏ token đã được cụm dài đại diện
    # CỘNG DỒN (§70): số nhãn ② được lan ngược và CỘNG điểm lên thành viên,
    # thay vì chọn đúng một nhãn. 0 = hành vi cũ (ba argmax nối tiếp).
    hub_chain_cong_don: int = 0
    # NHÃN TRỰC TIẾP (§71): hệ số cho cạnh `p → lớp` có sẵn của chính token
    # trong prompt. Chuỗi ba bước xưa nay LUÔN đi vòng (p → vật mang → nhãn →
    # thành viên) kể cả khi `p` đã có cạnh lớp của chính nó — tức bỏ phí thứ
    # nó đã học. Thước sinh v3 trên `classrich2`, 300 ca:
    #     0    top-1  5,7% · top-5 19,0%
    #     4    top-1 14,7% · top-5 35,0%
    #     8    top-1 20,7% · top-5 40,0%
    #     16   top-1 21,7% · top-5 42,0%   ← BÃO HOÀ (32 và 64 y hệt)
    # Trên 55 ca lớp CỤ THỂ (chim·thú·cây·cá…): 7,3% → 30,9% top-1, 27,3% →
    # 60,0% top-5; riêng ca SUY 10,7% → 42,9%.
    hub_chain_nhan_truc_tiep: float = 16.0
    # ĐỒNG-THÀNH-VIÊN của nhãn trực tiếp (§72): bước ③ áp cho nhãn ĐÃ BIẾT.
    # «chó là ___» thì `mèo`/`hổ` cũng hợp lệ. 0 = tắt.
    hub_chain_tv_truc_tiep: float = 0.0
    # KỀ NGỮ CẢNH cho đường TRỰC TIẾP (§72). Chẩn đoán: 67,2% ca trượt top-5
    # là lỗi XẾP HẠNG (cạnh tới lớp đúng CÓ SẴN nhưng bị nhãn khác của cùng
    # chủ ngữ vượt). Nhãn nào từng thật sự đứng ngay sau `là` trong câu train
    # thì được nhân (1+k). Thước sinh v3 · 300 ca:
    #     0 → top-1 21,7% / top-5 42,0%     2 → 24,3% / 45,7%
    #     8 → 25,0% / 45,3%                32 → 25,0% / 45,3% (bão hoà)
    # 55 ca lớp CỤ THỂ: top-1 30,9% → 43,6%.
    hub_chain_ke_truc_tiep: float = 2.0
    ar_eos_threshold: float = 0.05 # cho phép EOS khi agenda phủ đủ + w0→EOS đủ mạnh
    ar_order_decay: float = 0.5    # prior thứ tự (docx §2.3): boost token agenda kế tiếp theo
                                   # trật tự nguồn, giảm dần — index MỀM, W_seq vẫn quyết fluency
    ar_order_window: int = 3       # số token agenda kế tiếp được prior thứ tự
    # A3 phase-2: index backbone (docx §1 "index cứng") — v3 KHÔNG có cạnh vị trí R-slot (đo
    # được), nên trật tự phải dựa "thứ tự con trong cha". Token kế tiếp theo trật tự nguồn, nếu
    # ĐẠT ĐƯỢC từ w0 (W_seq/co-act) → cho thắng (xương sống); không đạt → W_seq bắc cầu (mềm).
    ar_index_backbone: bool = True
    # Trụ 2 (v4): kênh successor bậc-2 suy TỪ CHUỖI CHA (docx §1).
    # SỔ NỢ (đổi luật 2026-07-02): BẬT lại — đo 07-01 trung tính từng-token (index-backbone
    # chi phối, micro chỉ phá thế hòa); giữ sống để tiến hóa cùng Trụ 4 (decode dựa micro hơn).
    ar_seq_order2_weight: float = 0.5
    ar_seq_order2_min: float = 2.0   # tổng count tối thiểu để tin bậc-2 (chưa đủ → backoff bigram)
    # TK_031: VarOrderLM (PPM biến-bậc, ctx 1..4) trộn vào kênh micro — nhớ cụm dài → văn mượt.
    # Probe: −0.21 bits / +4pp top1 so bigram. Cần sidecar `<ckpt>.varlm.pkl`; thiếu = no-op.
    # LIFT tối thiểu của P(EOS | w0) so với P(EOS) nền, để coi w0 là mốc KẾT THÚC đơn vị
    # (at_boundary, TK_009). Thay `vocab.is_punct(w0)` — mốc phải HỌC từ dữ liệu, không đoán
    # từ danh sách dấu câu (Luật 0). 0 = tắt.
    #
    # Vì sao LIFT chứ không phải ngưỡng tuyệt đối: P(EOS|'.') phụ thuộc ĐỘ DÀI ĐƠN VỊ train,
    # nên hằng số tuyệt đối không chuyển được giữa corpus. Bản đầu chốt 0.25 từ corpus đồ chơi
    # (1 câu/đơn vị → P=1.000) và TRƯỢT trên network_v10 thật (đoạn 2-4 câu + bài dài → '.'
    # xuất hiện ~17 lần/đơn vị, chỉ 1 lần trước EOS → P=0.057 < 0.25 → ends_unit=False, verify
    # 2026-07-16 FAIL). Lift chia cho tỉ lệ nền nên tự hiệu chỉnh: corpus câu ngắn hay bài dài
    # đều cho cùng câu trả lời "token này kết thúc đơn vị nhiều hơn ngẫu nhiên bao nhiêu lần".
    ar_eos_boundary_min_lift: float = 2.0
    ar_varlm_weight: float = 1.0
    # TK_042 G10 — LM nơ-ron THAY kênh trôi chảy `micro` (= W_seq bigram).
    #
    # Đo G7 (CÙNG corpus 19.7M token + CÙNG held-out + CÙNG công thức bits):
    #   W_seq (micro hiện tại) = 7.088 bits/token · LM 11.4M = 5.594  ⇒ tốt hơn 1.494
    #   gap train/eval +0.811 (đã vắt được data) · 15/15 mốc eval ★ (không overfit)
    #
    # 0.0 = W_seq thuần (HÀNH VI CŨ, mặc định) · 1.0 = LM thuần · 0<w<1 = nội suy.
    # NỘI SUY chứ không NHÂN: Hadamard là AND — LM nói X, agenda nói Y ⇒ dìm cả hai.
    # Kênh `macro` (agenda) GIỮ NGUYÊN ⇒ đồ thị vẫn lái NỘI DUNG từ bên ngoài ⇒ LM không
    # cần in-context learning (thứ 11.4M tham số không có).
    ar_lm_weight: float = 0.0
    # TK_044 P3.2 — kênh COPY coreference (P3.1 cổng đạt).
    # "off" = không copy (nền P3.0). "hard" = argmax cứng (Plan B baseline — GIÒN BPE nhưng rẻ).
    # "hopfield" = truy hồi mềm Hopfield emb-SIG (Plan A — robust BPE, cần build KV-index).
    # Luật 0 CỨNG: copy lúc sinh CHỈ đọc TRỌNG SỐ đã học (coref/E/entity_register).
    # CẤM gọi cap_class/string/NER ở đây → vi phạm = kill-criteria (RUNBOOK §1).
    ar_copy_channel: str = "off"    # "off" | "hard" | "hopfield"
    ar_copy_boost: float = 15.0     # Plan B/A: score entity tokens tăng thêm bao nhiêu khi copy slot
    ar_copy_beta: float = 5.0       # (dành RETRIEVE antecedent sau — DETECT dùng cosine, xem dưới)
    ar_copy_cos_thresh: float = 0.5     # Plan A: fire nếu cosine(ctx, copy-slot đã học) VƯỢT ngưỡng
                                        # (KHÔNG dùng softmax-weight: chia mass cho n_coref → không scale)
    ar_copy_entity_gidf_min: float = 3.0  # gate entity_ids theo gidf (bỏ function word 'rất/học' khỏi
                                          # sentinel-SIG + boost). gidf = thống-kê học được, Luật-0 OK.
    ar_lm_path: str = ""  # "" = tắt. vd "out/g7_best_model.pt" (cần torch + vocab KHỚP)
    ar_lm_topk: int = 64  # chỉ lấy top-k của LM vào micro (32k ứng viên/bước = quá chậm)
    ar_index_margin: float = 0.5   # biên vượt trội của token index khi đạt được
    ar_patience: int = 2           # dừng nếu N bước liên tiếp KHÔNG phủ thêm agenda (chống trôi
                                   # dài khi W_seq không bắc được cầu tới token index kế tiếp)

    # --- v4 P1: nơ-ron chủ đề Concept:entity (composition) ---
    # facts_about chấm điểm qua theme neuron: fact mà thực thể đứng ĐẦU câu (câu VỀ nó, topic-first)
    # thắng fact chỉ NHẮC nó giữa câu. compose path + crisp KNOWLEDGE (TK_022 best_match).
    compose_theme_weight: float = 1.0
    # TK_022: crisp retrieval qua theme_neuron khi ReadoutMode=KNOWLEDGE (WH là gì).
    # 0 = tắt; 1.0 = fact entity-initial thắng fact chỉ nhắc entity (w_theme<0.22 trên top-1).
    retrieval_theme_weight: float = 1.0
    # TK_022b (tác giả: "thiếu fact, chỉ câu đơn"): KNOWLEDGE mode bồi K fact RỜI về thực thể
    # (mỗi fact 1 câu, KHÔNG đại từ/bridge — TK_019). 0 = tắt (1 câu như cũ).
    knowledge_extra_facts: int = 2
    # A1 (TK_024): pool bồi fact mở sang GIA ĐÌNH thực thể (theme_neuron primary) — vua/tướng
    # nhà Lý lên được dù không có cụm "Nhà Lý"; MMR đa dạng khía cạnh. 0.5 = ngưỡng trùng MMR.
    knowledge_family_expand: bool = True
    knowledge_mmr_lambda: float = 0.5
    # Nhánh C (2026-07-08): chống bồi-drift trên query KHÔNG-phải-entity ("Người chữa bệnh"→
    # primary "bệnh" danh-từ-chung → bồi fact wiki chứa "bệnh" lạc đề). Gate: chỉ bồi khi primary
    # VIẾT HOA (cap_class≥1, thuộc-tính-token precompute TK_012 — Luật 0.0 sạch). gidf KHÔNG tách
    # được ("bệnh" 5.94 > "Lý" 5.84). True = chỉ bồi cho entity có tên riêng.
    knowledge_extra_require_cap_primary: bool = True
    # TK_025 P2: cộng-hưởng CO_ACTIVATION — demote fact ngữ-cảnh-khác chứa cùng token trần
    # ("Lý thuyết điện từ" vs triều Lý) theo KÍCH HOẠT (Luật 0), không rule/chuỗi/sense.
    # score *= floor + resonance(fact, trường co-act của query). cap = #lân cận/seed.
    # TK_027: CỔNG NEO THỰC-THỂ — nơ-ron Entity:title (hồ-sơ bài wiki, key CỤM không fold) neo từ
    # câu đáp chính → demote fact khác domain (lý-thuyết vật-lý). Net-DƯƠNG: sửa "Lý thuyết điện
    # từ" + Syn 50→67. Cần checkpoint có Entity: (network_v5_entity); checkpoint cũ = no-op.
    knowledge_entity_resonance: bool = True
    knowledge_entity_top: int = 6         # #thực-thể neo gộp thành trường
    knowledge_entity_gate_frac: float = 0.1  # loại fact dot ≤ frac×max (ngoài domain neo)
    # TK_027 bước 4: re-rank câu đáp CHÍNH khi candidate sát nút + query khớp Entity:title đầy đủ
    # ("Mây là gì" → Đường Ngô Mây 0.267 ≈ mây-def 0.240 → chọn fact trung-tâm Entity:Mây). Chỉ
    # đụng khi sát nút (margin) → ca winner rõ (guard curated) không đổi.
    primary_entity_rerank: bool = True
    primary_entity_rerank_margin: float = 0.05
    # TK_027 bước 5: cộng-hưởng-fallback khi title-match trượt. ĐO net-ÂM: KHÔNG sửa "Triều Lý"
    # (pool bị token-"Triều" homograph áp đảo mọi tầng) + làm HỎNG Para 88→62 (fallback bắn nhầm ca
    # Para title-trượt → nhiễu). Default OFF; cơ chế giữ làm nợ. Xem TK_027.
    knowledge_entity_resonance_fallback: bool = False
    # TK_029: thưởng fact chứa cặp token CÓ HƯỚNG kề của query ("Triều Lý"→cặp triều→Lý) — dùng
    # thứ tự đã học (W_seq/ISG left→right), dìm fact chỉ chứa token rời/ngược (Korea "Lý…Triều Tiên",
    # TQ "Triều Kim"). score *= 1 + bonus×(idf-cặp-khớp / max-idf-query). 0 = tắt.
    directed_bigram_bonus: float = 1.5   # TK_029b: gate idf → chỉ cặp đặc-trưng → bonus mạnh an toàn
    directed_bigram_min_idf: float = 2.0  # bỏ cặp hư-từ generic (Lý→đã) khỏi thưởng-hướng
    # TK_045 §12.124 (③): cặp-hướng theo KHOẢNG CÁCH d≤dmax (W_skip) thay chỉ kề d=1. dmax=1 = TK_029
    # cũ y nguyên. §12.119/§12.122 đo: match theo thứ tự thu hồi cha nguồn +31pp v12. λ giảm theo d.
    # P44: mặc định dmax=4 + accum=sum (nối match-theo-d vào A).
    directed_bigram_dmax: int = 4
    directed_bigram_lam: float = 0.5
    directed_bigram_accum: str = "sum"  # §12.124 ③ / P44: 'sum' phân biệt, 'max' = TK_029
    # TK_026: cộng-hưởng NGHĨA từ-điển (sense) — demote fact khớp một nghĩa từ-điển KHÁC với
    # nghĩa entity query. Nguồn định-nghĩa curated (sạch hơn co-act). Sửa "Mây" (khí-tượng vs núi).
    knowledge_sense_resonance: bool = False  # TK_026: đo net-ÂM — sửa "Mây" nhưng va-chạm fold
    # "Lý"→"lý" gọi nghĩa vật-lý/sinh-lý → "Nhà Lý" tệ hơn. default OFF. Xem TK_026.
    knowledge_sense_min_field_idf: float = 3.0
    knowledge_coact_gate_frac: float = 0.7  # loại fact resonance < frac×max (ngoài ngữ-cảnh query)
    # TK_025 §P2: co-act corpus (fallback, nhiễu hơn) — default OFF, xem TK_025 §P2.
    knowledge_coact_resonance: bool = False
    knowledge_coact_cap: int = 40
    knowledge_coact_min_field_idf: float = 3.5
    # TK_023: chọn fact bồi theo điểm kích thích lan xuôi ("stimulus") hay legacy facts_about.
    knowledge_fact_rank: str = "stimulus"
    # Cổng salient: stimulus từ token phân biệt nhất ≥ ratio × idf(salient). 0 = tắt.
    knowledge_min_salient_stimulus: float = 0.15
    # Cổng WH cho bồi fact: "có gì" có ~3 filler (< wh_min_filler_diversity=5) — hạ riêng
    # cho extra facts, không đụng best_match/T2 (đo: Mùa hè có gì miss fact).
    knowledge_extra_wh_min_filler: int = 3
    # Trụ 7 T2 (TK_004): sinh bài dài bám agenda t1 SFT — generate_essay (opt-in, không đụng
    # generate/composed). Khớp query→episode qua match_ids (instruction).
    essay_min_match: float = 0.15   # dưới ngưỡng → fallback composed. (TK_005: thang điểm
                                    # co lại khi thêm bigram vào mẫu số → 0.30 cũ ≈ 0.15 mới)
    essay_max_segments: int = 8     # closure trần: tối đa số câu/ý phát ra
    # TK_006: phát member cộng hưởng (tri thức cũ, w<1.0) sau khi phát xong provenance.
    essay_resonance_k: int = 2
    # review TK_006: thang điểm resonance trên bag-dài thấp (top thực tế ~0.05-0.18 trên _t2)
    # → 0.10 lọc gần hết; 0.07 = trên sàn build (0.05), giữ ứng viên khá. Sổ nợ nếu trộn rác.
    essay_resonance_min: float = 0.07
    # TK_008 (1) — gate chủ đề: member cộng hưởng phải chia ≥N bigram MỚI (ngoài câu hỏi) với
    # thân bài. Rút từ đọc tay TK_006: rác ("tập trung QUÂN ĐỘI") chỉ lặp cụm câu hỏi; mix tốt
    # ("sức khỏe"↔bài thể dục) chia nội dung mới. 0 = tắt (sổ nợ nếu đo xấu).
    essay_resonance_gate: int = 1
    # TK_009 (Trụ 4 pilot): sinh segment essay TOKEN-LEVEL bằng A3 (member = agenda) thay vì
    # dội verbatim. "verbatim" = phase-1; "ar" = pilot (đo A/B trước khi đổi default).
    # ĐO 07-02 vòng 4 (TK_009 §6): ar đạt điều kiện flip — vb_seg 100→34% THẬT (hết xáo
    # đầu-segment), cov/ord/match giữ 79/88/88. Giá: chậm hơn (mỗi segment 1 vòng A3) — sổ nợ.
    essay_segment_mode: str = "ar"
    # TK_010: đáp dài ≥ ngưỡng token → coi cha là t1 (đường thống nhất) dù chưa có fact_links
    essay_t1_min_answer: int = 20
    essay_ar_min_cov: float = 0.6   # A3 phủ agenda dưới ngưỡng → fallback verbatim cả bài
    # HỢP NHẤT sinh (generate): điểm vào duy nhất = plan (truy hồi) → A3 (sinh token). Lưới an
    # toàn: khi vòng A3 phủ agenda < essay_ar_min_cov, dội đáp truy hồi verbatim (giữ đúng fact
    # cho câu QA đã thuộc). Tắt (=False) để đo "thuần A3" không lưới. Chỉ áp cho plan crisp.
    ar_verbatim_fallback: bool = True


    # --- TK_017: position-targeted FRAME_BRACKET gate (Mode B) ---
    # 0 = tắt (ablate). Targeted Δ −0.94 bits @ w=0.5 là ORACLE (cận trên); runtime gate wired
    # đo THẬT Δ −0.052 (hơi dương, fire 93%) — TK_017 §11 F0.
    ar_frame_bracket_weight: float = 0.5
    ar_frame_bracket_floor: float = 0.2
    ar_frame_bracket_min_gap: int = 2
    # TK_020 (duyệt 07-06): KHOẢNG CÁCH là tham số cạnh ngang — boost closer × P_pair(gap=d)
    # (đo: 3.926/22.6% thắng broadcast 3.939/22.4%). False = broadcast PMI như cũ.
    ar_gap_prior: bool = True
    # TK_020 ctx-resonance (ý tác giả): cạnh có provenance trong CHA đang kích hoạt bởi prompt
    # → nhân (1+w) — "nếu trời mưa thì" ưu tiên cạnh học từ cha "nếu trời mưa thì…". 0 = tắt.
    ar_hlink_ctx_weight: float = 1.0
    # TK_020 §4.3 — OVERRIDE closer khi điều kiện KÉP: P(gap=d) ≥ min_p VÀ ctx-hit (provenance
    # trong cha đang kích hoạt). Fire hẹp + tự tin → được ép thắng như index-backbone. 0 = tắt.
    ar_gap_override_margin: float = 0.5
    ar_gap_override_min_p: float = 0.25
    # TK_036 — skip span token→cụm (gap 2–8): boost cụm đích xa khi w0=left + agenda có cụm right.
    ar_skip_span_weight: float = 0.35
    ar_skip_span_floor: float = 0.15
    ar_skip_gap_prior: bool = True
    ar_skip_ctx_weight: float = 0.75
    # Candidate-set AR: chỉ chấm điểm tập ứng viên thưa thay vì dense O(V) mỗi bước.
    ar_candidate_scoring: bool = True
    # TK_021 T1: ranking focus bare = fold_gidf/(1+pos)^alpha + demote FB-opener.
    # alpha=0.3 (đo t1_sim: lật đúng phở/mưa/Lý/Trần, giữ cấy/trị). <0 = hành vi cũ.
    focus_pos_alpha: float = 0.3
    # TK_021 T2: WH-detector thật — cụm anchor chỉ coi là câu hỏi khi filler đa dạng
    # (anchor_left_fillers ≥ ngưỡng). 0 = hành vi cũ (mọi cụm ISG).
    wh_min_filler_diversity: int = 5
    # TK_021 T3a: tie-break entity_cands theo vị-trí-topic (primary xuất hiện sớm trong cha).
    topic_first_tiebreak: bool = True
    topic_first_pos_alpha: float = 0.4
    # TK_021 T3b: đáp phải chứa primary focus — mọi khung QA, không chỉ hub "thành lập năm nào".
    slot_consistency_all_frames: bool = True
    # R1 (TK_017 §11 F5): plan FALLBACK (best_match trượt → facts_about) phải PHỦ focus phân
    # biệt nhất của query; không phủ → plan rỗng → W_seq + bracket-gate dẫn (không bịa fact
    # lạc đề kiểu "nếu trời mưa"→"Mặt trời cung cấp ánh sáng"). False = hành vi cũ.
    # R1-G1: prompt mở đầu bằng bracket-OPENER (cạnh FRAME_BRACKET mined) + không frame →
    # continuation (W_seq+bracket), bỏ retrieval crisp (score không tách drift/Syn — đo 07-05).
    ar_fallback_focus_gate: bool = True
    ar_opener_continuation: bool = True
    # R2 (TK_017 §13 / doc 03 A0): blend q·k directed (SVD PPMI-có-hướng của W_seq) vào kênh
    # micro của `_wseq_fallback`. Đo A0 07-05: ruler 4.028→3.926 (Δ−0.103 DƯƠNG); unseen Δ=0
    # (chưa chứng minh tổng-quát-hoá — sổ nợ). 0 = tắt.
    ar_qk_weight: float = 0.5
    # R2b: neo TOPIC cho continuation (doc 05 H_mid) — seeds = content token của prompt;
    # field spreading giữ chủ đề trong `_wseq_fallback` (không agenda). 0 = tắt.
    ar_continuation_topic_weight: float = 1.0
    ar_continuation_topic_floor: float = 0.15
    # R3b (TK_017 §14): continuation = pattern-completion trên POOL VĂN NÓI (priority=0.3,
    # two-tier — chỉ đọc ở opener-route). Mẫu khớp ≥ ngưỡng → phát đuôi sau điểm khớp.
    ar_conv_pool: bool = True
    conv_pool_min_score: float = 0.10  # score đã bị priority=0.3 scale (đo: mẫu tốt 0.12-0.17)
    # R4 (TK_017): mở pattern-completion cho prompt mở KHÔNG frame (essay tier + conv pool)
    # trước A3/W_seq; khớp suffix đủ dài để tránh bigram lẻ ("em" nuốt cả câu).
    ar_pattern_completion: bool = True
    pattern_completion_min_score: float = 0.12
    # R4b: đuôi mẫu pattern-completion → agenda A3 (không verbatim); fallback nếu cov thấp.
    ar_tail_as_agenda: bool = True
    ar_tail_verbatim_fallback: bool = True
    # Bracket-opener (R3b đo): giữ verbatim-tail — A3 trên suffix ngắn thường salad hơn copy.
    ar_tail_as_agenda_opener: bool = False
    # Two-tier sinh essay/QA: pool văn nói (priority≈0.3) KHÔNG cạnh tranh best_match.
    generation_min_priority: float = 0.5
    # TK_035: pool đoạn 2–4 câu (priority≈0.36) — continuation dài, giữ , . ! ?
    paragraph_pool_min_score: float = 0.08
    paragraph_prefer_long_tail: bool = True

    # --- Multi-zone (FACT head) + a_style (STYLE head) — opt-in, 2026-07-23 ---
    # Multizone: thay facts_about đơn bằng zone_retrieve (cap-span) → multi-segment agenda.
    # False = hành vi cũ (1 focus → 1 fact). Đo A/B trước khi bật default.
    ar_multizone: bool = False
    ar_multizone_cos_min: float = 0.4   # RUNBOOK v4: hạ 0.5→0.4 cho Bạch Đằng
    ar_multizone_gidf_min: float = 3.0
    ar_multizone_gidf_solo: float = 6.0
    # TK_045 §12.46: nới vùng sang TRÁI khi PMI(prev, thực-thể) ≥ ngưỡng (không gộp
    # hoa/thường — §12.45③). Vá ca vùng `Lý` mất `nhà` ⇒ facts_about trả gia phả.
    # 0 = tắt. MẶC ĐỊNH OFF chờ đo.
    ar_multizone_left_pmi_min: float = 0.0
    ar_multizone_beta: float = 5.0
    ar_multizone_facts_top: int = 1     # 1 fact/zone → segment (tránh salad)
    ar_multizone_cover_min: float = 0.5  # fact phải phủ ≥ tỉ lệ token zone (đa-token); 0=tắt
    # a_style: CO_ACT neighbors của token HOW (cap=0) trong prompt → Hadamard gate.
    # 0 = tắt (no-op). Skeleton — chưa Style hub; đo probe_style_heads trước khi tăng.
    ar_style_weight: float = 0.0
    ar_style_floor: float = 0.2
    ar_style_gidf_min: float = 4.5
    ar_style_c_min: float = 2.0
    ar_style_top_k: int = 48
    ar_style_max_seeds: int = 2
    # Chèn marker style tại ranh giới (sau khi Hadamard thường bị agenda át — đo 2026-07-23).
    # True: tại _at_boundary, cộng mạnh top style token + tạm bỏ index-backbone 1 bước.
    ar_style_boundary_inject: bool = False
    ar_style_boundary_boost: float = 25.0
    ar_style_boundary_max: int = 2  # tối đa số lần chèn / lượt sinh
    # Inventory lọc + phrase inject (thay raw top W×IDF → tránh `cốt`).
    ar_style_use_inventory: bool = True
    ar_style_prefer_phrase: bool = True
    ar_style_min_boundary: float = 0.5
    # Ưu tiên n-gram thân styled_sft* (label-conditioned) trước graph inventory.
    ar_style_corpus_inventory: bool = True
    # Ưu tiên cụm discourse (điệp âm / W_seq khóa) hơn unigram topic.
    ar_style_discourse_ngram: bool = True
    # Mixture: cửa cosine — thử MỌI bước decode (không chỉ sau dấu câu).
    # score = cos(w0, style) hoặc 0.5*(cos(w0,style)+cos(style,agenda_next)) nếu còn agenda.
    ar_style_mixture: bool = False
    ar_style_cos_thresh: float = 0.12  # sàn E-cosine; đo/calib bằng probe_style_mixture
    ar_style_mixture_cooldown: int = 2  # tối thiểu #token giữa 2 lần chèn style
    # Style chỉ thắng nếu bridge ≥ cos(left,agenda)+margin (tránh xé cụm đang mượt).
    ar_style_cos_margin: float = 0.05
    # True: chỉ xen style tại ranh giới câu HOẶC seam giữa 2 segment MZ — giữ fact liền.
    ar_style_fact_safe: bool = True
    # Setup→Pivot→Payoff: thêm segment frame + mẩu thân hài (lệch entity trên E).
    ar_style_payoff: bool = False
    ar_style_payoff_max_tokens: int = 28
    ar_style_payoff_setup_segs: int = 1  # số fact segment trước pivot
    ar_style_payoff_setup_max_tokens: int = 48  # cắt câu setup đầu (chừa chỗ joke)
    ar_style_payoff_keep_tail: bool = False  # True = giữ fact sau payoff (dễ đầy max_len)
    # True = copy cứng pivot/payoff (legacy). False = soft W_seq/ctx decode.
    ar_style_payoff_hard_copy: bool = False
    # Khóa cụm pivot ngắn liền sau fact (`. thế nhưng`) — soft chỉ thân payoff.
    ar_style_payoff_lock_pivot: bool = True
    # Soft decode: peaky (Việt→Nam) vs hub (nhưng…) trên hàng W_seq.
    ar_style_soft_peak_frac: float = 0.35
    ar_style_soft_tie_eps: float = 0.08
    ar_style_soft_hub_wseq: float = 0.35  # scale W_seq leaders khi hub
    ar_style_soft_ctx: int = 6
    ar_style_soft_ctx_weight: float = 0.6
    ar_style_soft_agenda_weight: float = 0.75
    ar_style_soft_index_margin: float = 0.05  # peaky: backbone chỉ khi agenda cạnh W_seq
    ar_style_soft_slip: bool = False  # True = cho trượt 1–2 agenda (dễ salad)
    ar_style_soft_fluency_weight: float = 0.55  # nhân scores với cos(w0, cand)
    # Số token đầu thân payoff copy cứng (sau pivot) — giảm salad ngay seam.
    ar_style_soft_lead_lock: int = 8

    # --- Spectral stack 4 tầng (Attn→Amp→top-n), residual SVD PPMI(W_seq) — opt-in ---
    # Không gán nghề tầng (cú pháp/style/fact); chuyên hóa đo sau bằng probe.
    # Khi bật: blend a_out vào AR; TẮT hard-copy / pivot_lock / lead_lock trên path đó.
    ar_spectral_stack: bool = False
    ar_spectral_stack_weight: float = 0.55
    ar_spectral_stack_floor: float = 0.15
    ar_spectral_stack_d: int = 32  # nửa phổ / tầng (Wa + Wffn = 2d)
    # Forward: ngữ cảnh có vị trí (mặc định positional).
    ar_spectral_query_mode: str = "positional"  # "positional" | "w0" | "history"
    ar_spectral_s_wseq_k: int = 64  # top-k W_seq vào cửa sổ S (gộp multi-token)
    ar_spectral_combine: str = "residual"  # "residual" | "last" | "hadamard"
    ar_spectral_succ_inject: bool = False  # True = legacy bơm Wa_L0 full-vocab
    ar_spectral_use_amp: bool = True
    ar_spectral_attn_temp: float = 0.45  # <1 → softmax sắc hơn
    ar_spectral_wseq_attn_beta: float = 2.0  # cộng β·log1p(W_seq) vào logits attn
    # Continuation / stack: chồng chập history có decay vị trí.
    ar_ctx_positional: bool = True
    ar_ctx_lambda: float = 0.85
    ar_ctx_len: int = 12
    ar_ctx_content_gate: bool = True  # hạ cand kém liên quan content prompt (cos E)
    ar_ctx_content_gate_floor: float = 0.12
    # N-gram / token xa: VarLM (1..4) + order-2 cha + neo content xa.
    ar_ngram_weight: float = 1.15  # VarOrderLM → micro (cần sidecar .varlm.pkl)
    ar_order2_cont_weight: float = 0.55  # trigram ParentStore trên continuation
    ar_distant_content_weight: float = 0.4  # neo content xa (mode bên dưới)
    # Trọng số ma trận sequential: raw W_seq hub-nặng (Ċ→Giải); PPMI + skip punct tốt hơn.
    ar_wseq_ppmi: bool = True  # positional/local dùng directed PPMI(W_seq) thay count thô
    ar_wseq_skip_structural: bool = True  # không lấy successor từ Ċ / punct
    ar_distant_mode: str = "ppmi"  # "ppmi" | "coact" | "e" | "wseq"
    # W_know / Oja (legacy channel) — tắt khi ar_matrix_core=True.
    ar_know_weight: float = 0.0  # span-next thủ công (deprecated)
    ar_know_bi_weight: float = 0.0
    ar_oja_compete: float = 0.55
    ar_gidf_target_weight: float = 0.0
    # Matrix core kiểu LLM: chỉ logits = Σ β·(matvec / q·k / stack).
    ar_matrix_core: bool = True
    ar_matrix_parent_weight: float = 0.9  # bag emit cùng cha (thấp — tránh túi từ chung)
    # TK_045 §12.2: hard-AND gộp lớp hoa/thường (`Nhà`≡`Ġnhà`) — cha đúng viết thường không bị
    # nhân 0. False = hành vi cũ (ablate).
    ar_matrix_hard_and_casefold: bool = True
    # TK_045 §12.2c vá(3): path khớp content (bỏ hư-từ, fold hoa/thường).
    # A/B v12 (M4): soft=True cứu hung (succ 0→387) NHƯNG regress số Lý/Bạch.
    # §12.13: knob này từng bị commit 6753a9a (revert copy-số v1) XOÁ khỏi HCAConfig ⇒
    # `getattr(config, ..., True)` rơi về True ⇒ soft BẬT ngầm trong MỌI lần đo từ đó,
    # kể cả A/B v12 của M4. Khôi phục default False + sửa fallback getattr về False.
    # §12.13 đặt True: sau khi chữ-số hết bị coi là hư-từ, soft KHÔNG còn hại số
    # (mini: bach/ly/bach2 sạch) mà vẫn cứu `hung`→Lam Sơn. Cũng là hành vi ĐANG chạy
    # trên v12 từ 6753a9a (do getattr fallback), nên log v12 cũ vẫn so sánh được.
    ar_matrix_path_soft: bool = True
    # TK_045 §12.5 v2: chữ-số kế → CHÉP cụm số từ cha ĐANG KHOÁ (lock_pi).
    # MẶC ĐỊNH OFF: dev KHÔNG verify được end-to-end (mini bị OOM-kill) — v1 từng regress
    # ('18110') nên KHÔNG bật mù. M4 A/B rồi mới bật (RUNBOOK §12.5).
    ar_matrix_copy_number: bool = True  # §12.13f v12 A/B: ly→1010, bach→1288; giữ OFF nếu regress
    # TK_045 §12.14 chép-lùi: khi path KHÔNG khoá được (số đứng trước phần khớp, prompt đảo
    # trật tự) mà `act` vẫn trỏ đúng cha → chép cụm số của cha đang kích. Cổng an toàn =
    # phủ 100% content history + emit có ĐÚNG 1 cụm số (v1 chép nhầm `18110` vì thiếu cổng này).
    # Chỉ chạy khi `ar_matrix_copy_number` bật. False = ablate.
    ar_matrix_copy_number_from_act: bool = True
    # TK_045 §12.27: cha QA đang kích khớp câu hỏi ⇒ chép span ĐÁP. Path không làm
    # được (emit cha QA = answer_ids, không giao với câu hỏi ⇒ suffix-match luôn trượt).
    # Cổng: history phủ ≥80% match_ids của cha. MẶC ĐỊNH OFF — chờ đo (§12.27).
    # TK_045 §12.29: khi handoff episode cạn, lấy span fact của ZONE kế (multizone) —
    # giữ matrix core, chỉ thêm NGUỒN span. Cho ca GHÉP (T3). MẶC ĐỊNH OFF — chờ đo.
    ar_matrix_zone_handoff: bool = False
    ar_matrix_qa_answer_copy: bool = True  # §12.42: T4 9/10 ≥ cổng 6/10; T1/T2/T5/T7 ≡ OFF
    # §12.28: cho phép chép cả cha CÂU TRẦN (không chỉ cha QA) khi nó phủ nội dung
    # câu hỏi — tri thức thật thường nằm ở câu trần. False = chỉ chép cha QA.
    ar_matrix_qa_answer_statement: bool = True
    # TK_045 §12.15/§12.16 CỔNG TIỀN-ĐỀ: giao các cha của `last_k` token gidf-cao CUỐI
    # prompt = rỗng ⇒ trí nhớ chưa từng thấy tổ hợp ⇒ DỪNG (thà im còn hơn bịa năm sai).
    #
    # MẶC ĐỊNH **OFF** — §12.20. Số v12 của §12.19 đều tốt (THƯỚC 0% · qa_supervised 0% ·
    # BỊA 100% @ last_k=0) nhưng CẢ BỐN nhóm đó đều là prompt ĐƠN-FACT, nên mù với ca GHÉP:
    # đo lại thì 5/5 prompt so-sánh/đa-thực-thể bị bịt miệng ("so sánh nhà Lý và nhà Trần",
    # "Cả Hoa Lư và Thăng Long đều từng là"…). Luật "không cha nào chứa ĐỦ token đặc-hiệu ⇒ im"
    # chính là "từ chối mọi thứ cần hơn MỘT trí nhớ" = tắt luôn composition (TK_024,
    # probe_retrieval_compose). Bịa và ghép KHÔNG tách được bằng tín hiệu này.
    # Bật khi chạy chế độ QA-đơn-fact và chấp nhận mất ghép.
    ar_matrix_premise_guard: bool = False
    # 0 = cả prompt. §12.19 v12: k=0 và k=10 ĐẠT cổng đơn-fact; k=6 chỉ bắt 62.5% bịa.
    ar_matrix_premise_last_k: int = 0
    # TK_045 §12.13: path trắng tay → quét lại cha với sàn act = 0 (cha act đỉnh có thể
    # KHÔNG path-match nổi, sàn tương-đối của nó giết cha đúng). False = ablate.
    # TK_045 §12.28: token có (số cha PHÁT)/(số cha KHỚP) < ngưỡng = chỉ mang HÌNH THỨC
    # câu hỏi (đo mini). Trên v12 er(WH)≈1 (§12.42) — WH-slot positional mới là tín hiệu chính.
    # 0.9 = bật kèm qa_answer_copy (§12.42 cổng đạt).
    ar_matrix_and_form_min: float = 0.9
    # TK_045 §12.43/§12.48: WH-slot HỌC từ cha QA ∪ luật vị trí (+ phủ quyết ask_form).
    # §12.47: learned-only (thiếu `ai`) hỏng T4 7→5. §12.48 hợp+veto: T4 ≡ luật vị trí 7/10
    # trên v12 (§12.49) ⇒ bật default.
    ar_matrix_wh_learned: bool = True
    # θ của M4 (§12.43 M4c): `n_Q/(n_Q+n_A) ≥ θ` ⇒ WH-slot. 0.8 CHẶT hơn ngưỡng
    # `n_A/n_Q < 0.9` ban đầu của tôi (≈0.53) ⇒ ít bắn nhầm thực thể hơn.
    ar_matrix_qa_ask_form_min: float = 0.8
    # TK_045 §12.48: `ask_form` < ngưỡng ⇒ BÁC đề cử của luật vị trí §12.42 (token đó rõ
    # ràng là NỘI DUNG: Ngô .40 · Lê .25 · Lý .23, còn `ai` .53 không bị bác).
    # §12.49: đạt cổng T4 ≥7/10 trên v12 ⇒ default 0.45.
    ar_matrix_wh_veto_max: float = 0.45
    ar_matrix_path_floor_retry: bool = True
    ar_matrix_parent_gamma: float = 1.5
    ar_matrix_parent_succ_weight: float = 5.0  # successor(w0) trong cha đang kích
    ar_matrix_parent_succ_boost: float = 3.0
    ar_matrix_parent_gate: float = 0.75
    ar_matrix_parent_ctx_len: int = 0  # 0 = full history (không cửa sổ)
    ar_matrix_parent_topk: int = 12  # sparsify act → ít lẫn thời đại
    ar_matrix_parent_gidf_hi: float = 4.0
    ar_matrix_parent_path_weight: float = 4.0  # khớp suffix emit cha (anti-drift)
    ar_matrix_know_weight: float = 1.2  # hạ — fluency toàn cục
    ar_matrix_know_lr_weight: float = 0.5
    # §12.41 M4: eval_noqk_v12 ≡ eval_v12_OFF_v2 (0 field-diff, gồm T1/T7) ⇒ kênh chết.
    ar_matrix_qk_weight: float = 0.0
    ar_matrix_stack_weight: float = 0.35
    ar_matrix_readout_weight: float = 0.0
    ar_matrix_eos_weight: float = 1.2  # ctx·end_after → logits[EOS] (cạnh tranh argmax)
    # W_skip: mỗi token history đề xuất next ở đúng d, tích mềm (+ thưởng hard∩). Mặc định OFF.
    ar_matrix_w_skip_weight: float = 0.0
    ar_matrix_w_skip_d_max: int = 12
    ar_matrix_w_skip_min_count: int = 2
    ar_matrix_w_skip_topk: int = 64
    ar_matrix_w_skip_ctx_max: int = 2  # probe mini: ctx=2 soft10|multi tốt nhất (0.579)
    ar_matrix_w_skip_hard_bonus: float = 2.0
    # True: khi hard∩ ≠ ∅ chỉ bơm điểm cho token trong ∩ (đúng ý giao); soft chỉ khi ∩ rỗng.
    ar_matrix_w_skip_hard_gate: bool = False
    ar_matrix_w_skip_gidf_min: float = 4.0  # lọc hub khỏi đề xuất (cùng ngưỡng cột lạc)
    # §12.106 — kênh PHIẾU W_skip cho GIẢI MÃ (đặc tả tác giả §12.105). Khác đám knob trên: cộng
    # mềm cả cửa sổ, chuỗi ĐẦY ĐỦ (không content_only), không cổng cứng. 0 = tắt.
    # Hiệu chỉnh mini (§12.105d): λ=0.5 · topk=12 · nhiệt 0.3–0.4 cho văn "bắt đầu đúng".
    # LƯU Ý: bật sampling cần `hybrid_temperature > 0` VÀ `ar_sharp_gate=False` — cổng entropy
    # (mặc định True, ngưỡng 2.2) quay về argmax đúng lúc phân phối phiếu phẳng ⇒ vô hiệu sampling.
    ar_matrix_w_skip_decode_weight: float = 0.0
    ar_matrix_w_skip_decode_lam: float = 0.5
    ar_matrix_w_skip_decode_topk: int = 12
    ar_matrix_w_skip_decode_dmax: int = 16
    # §12.136 — phiếu W_skip primary: 0 = full hàng (không cắt topk). topk=12 cắt
    # Tổ→quyết rank67, Thái→quyết rank123, Lý→quyết rank73, vua→quyết rank202.
    ar_w_skip_vote_topk: int = 0
    # §12.139 — `prod`: HỢP đề xuất (không giao) + soft-prod × số phiếu rồi softmax.
    # `sum`: cộng phiếu λ^(d-1) (cũ §12.106).
    ar_w_skip_vote_agg: str = "prod"
    # Cửa sổ proposer cho Hadamard; 0 = dùng hết d_max (sau lọc nội dung).
    ar_w_skip_prod_window: int = 0
    # §12.140 — bão hoà hub (mọi token). Oja-L2 hàng TẮT (phá soft-prod).
    # deg ra TẮT mặc định (0.35+indeg làm xáo giữa câu); indeg=1.0 hạ hub đích.
    ar_w_skip_oja_row: bool = False
    ar_w_skip_oja_degree_beta: float = 0.0
    ar_w_skip_oja_compete: float = 0.0
    ar_w_skip_oja_indeg_beta: float = 1.0

    # Sàn gate flu cho token đã được ≥2 proposer (multi-d) — tránh Hadamard d=1 diệt
    # quyết (rank67) trong khi phiếu multi-d đã cộng. 1.0 = không減衰 multi-d.
    # Với agg=prod: token sống qua giao đã được bảo vệ (multi = score>0).
    ar_w_skip_multivote_gate_floor: float = 1.0
    # TK_045 §12.131/133 — W_skip làm đường sinh chính + ưu tiên agenda MZ.
    ar_w_skip_primary: bool = True
    # Opt-in sinh (mặc định W_skip): frame_da_khe=P112 · compose_2cha=TK_051 P203 ·
    # analogy_la=TK_046 P98G (điền X là ?).
    generation_mode: str = "w_skip"  # w_skip | frame_da_khe | compose_2cha | analogy_la
    # >0 = softmax + multinomial theo xác suất; 0 = argmax top-1 (đã thử — hub lặp)
    ar_w_skip_primary_T: float = 0.5
    # §12.132 chống salad khi primary (giữ; MZ agenda §12.133 là đòn chính chống trôi)
    ar_w_skip_same_sent_beta: float = 1.0
    ar_w_skip_fluency_weight: float = 1.0  # 0 = tắt cổng d=1
    # 0 = cứng: token không có W_skip[d=1] từ w0 bị diệt (tránh Thánh→Năm qua floor lách)
    ar_w_skip_fluency_floor: float = 0.0
    ar_w_skip_fluency_topk: int = 64
    # True: khớp agenda STRICT (không skip token lệch) — tránh Hoa/về/Lư đảo → boost nhầm Đại
    ar_w_skip_agenda_strict: bool = True
    # §12.133 — ưu tiên token thuộc agenda multizone (zone_retrieve → flat_agenda).
    # True: ép ar_multizone khi dựng plan cho W_skip primary.
    ar_w_skip_use_mz_agenda: bool = True
    # Hệ số × max(vote) cộng vào token agenda chưa phủ (cửa sổ thứ tự).
    ar_w_skip_agenda_boost: float = 2.5
    # §112 — 4 → 3. Agenda LÀ câu nguồn đúng thứ tự (§111.1); cửa sổ cho phép
    # mỗi bước chọn bất kỳ trong N mục kế tiếp, và N=4 đủ rộng để `hoạt động rất
    # rộng` phát thành `hoạt · rất · động`.
    #
    # §113 ĐÍNH CHÍNH §112: §112 chọn 3 theo TRUNG THÀNH, nhưng tách theo từng
    # prompt cho thấy thước đó MÙ VỚI THỨ TỰ — nó đo TẬP TỪ:
    #     hổ  w=3 «hoạt RẤT ĐỘNG thể…»          → 100%
    #         w=1 «HOẠT ĐỘNG RẤT RỘNG, VÀ CÁC…» → 100%   (chấm NGANG NHAU)
    #     lúa w=3 rối hoàn toàn                  → 100%
    #         w=1 gần đúng, lệch ở đuôi          →  85%   (thước PHẠT câu đúng)
    # 8/10 prompt chênh đúng 0,0; chênh tổng 4,5 điểm là hiện vật TỈ LỆ GỘP
    # (prompt `mặt trời` sinh 23 từ ở 21,7% kéo pool xuống).
    # Chọn theo hai thước CÓ THỂ thấy thứ tự: TRẬT TỰ 89,8 → 93,8 và LẶP
    # 14,5 → 10,8, cả hai đều nghiêng về w=1.
    ar_w_skip_agenda_order_window: int = 1
    ar_w_skip_agenda_order_decay: float = 0.6
    # §12.135 — khi còn pending: nhân điểm token NGOÀI cửa sổ pending (trừ EOS/punct).
    # Vá Tổ→chức (W_skip d=1 flu=123 «Tổ chức») cạnh tranh quyết (chỉ +ag) → sampling trúng chức.
    ar_w_skip_nonagenda_scale: float = 0.05
    # §12.144/145 — chỉ consume CẠNH (src,d)→nxt; KHÔNG chặn token
    # (chặn token giết `kinh→đô` dù cạnh đó chưa dùng — đo 2026-08-01).
    # §83 — tín hiệu thứ tự BẬC-2 của cha (`ParentStore.seq_next`) trên đường
    # production. `W_seq` là bigram bậc-1 ⇒ word-salad; kênh này chỉ XẾP LẠI
    # ứng viên đang sống, không hồi sinh đích đã consume. 0 = tắt.
    # §86 — nở truy vấn theo hoa/thường khi truy hồi cha. 0 = tắt.
    retrieval_gop_hoa_thuong: bool = False
    ar_w_skip_order2_weight: float = 0.0
    # §84 — bậc-2 lấy từ CHA ĐANG HOẠT thay vì gộp toàn cục. Bản toàn cục (§83)
    # bị chuỗi phổ thông thống trị ⇒ đổi tri thức lấy trôi chảy.
    ar_w_skip_order2_cha_hoat: bool = False
    ar_w_skip_order2_top_n: int = 20
    ar_w_skip_consume_edges: bool = True
    ar_w_skip_consume_tokens: bool = False
    # Nhân trọng số cạnh đã consume (0 = tắt hẳn; >0 = giảm điểm mềm).
    ar_w_skip_consume_edge_scale: float = 0.0
    # Chiều ngược: gần luôn; xa trừ argmax d=1 (giữ kinh→đô). 0 = tắt ngược.
    ar_w_skip_consume_reverse_window: int = 4
    # Hồi cạnh đã consume sau N bước decode (0 = không hồi); hold nếu đích còn gần.
    ar_w_skip_consume_recover_steps: int = 8
    # §12.145 — phạt mềm token đã có trong cửa sổ gần (túi xáo, không cấm vĩnh viễn).
    ar_w_skip_local_repeat_window: int = 8
    ar_w_skip_local_repeat_scale: float = 0.15
    ar_w_skip_local_repeat_min_count: int = 1
    # Phạt thêm token trong near_k nội dung gần nhất (chống chu kỳ liền).
    ar_w_skip_local_repeat_near_k: int = 3
    ar_w_skip_local_repeat_near_scale: float = 0.08
    # Khi còn outsider score>0: nhân thêm lên token trong túi gần.
    ar_w_skip_local_repeat_outsider_prefer: float = 0.05
    # Chỉ dùng khi consume_tokens=True (mặc định tắt).
    ar_w_skip_consume_gidf_min: float = 4.0
    # §12.141/P42 — EOS từ end_after vào pool primary (0 = giữ median cũ).
    ar_w_skip_eos_end_after_beta: float = 2.0
    # §12.141/P42 — sau biên: boost W_agenda mở câu sau (0 = tắt).
    ar_w_skip_sent_open_beta: float = 1.0
    # Boost mở câu từ W_skip pre-boundary (Long→Đây) — ưu tiên hơn W_agenda nhiễu.
    ar_w_skip_sent_open_wskip_beta: float = 2.5
    # Sau biên: nhân điểm token thuộc câu đã đóng (1 = tắt hạ).
    ar_w_skip_post_boundary_dampen: float = 0.08
    # Khi đã ≥1 biên trong đoạn sinh: nhân thêm điểm EOS.
    ar_w_skip_episode_eos_boost: float = 2.2
    # Đứng trên biên sau đoạn đủ dài: nhân EOS × median (tránh mở câu 2 khi cạnh yếu).
    ar_w_skip_boundary_eos_beta: float = 1.15
    # §151 — DỪNG THEO ĐỘ TIN: dừng khi xác suất đỉnh của phân bố điểm tụt dưới
    # ngưỡng này. Đo được: sinh bình thường p_top1 ∈ [0,38;0,96], hết nguyên
    # liệu [0,029;0,082] — khoảng trống 4,6×. 0 = tắt.
    #
    # §153 — HẠ 0,15 → 0,05. Thước KHÔNG phân định được giữa hai mức: TRUNG
    # THÀNH là TỈ LỆ nên cắt nhiều hơn thì tự tăng (hiện vật độ dài §113), và
    # `+1,5` của 0,10/0,15 chỉ phản ánh cắt sâu hơn. Phân định bằng ca đọc được:
    #   0,05  «muối là ▸ sản phẩm axit VÀ BAZƠ.»  ĐÚNG
    #   0,15  «muối là ▸ sản phẩm axit»           SAI — muối là sản phẩm của
    #                                              axit VÀ bazơ, cắt vế thành SAI
    # Chọn 0,05: «sai mà đọc như hoàn chỉnh» nguy hiểm hơn «hỏng mà trông hỏng»
    # (cùng dạng `bộ binh`↔`kị binh` tác giả đã chỉ ra). 0,05 vẫn giữ phần lớn
    # lợi ích cấu trúc: TRẬT TỰ 97,2 → 99,2 và phương sai ±1,8 → ±0,3.
    ar_w_skip_do_tin_min: float = 0.05
    # §165 — tiêm cạnh REINFORCED (loại suy đã xác nhận, TK_054) vào phiếu sinh,
    # scale theo rv. 0 = tắt. Chỉ có tác dụng trên checkpoint có cạnh REINFORCED.
    ar_w_skip_reinforced_beta: float = 0.4
    # §171 — chỉ hồi sinh đề cử loại suy khi p_top1 < ngưỡng này (mạng KHÔNG
    # chắc). Cao = hồi sinh nhiều hơn. Trên checkpoint không cạnh REINFORCED: trơ.
    ar_w_skip_reinforced_conf: float = 0.2
    # Số token nội dung trước biên giữ lại khi vote đứng trên biên.
    ar_w_skip_vote_pre_boundary_k: int = 2
    # §12.146/P43 — dual-channel: surface_vote ⊙ content_gate (agenda+entity+prompt).
    ar_w_skip_content_gate: bool = True
    ar_w_skip_content_gate_floor: float = 0.15
    # TK_045 §12.127/P33 — order→act trên path đo (A.T@ctx). Default OFF; P31 chứng minh lãi
    # trên rank_matches. Multiplier sau bag act, trước hard-AND. Không rebuild sparse A.
    ar_matrix_order_act: bool = False
    ar_matrix_order_dmax: int = 4
    ar_matrix_order_lam: float = 0.5
    ar_matrix_order_beta: float = 1.5
    ar_matrix_order_min_idf: float = 2.0
    ar_matrix_order_accum: str = "sum"  # 'sum' | 'max' — khớp P31 B
    ar_know_qk_dim: int = 64
    # TK_045 P7 / P0.2 — L2R/R2L+d trong câu + SENT_SUCC agenda
    ar_sent_intra: bool = True
    ar_sent_lr_lam: float = 0.85  # f(d)=λ^{d-1}
    ar_sent_lr_d_max: int = 12
    ar_sent_lr_content_only: bool = True
    ar_sent_lr_use_gidf: bool = True
    ar_sent_lr_write_adj: bool = False  # ghi EdgeKind SENT_* vào adjacency (tuỳ)
    ar_sent_r2l_M: int = 3  # số token đuôi đọc R2L
    ar_sent_r2l_alpha: float = 0.85
    ar_sent_l2r_weight: float = 2.0  # cont ⊙ emit → logits
    ar_sent_l2r_early: int = 4
    ar_sent_l2r_parent_topk: int = 8
    ar_sent_l2r_parent_mask: bool = True  # bắt buộc cổng cha
    # Agenda trong logits = 0: đa câu chỉ qua handoff khi EOS thắng (P0.3).
    # Bơm W_agenda vào logits giữa câu → nhảy câu 2 khi cont/mask rỗng ("đượcNgô…").
    ar_sent_agenda_weight: float = 0.0
    ar_sent_agenda_handoff: bool = True  # P0.3: EOS thắng nhưng còn câu sau → tiếp
    ar_sent_agenda_handoff_w_agenda: bool = False  # fallback W_agenda (dễ hub; mặc định tắt)
    ar_sent_agenda_handoff_span: int = 28  # số token tối đa chèn từ emit câu kế
    ar_sent_agenda_handoff_max: int = 1
    # §12.74: cổng phủ cho span nối tiếp — đòi span chia >= N fold nội dung (gidf>=4) với CÂU
    # TRƯỚC. 0 = tắt. Hiệu chỉnh trên mini: N=3 giữ 39,7% ca đúng, lọt 3,4% ca trôi.
    ar_sent_agenda_handoff_cover_min: int = 0  # 1 handoff = 2 câu (tránh dính câu 3 lỗi)
    ar_sent_agenda_handoff_ratio: float = 0.12  # min concentration W_agenda (không so eos_logit)
    ar_sent_agenda_handoff_min: float = 0.15  # mass tuyệt đối tối thiểu
    # P7.2c — Attn từ L2R, enrich kiểu FFN từ R2L
    ar_matrix_qk_from_l2r: bool = True
    ar_matrix_r2l_ffn_weight: float = 0.35
    ar_sent_qk_dim: int = 64
    ar_sent_qk_nnz_cap: int = 2_000_000  # vượt + thiếu sidecar → fallback runtime
    # P5 — phân bố sắc trước khi sample
    ar_sharp_gate: bool = True
    ar_sharp_entropy_max: float = 2.2  # H(top-k) nats; cao → argmax / hạ T
    ar_sharp_temp_scale: float = 0.45  # nhân T khi phẳng (nếu không argmax)
    ar_sharp_fallback_argmax: bool = True
    # P6.A / P7.5 — h dài từ hist + anchors R2L + cont L2R
    ar_matrix_h_weight: float = 0.45
    ar_matrix_h_anchor_k: int = 8
    ar_matrix_h_cont_k: int = 8
    # Bỏ retrieval/agenda (MZ/fact/pattern) → AR thuần micro (+ stack nếu bật). Đo stack “có ăn”.
    # True = bỏ MZ/fact/pattern retrieval → matrix/_wseq thuần.
    # Không tắt ar_sent_agenda_handoff (đa câu qua episode ở EOS).
    ar_no_agenda: bool = False


def hca_config_for_grade(grade: int = 3) -> HCAConfig:
    """K_max theo lớp — grade 3 dùng 24 token (câu QA / ghép dài hơn)."""
    k_by_grade = {1: 8, 2: 12, 3: 24, 4: 24, 5: 32, 6: 32, 7: 32}
    return HCAConfig(k_max=k_by_grade.get(grade, 16))
