# TK_108 — aivie viết code và dùng công cụ (code + tool use)

*Viết 13/10/2026. Nối TK_107 (thân 2,5B / 8K, mốc 20 tỷ token, P3 đang chạy).*

> **Mốc mới nhất (22/10): cx3b** — expert code lớn 1,21B: tool-use 55%, HumanEval 13,4%, MBPP 23,3%. Trước TK_108 các số này là 0 · 6,1 · 5,1.
> Kiến trúc xem `TK_108_KIEN_TRUC_CX3B.md`; quá trình xem §9–§13.

## 0. Mục tiêu và mốc hiện tại

**Mục tiêu:** aivie viết code đúng, **tự chạy – đọc lỗi – sửa**, và **làm việc trong thư mục dự án bằng công cụ** (đọc/ghi file, chạy
test, tìm kiếm), giải thích code bằng tiếng Việt chuẩn thuật ngữ.

**Mốc đo trên Mac (13/10, bộ đo `scripts/code_eval.py`, 40 bài HumanEval + 20 bài tool-use):**

| Model | HumanEval một lần | Sau ≤2 vòng tự sửa | Tool-use (20 bài) |
|---|---|---|---|
| 1.3B v5 (có expert code) | 17,5% | 17,5% | 0% — 86% lượt không gọi công cụ |
| 2,5B SFT bước 2.500 (chưa expert) | 0% — chép lại docstring rồi giải thích, không viết thân hàm | 5% | đang đo |

**Mục tiêu đo được (ước lượng, sẽ chỉnh khi có số thật):** HumanEval ≥ 30% một lần, ≥ 40% sau tự sửa; tool-use ≥ 50% bài thành công;
MBPP ≥ 35%. Model 2,5B có trần: các model code nhỏ tốt nhất được train hàng nghìn tỷ token, ta có 20 tỷ → đòn bẩy là **dữ liệu chất
lượng cao có kiểm chứng + RL bằng test + vòng lặp công cụ**, không phải lượng token.

## 1. Đã có (C0)

- `scripts/code_agent.py` — khung chạy có công cụ: **sandbox** (macOS `sandbox-exec` / Linux `bwrap`: cấm mạng, cấm ghi ngoài thư
  mục làm việc, khoá đọc `~/.ssh`/token HF; giới hạn CPU/bộ nhớ/thời gian/đầu ra), chế độ **debug** (viết → chạy test công khai →
  đưa lỗi lại → sửa) và chế độ **tool** (giao thức `<tool>{"name","args"}</tool>` → `<obs>…</obs>`, xong `<done>…</done>`),
  **ghi quỹ đạo** từng lượt. **Bộ công cụ kiểu Claude Code / Cursor** (nâng cấp 13/10):

  | Công cụ | Tham số | Hành vi |
  |---|---|---|
  | `list_dir` | `path` | liệt kê thư mục |
  | `glob` | `pattern` | tìm file theo mẫu tên (`**/*.py`) |
  | `grep` | `pattern` (regex), `path`, `glob`, `context`, `ignore_case` | tìm trong file, có dòng ngữ cảnh, tối đa 80 dòng khớp |
  | `search_code` | `query` (lời) | tìm hàm/lớp liên quan (BM25 trên khối def/class) — như "codebase search" của Cursor |
  | `read_file` | `path`, `offset`, `limit` | đọc theo ĐOẠN, có số dòng, chỉ chỗ đọc tiếp |
  | `edit_file` | `path`, `old`, `new`, `all` | thay đoạn `old` phải DUY NHẤT (như `Edit` của Claude Code); **phải `read_file` trước**; không thấy → gợi ý dòng gần giống nhất; trả về đoạn sau sửa |
  | `write_file` | `path`, `content` | tạo file mới (file đã có phải đọc trước) |
  | `delete_file` | `path` | xoá file |
  | `bash` | `command`, `timeout` | shell trong sandbox, KHÔNG mạng |
  | `run_tests` | — | lệnh test của dự án |
  | `lint` | `path` | lỗi cú pháp + pyflakes (tên chưa định nghĩa, import thừa) |

  Kiểm: từng công cụ + đường lỗi (sửa chưa đọc, `old` không duy nhất, sai khoảng trắng, ghi ngoài dự án, bash không mạng);
  model giả đi đúng quy trình "glob → chạy test → đọc → edit_file đúng dòng → lint → test" → **40/40**.
- `scripts/code_eval.py` — HumanEval 164 (test công khai = ví dụ `>>>` mà lời giải chuẩn qua: 69 bài), MBPP sanitized test 257,
  **40 bài tool-use** dựng từ MBPP train/validation (20 sửa lỗi cài sẵn + 20 viết thân hàm; đã kiểm: lời giải gốc qua, bài giao hỏng;
  model giả ghi lời giải đúng → 40/40).
- Bẫy đã bắt: Python 3.12 `-I` không cho test import `src.py`; cờ `-E` bỏ qua `PYTHONDONTWRITEBYTECODE` → `.pyc` cũ được dùng lại
  sau khi sửa file (cùng cỡ, cùng giây) → chấm sai → dùng `-B`; doctest dính `"""` vào kết quả mong đợi.
- Dữ liệu ảnh code (TK_107): 20K ảnh code/terminal train + 3K cho vòng thầy + 300 test, đã lên HF.

## 2. Dữ liệu — nguồn, giấy phép, cách lấy (đã kiểm trên HF ngày 13/10)

### 2.1 Code nguồn mở (mid-training)

| Nguồn | Nội dung | Giấy phép / truy cập | Có sẵn nội dung? | Dùng |
|---|---|---|---|---|
| **Stack-Edu** (`HuggingFaceTB/stack-edu`) | code GitHub đã chấm "giá trị giáo dục", nhiều ngôn ngữ, kèm `detected_licenses` | không khoá; mỗi file theo giấy phép gốc | **KHÔNG** — chỉ `blob_id`, nội dung tải từ S3 Software Heritage (cần tài khoản AWS) | nguồn chính, lọc lấy giấy phép dễ dãi |
| **The Stack v2 / dedup** | kho code mở lớn nhất, có cơ chế tác giả xin rút | khoá tự động (đồng ý điều khoản) | KHÔNG — như trên | dự phòng |
| **The Stack v1 dedup** | bản cũ hơn | khoá tự động | **CÓ** nội dung | dự phòng nếu S3 vướng |
| `codeparrot/github-code-clean` | code GitHub 30+ ngôn ngữ, đã lọc cơ bản | Apache-2.0 (bộ dữ liệu); từng file kèm giấy phép | **CÓ** | nguồn nhanh, không cần AWS |
| `OpenCoder-LLM/opc-annealing-corpus` | `algorithmic_corpus` · `synthetic_code_snippet` · `synthetic_qa` (dùng cho đoạn hạ lr của OpenCoder) | ODC-By | CÓ | **đoạn "chốt"** |
| `OpenCoder-LLM/opc-fineweb-code-corpus` | trang web về code lọc từ FineWeb | MIT | CÓ | văn bản kỹ thuật |
| **CommitPackFT** (`bigcode/commitpackft`) | commit thật: code trước/sau + mô tả sửa | MIT | CÓ (nạp file trực tiếp — loader script cũ không chạy với `datasets` mới) | **sửa lỗi / chỉnh sửa code** |
| Notebook (`codeparrot/github-jupyter-*`) | code xen giải thích + kết quả | Apache-2.0 | CÓ | "viết → chạy → đọc kết quả" |
| `codeparrot-clean` | Python | — | CÓ | **đã dùng ở P1** (~8% của 20 tỷ) |

**Giấy phép:** chỉ giữ file MIT / Apache / BSD / ISC / Unlicense / CC0 (theo `detected_licenses` hoặc trường license của file). GPL/
AGPL/LGPL và "không rõ" → **loại** (aivie là sản phẩm thương mại).

### 2.2 Sách và tài liệu mở

| Nguồn | Giấy phép | Thương mại |
|---|---|---|
| Tài liệu Python (docs.python.org), Rust Book, Go docs, PostgreSQL docs, Node.js docs | PSF / MIT·Apache / BSD / PostgreSQL / MIT | ✅ |
| MDN Web Docs | CC BY-SA 2.5 | ✅ (ghi nguồn) |
| SICP, Dive Into Python 3, Open Data Structures | CC BY-SA | ✅ |
| Wikipedia mảng khoa học máy tính (vi + en) | CC BY-SA | ✅ |
| Think Python, Eloquent JavaScript, Pro Git | CC BY-**NC** | ❌ chỉ nghiên cứu — KHÔNG đưa vào bản thương mại |
| Giáo trình ĐH Việt Nam, VNOI wiki, blog kỹ thuật Việt | kiểm từng nguồn | tuỳ |

Ước lượng 100–200 triệu token. Tiếng Việt hiếm → **thầy viết lại/dịch** sách mở sang tiếng Việt (giữ nguyên code) + **"sách tổng
hợp" theo đề cương** (mở rộng `gen_code_textbook.py`), mọi ví dụ code phải **chạy được**.

### 2.3 Môi trường việc thật (tool-use / RL)

| Nguồn | Nội dung | Giấy phép |
|---|---|---|
| **SWE-Gym** (`SWE-Gym/SWE-Gym`) | issue thật + `base_commit` + `patch` + `test_patch` + `FAIL_TO_PASS` / `PASS_TO_PASS` | MIT |
| SWE-bench Lite | 300 việc — **chỉ dùng để ĐO**, không train | — |
| Bài tự dựng (như 40 bài hiện có) | cài lỗi / bỏ thân hàm / traceback từ code đã có test | tự tạo |

## 3. Lọc dữ liệu (pipeline `build_code_cpt.py`, chạy trên box CPU như P1)

1. **Giấy phép** (mục 2.1) · bỏ file tự sinh / nén một dòng / dữ liệu (dòng > 1.000 ký tự, tỉ lệ chữ-số bất thường, `.min.js`, lock file).
2. **Khử trùng** chính xác (hash) + gần đúng (MinHash, như P1).
3. **Chấm chất lượng**: Stack-Edu đã có `int_score` (giữ ≥ 3); nguồn khác → bộ chấm giáo dục kiểu `edu_classifier.py` cho code.
4. **Bí mật / thông tin cá nhân**: quét API key, token, mật khẩu, email, khoá riêng → che hoặc bỏ file.
5. **Chống rò bộ đo**: loại file chứa đề/lời giải HumanEval, MBPP, SWE-bench Lite (so n-gram 13 token).
6. **Đóng gói theo repo**: file cùng repo xếp liền (README → mã nguồn → test) để học cách các file gọi nhau; một phần dạng **FIM**
   (điền vào giữa: tiền tố / hậu tố / phần giữa) để hoàn thành code giữa dòng.

## 4. Kế hoạch theo giai đoạn

| Giai đoạn | Việc | Dữ liệu | Máy / thời gian (ước) |
|---|---|---|---|
| **C0** ✅ | Bộ đo + khung chạy có công cụ | — | Mac |
| **C1** | Đo mốc đầy đủ (HumanEval 164, MBPP 257, tool-use 40) cho v5, bản SFT cuối, bản SFT+expert | — | 1×H100, vài giờ (Mac quá chậm: 1–3 phút/bài) |
| **C2** | **Dữ liệu code đã chạy thật**: thầy sinh bài + test → nhiều lời giải → giữ lời giải QUA test; kiểu bài: viết hàm, sửa theo traceback, viết test, refactor, SQL, bash, đọc hiểu code (vi + en) | 100–200K mẫu | thầy trên 1–2×H100, ~1 ngày |
| **C3** | **Quỹ đạo tool-use**: thầy chạy khung `code_agent.py` (chế độ tool) trên bài tự dựng + SWE-Gym, giữ quỹ đạo THÀNH CÔNG (test qua) | 20–50K quỹ đạo | 1–2×H100, ~1 ngày |
| **C4** | Train lại **expert code** (hoặc SFT bổ sung) trên C2 + C3 + ảnh code; loss chỉ tính lượt model, KHÔNG tính `<obs>` | — | 1×H100, ~0,5 ngày |
| **C5** | **RL bằng test** (GRPO): mỗi bài sinh 8 lời giải, thưởng = test qua (+ thưởng nhỏ cho đúng định dạng công cụ); cả bài một-lần lẫn bài tool-use nhiều lượt | bài có test (C2 + tool-use) | 1×H100, 1–2 ngày |
| **C6** (nếu vẫn chạm trần) | **Mid-training code** từ `cpt_b_stable.pt` (HF công khai): 3–10 tỷ token; mở ngữ cảnh 16–32K bằng dữ liệu đóng gói theo repo; đoạn hạ lr dùng sách + opc-annealing + Stack-Edu điểm cao | mục 2.1–2.2 | 4×H100, 1–3 ngày (~40K tok/s → 10 tỷ ≈ 70 giờ) |

**Trộn gợi ý cho C6:** Stack-Edu / github-code-clean (Python, JS/TS, SQL, bash, Java, C++, Go) ~40% · CommitPackFT + notebook ~10% ·
sách + tài liệu mở ~5% (tăng trọng số ở đoạn hạ lr) · toán ~10% · **ôn lại tiếng Việt + dữ liệu cũ ~35%** (chống quên — bài học
`qlora-data-da-dang`). Sau C6 phải chạy lại SFT + expert (P3) trên thân mới.

**Thứ tự đề xuất:** C1 → C2 → C3 → C4 → C5, đo sau mỗi bước bằng C1; chỉ làm C6 nếu C2–C5 không đủ.

## 5. Khung chạy khi dùng thật (UI / API)

- UI chat: nút **"Chạy thử"** cho khối code → sandbox → hiện kết quả; chế độ **trợ lý code** tự lặp "viết → chạy test → sửa" tối đa N
  vòng và trả lời KÈM kết quả chạy thật.
- Tool-use trong UI dùng đúng giao thức của C3 (model đã học) — thư mục dự án ảo cho mỗi cuộc trò chuyện.
- RAG tài liệu thư viện (docs mở ở mục 2.2) cho câu hỏi API.

## 6. Rủi ro và cách chặn

| Rủi ro | Chặn |
|---|---|
| Bản quyền / giấy phép | chỉ giấy phép dễ dãi; loại NC/GPL/không rõ; giữ danh sách nguồn + giấy phép trong README dữ liệu |
| Code chứa bí mật / thông tin cá nhân | quét và che ở bước lọc |
| Rò bộ đo → điểm ảo | lọc n-gram với HumanEval/MBPP/SWE-bench; bộ tool-use dựng từ phần MBPP KHÔNG dùng để đo |
| Model nghiêng sang "giải thích thay vì viết code" (đã thấy ở SFT bước 2.500) | tỉ lệ mẫu "viết code hoàn chỉnh" cao; RL thưởng theo test |
| Quên tiếng Việt / chat khi tăng code | ôn lại ~35%; expert có cổng (thân đóng băng) cho C4 |
| Sandbox bị lách khi chạy code model | sandbox hệ điều hành + giới hạn tài nguyên; chạy RL/thầy trên box dùng-xong-bỏ, không có khoá quan trọng |
| Stack-Edu cần AWS để tải nội dung | bắt đầu bằng nguồn có sẵn nội dung (github-code-clean, opc, CommitPackFT); thêm Stack-Edu khi có tài khoản AWS |

## 7. Quyết định

**13/10: user chọn phương án C** — giữ thứ tự: xong P3 (TK_107) rồi mới làm C1 → C5; không thuê box song song, không mid-training trước.

### Câu hỏi còn mở

1. Thứ tự C1 → C5 (không mid-training trước) — đồng ý?
2. Thầy cho C2/C3: Gemma-4-31B (đang dùng) hay thêm một model mạnh về code có giấy phép mở (kiểm giấy phép trước khi dùng)?
3. Tài khoản AWS để tải nội dung Stack-Edu / The Stack v2 (chỉ cần cho C6).
4. Box: C1–C5 chạy trên 1×H100 hiện tại sau khi P3 xong, hay thuê thêm box cho thầy chạy song song?

## 8. Làm song song lúc chờ P3.3' (14/10)

- **Kiểm chứng cổng P3** (`out/bao_cao_kiem_chung_cong_expert.md`): cổng đóng không phải nguyên nhân. Kết quả từng chế độ:
  - normal: HumanEval 4/40, Excel 17/27
  - off: 0/40, 3/27
  - open: ra rác
- **C2 phần không cần thầy:** `scripts/c2_build_verified.py` → `data/c2/oci_verified.jsonl`.
  - **99 658 bài** Python đã chạy lại test trong sandbox, qua 97,1%; giữ `tests` để dùng cho C5.
  - Nguồn: OpenCodeInstruct phần 0–3/50 (CC-BY-4.0). Đã bỏ bài giống HumanEval/MBPP.
  - Còn thiếu: bản tiếng Việt (thầy dịch đề), các kiểu sửa lỗi / SQL / bash. Mở rộng: thêm phần 4–49 (mỗi phần khoảng 25K bài qua).
- **Expert 2 giai đoạn:** `expert_office_1p3b.py trainGEN2`.
  - Giai đoạn A: cổng mở hẳn, mỗi bước bật một expert trên dữ liệu miền của nó (`--domain`).
  - Giai đoạn B: đóng băng FFN, cổng khởi tạo gần mở (`--gate-init 3`), học cổng trên `--mix` chung.
  - Đã chạy thử trên Mac (thân 1.3B, MPS). Dùng cho C4.

**14/10: quyết định của user**
- TK_108 chạy trên **box H100 hiện tại** (Vast P3). Xong P3 thì vào C1 ngay.
- Cách chạy C1: `scripts/box_run_c1.sh` tự chờ "P3.3' XONG", rồi chạy song song 3 máy chủ UI trên GPU để đo HumanEval, MBPP và tool-use.
- Nơi học từng bước:
  - **C2–C5:** tạo **expert code MỚI** bằng `trainGEN2`, rồi học RL trên chính expert đó. Thân, expert vp/hoc và phần đọc ảnh đều đóng băng.
  - **C6:** chỉ làm khi chạm trần. Cho thân học thêm code, sau đó học lại mọi expert.

**15/10: chọn thầy cho C2/C3**
- **C2:** giữ Gemma-4-31B. Lý do: bộ lọc chạy test đã loại lời giải sai, và Gemma dịch đề tiếng Việt tốt.
- **Đầu C3:** thử nhỏ để chọn thầy, KHÔNG quyết theo bảng xếp hạng.
  - Cho Gemma-4-31B và Qwen3-Coder-30B-A3B cùng chạy khoảng 50 bài tool-use giống hệt nhau, qua khung `code_agent` (chế độ tool).
  - Đo tỉ lệ thành công (test qua) và tốc độ. Thầy nào thắng thì sinh quỹ đạo cho C3.
- Trước khi dùng Qwen: kiểm giấy phép trên model card chính thức (các trang tổng hợp ghi là Apache 2.0).
- Số liệu tham khảo, phần lớn từ trang tổng hợp:
  - Gemma-4-31B: LiveCodeBench khoảng 80%.
  - Qwen3-Coder-30B-A3B: SWE-bench Verified khoảng 52%. Là MoE, chỉ 3B tham số hoạt động nên sinh nhanh.

**15/10: đo tokenizer `bpe_v1` (khoảng 32K token) trên code** (2000 lời giải Python từ C2; đơn vị: ký tự/token)

| Tokenizer | Code Python | Đề tiếng Anh | Tiếng Việt |
|---|---|---|---|
| aivie | 2,57 | 2,44 | 2,95 |
| Qwen2.5-Coder (151K) | 3,96 | 3,58 | 3,25 |

- Cùng một đoạn code, aivie tốn khoảng **1,5×** số token so với Qwen-Coder.
- Khứ hồi đúng 2000/2000; không có UNK; thụt 4 hay 8 dấu cách đều gộp thành 1 token.
- **Điểm yếu:** từ khoá ở đầu dòng (sau token thụt lề, nên không có tiền tố `Ġ`) bị cắt vụn: `return` → `ret|urn`, `while`, `elif`, `except`, `raise`, `with`, `yield`. Có dấu cách đứng trước cũng vậy: `except`, `lambda`, `False`, `append`, `dict`, `enumerate`.
- 14,7% số token trong code chỉ là khoảng trắng hoặc xuống dòng.
- **Kết luận:** không chặn C2–C5. Khi làm C6, cân nhắc thêm khoảng 4–8K merge học từ code và thêm hàng embedding mới. Với cách "expert lớn", có thể chỉ học các hàng embedding mới, các hàng cũ đóng băng.

## 9. C1 → C4: đo mốc, dữ liệu code chạy thật, quỹ đạo tool-use (15–16/10)

**C1 — mốc p3r1 (15/10):** HumanEval 6,1 · MBPP 5,1 · tool-use 0% (82% lượt không gọi công cụ). Riêng Excel: vòng thầy P3.3' làm
tụt Excel từ 17/27 xuống 9/27 vì thầy viết công thức lẫn trong câu. Đã sửa bằng `build_office_fix.py` (đưa công thức vào khối
```` ```excel ````) và học thêm expert văn phòng, ra **p3r1xl**: Excel 13/27, công thức trong khối 27/27.

**C2 — dữ liệu code đã chạy thật (`c2_teacher.py`, thầy Gemma-4-31B):**

| Loại | Làm | Giữ |
|---|---|---|
| vi — dịch đề, khối code giữ y hệt | 30 000 | 29 999 |
| debug — cài lỗi → lỗi thật → bản sửa qua test | 20 000 | 12 908 |
| predict — kết quả lấy từ chạy code thật | 10 000 | 9 111 |
| sql — 2 lời giải độc lập phải trùng kết quả | 8 000 | 7 654 |
| oci_verified (OpenCodeInstruct, không cần thầy) | — | 99 658 |

**C2b — code đa ngôn ngữ:**
- A: 10 ngôn ngữ, kiểm cú pháp bằng tree-sitter/sqlglot, ra 361 824 mẫu.
- C2b+: Next.js, Tailwind/WebSight, Svelte 5, Kotlin Compose, Flutter, shadcn.
- B: thầy viết test rồi chạy thật bằng node, tsc, javac, dotnet, php, go, bash, Chromium (HTML chụp 2 khổ và để thầy nhìn ảnh chấm) và SQL Server 2022 (T-SQL). Thầy sửa tối đa 2 lượt và nhật ký thầy được lưu đầy đủ.
- Cứu được khoảng 35% mẫu có đề tiếng Trung lẫn từ OpenCoder bằng cách cho thầy dịch đề.
- Từ nhật ký thầy dựng thêm dữ liệu "sai → lỗi → sửa" (`c2b_fixmix.py`): lượt sai đánh dấu `nolearn`, chỉ học bản sửa đã chạy qua.

**C3 — quỹ đạo tool-use (`c3_teacher.py`):**
- Bài dự án nhỏ dạng fix và implement dựng từ C2.
- Thử nhỏ chọn thầy trên 50 bài: **Gemma 41/50 · Qwen3-Coder 28/50** → chọn Gemma.
- Chỉ giữ quỹ đạo thành công và đúng giao thức.
- **C3-R:** bài có lượt SAI chèn sẵn (TAB, sửa sai, lặp lại; `nolearn`), thầy phải phục hồi trong ≤ 15 bước.

**C4 — expert code nhỏ (0,3 tỷ tham số, 12 điểm cắm):**

| Bản | HumanEval | MBPP | Tool-use | Ghi chú |
|---|---|---|---|---|
| c4e (C4 sớm, `trainGEN2` 2 giai đoạn) | 7,3 | 9,3 | 0 | 99,8% lượt không gọi công cụ |
| c4b (C4 lần 2, + C3 45%) | 11,0 | 11,7 | 32,5 | |
| c4c (C4 lần 3, + C2b + C3-R) | 11,0 | 9,7 | 22,5 khung v2 → **45 khung v3** | MultiPL-E 2,4 |

**Khung công cụ v2 → v3 (`code_agent.py`)** — sửa 27/40 bài hỏng vì vòng lặp khi gặp lỗi:
- Gọi y hệt lượt trước thì nhắc model đổi cách.
- `edit_file` lệch TAB hoặc dấu cách thì chỉ ra đúng đoạn trong file (v3 tự khớp thụt lề).
- `write_file` có nội dung y hệt thì báo lại cho model.
- Đặt `AIVIE_TOOL_GUARD=0` để dùng lại khung cũ khi so sánh.

**Bộ đo đa ngôn ngữ `code_eval_ml.py`:** MultiPL-E (MIT), HumanEval × JS/TS/Java/C#/PHP/Go/Bash. Mỗi lần đo lấy 30 bài/ngôn ngữ, chạy thật trong sandbox.

## 10. C5: DPO và GRPO — không hơn, đã dừng (17/10)

- **C5a DPO** (2 009 cặp đúng ↔ sai từ nhật ký thầy, gốc c4c): HumanEval 7,9 so với 11,0, MBPP khoảng 7 so với 9,7, MultiPL-E thấp hơn. **Bỏ.**
- **C5b GRPO** (`c5b_grpo.py`, thưởng = test qua): dừng ở khoảng 172 bước.
  - Bộ kiểm cố định 24 đề đứng yên ở 12–13/24. Python 39% → 41%.
  - Đa ngôn ngữ khoảng 0%, nên 40% số đề không có tín hiệu thưởng.
  - Kết luận: **trần là năng lực expert nhỏ chứ không phải cách học** → chuyển sang expert lớn.

## 11. Pilot expert code LỚN (17–18/10) — thành công

- 24 điểm × ẩn 8192 = 1,21 tỷ tham số.
- Pretrain riêng expert trên thân đóng băng (`trainCPTX`, cổng mở): 321 triệu token (`build_code_expert_data.py`).
- Sau đó SFT bằng đúng dữ liệu C4 lần 3.

| Bản | Tool v3 | HumanEval | MBPP | MultiPL-E | Excel | easy20 |
|---|---|---|---|---|---|---|
| c4c (expert nhỏ, mốc) | 45 | 11,0 | 9,7 | 2,4 | 16/27 | 15/20 |
| cx (expert lớn + SFT) | 45 | 13,4 | 17,1 | 5,7 | 14/27 | 16/20 |
| cx2 (cx + SFT bù tool-use 600 bước) | 47,5 | 13,4 | 15,6 | — | 14/27 | 17/20 |

**Bẫy đo đã bắt trong pilot:**
- 4 máy chủ UI chạy expert lớn cần 81/80 GB. Câu dài bị OOM, máy chủ nuốt lỗi và trả rỗng, nên lần đo đầu ra tool-use 5–10% (số hỏng). Giờ máy chủ in `[lỗi /api/chat]` và chạy tối đa 2 máy chủ/GPU 80 GB.
- UI cũ để `MAXCTX` 2000, trong khi quỹ đạo C3 train ở 4096. Đã thêm `--max-ctx 4096`. Riêng việc tăng max_new lên 1024 mà vẫn để ctx 2000 làm tool-use tụt từ 47,5 xuống 20.
- Fork bomb từ code model sinh ra (MultiPL-E sh) để lại 5 653 tiến trình làm cạn box. Sửa bằng `run_group` + `kill_group` và RLIMIT_NPROC cho user sbx.

## 12. Chạy thật: cx3 → cx3b (19–22/10, box 2×A100 SXM4 80 GB, $1,67/giờ)

**Dữ liệu `data/cx3`:**
- 3,4 tỷ token, đẩy lên HF `nn-aivie-cpt-data/cx3` (công khai).
- Nhóm `gh` lấy từ `codeparrot/github-code-clean`, chỉ giữ giấy phép mit, apache-2.0, bsd, isc, unlicense, cc0, có giới hạn theo từng ngôn ngữ.
- Tập giữ riêng trùng với pilot. Vòng đầu dùng 1,5 tỷ token.

**Pretrain cx3pt:**
- Chạy `box_run_cx3_pt.sh`: `torchrun` 2 GPU, all_reduce tay, `--resume`, 23 000 bước, 9,1K token/giây, 47 giờ.
- `box_cx3_hf_sync.sh` đẩy expert + trạng thái chạy tiếp lên HF mỗi 1000 bước (squash lịch sử) để phòng box bị thu hồi.
- Loss giảm 25–27% so với mốc tương đương pilot ở algo, snip, files (bảng trong `TK_108_KIEN_TRUC_CX3B.md` §5).

**SFT:**
- **cx3** (`box_run_cx3_post.sh`): trộn C4 lần 3 + C3-R2, 1500 bước, lr 1e-4.
- **cx3b** (`box_run_cx3b.sh`): SFT bù tool-use 600 bước, lr 5e-5.

| Bản | Tool v3 | HumanEval | MBPP | MultiPL-E | Excel | easy20 |
|---|---|---|---|---|---|---|
| cx2 | 47,5 | 13,4 | 15,6 | 5,7 | 14/27 | 17/20 |
| cx3 | 42,5 | 12,2 | 22,2 | 6,7 (ts 16,7 · java 13,3 · js 6,7 · go 6,7 · php 3,3 · cs 0 · sh 0) | 16/27 | 16/20 |
| **cx3b** | **55,0** (fix 75 · implement 35) | 13,4 | **23,3** (sau sửa 24,9) | — | 15/27 | 16/20 |

**Bài học:**
- Bước SFT bù tool-use ở pilot trông như "không cần" (cx 45 ≈ cx2 47,5). Nhưng sau pretrain dài thì nó **rất cần** (cx3 42,5 → cx3b 55). Đây là bước cố định trong công thức.
- Pretrain dài giúp rõ ở MBPP và đa ngôn ngữ. Riêng HumanEval gần như đứng yên (12–13%) qua cx2, cx3, cx3b.

**Bẫy trên box mới (NGC image):**
- `hf` lỗi typer → `PIP_CONSTRAINT= pip install -U "typer>=0.16"`.
- Python hệ thống thiếu `transformers`, nên UI chết khi nạp SigLIP.
- `PIP_CONSTRAINT` ghim `datasets` 2.2.1 trong venv đo, lệch với `fsspec` mới, gây lỗi `'**' can only be an entire path component`. Sửa bằng `datasets>=3` + `numpy>=2`.
- `hf download` nhiều `--include` có lúc bỏ sót file (`data/c2/oci_verified.jsonl`). Script chuẩn bị giờ kiểm từng file.

**Phần cứng (tốc độ đo trên đúng bài này):**

| Box | Tốc độ | Giá |
|---|---|---|
| 1×H100 SXM | khoảng 9,0K token/giây | |
| 2×A100 SXM4 | 9,1K token/giây | $1,67/giờ — rẻ nhất tính theo token |
| 4×A100 | 17,9K token/giây | $6,4/giờ |

## 13. Thử thực tế trên M4 (21/10) và bước tiếp theo

- **8 đề thực tế nhiều ràng buộc trên cx2:**
  - 3/3 bài có test chạy thật đều trượt: tính hoá đơn VAT, đếm lỗi 5xx trong log nginx, debug lỗi `gio=[]` mặc định.
  - T-SQL không JOIN bảng. Excel dùng SUMIF sai, đúng ra phải là SUMIFS. Debounce JS sai. Tư vấn tối ưu .NET thì lạc đề.
  - Chỉ có email là tốt.
  - Kết luận: model viết code theo khuôn mẫu và không bám ràng buộc của đề. Đây là năng lực của thân đóng băng.
- `serve_hybrid_ui.py --half`: nạp bf16. Ở fp32 bản này cần khoảng 17 GB nên M4 24 GB bị swap, sau 2,5 giờ không ra nổi một câu trả lời.
- **Việc tiếp theo cần chốt:**
  1. Chạy lại 8 đề thực tế trên cx3b để so trực tiếp với cx2.
  2. Pretrain tiếp expert lên đủ 3,4 tỷ token (`--resume` từ `nn-aivie-cx3-ckpt`), rồi SFT và bù tool-use.
  3. Làm C6 (học vào thân) nếu cần vượt trần hiểu đề.

**22/10: cùng 8 đề thực tế trên cx3b** (`scripts/m4_probe_real.py`, đáp án lưu ở `out/m4_probe_cx3b.jsonl`):
- Kết quả gần như y hệt cx2: 3 bài có test chạy thật vẫn trượt cả 3, T-SQL vẫn không JOIN, Excel vẫn SUMIF (lần này đúng cột B nhưng thiếu điều kiện tháng), debounce sai, tư vấn .NET lạc đề sang viết class C#, chỉ email tốt.
- Kiểm thêm 3 đề ngắn để chắc M4 bf16 không làm hỏng model:
  - Đề tiếng Anh đúng khuôn MBPP (`Your code should pass: assert …`) → **QUA**.
  - Đề tiếng Việt ngắn (đảo thứ tự từ) → trượt: tách theo dấu phẩy.
  - Đề không nói ngôn ngữ → model viết JavaScript.
- **Kết luận:** model giỏi lên ở **đúng khuôn đề của bộ đo** (MBPP 15,6 → 23,3; tool-use trong khung 47,5 → 55), nhưng không chuyển sang đề tự do và đề tiếng Việt nhiều ràng buộc.
  - Nguyên nhân 1: thân đóng băng hiểu đề yếu.
  - Nguyên nhân 2: dữ liệu SFT thiên về khuôn "đề ngắn tiếng Anh → một hàm".
- **Hướng sửa rẻ trước C6:** SFT thêm đề thực tế nhiều ràng buộc bằng tiếng Việt, có test (thầy sinh + chạy thật như C2). Ví dụ: nghiệp vụ hoá đơn, log, báo cáo SQL có JOIN, Excel nhiều điều kiện. Đồng thời đo bằng một bộ đề kiểu này thay cho việc chỉ nhìn HumanEval/MBPP.

## 14. C7: đề thực tế tiếng Việt (22/10)

- **Dữ liệu (`scripts/c7_real.py`):** thầy Gemma-4-31B FP8 chạy 2 vLLM, mỗi GPU A100 một bản. Đề thuộc 8 kiểu: Python nghiệp vụ, Python dữ liệu, Python chuỗi, debug, class, SQL, Excel, JS.
- **Hai lớp kiểm bằng chạy thật:**
  1. Lời giải mẫu phải qua 8–12 test.
  2. Thầy giải lại độc lập chỉ từ đề, và lời giải đó cũng phải qua đúng bộ test.
  Tỉ lệ giữ khoảng 28%, khoảng 13,5 đề/phút.
- **Bộ đo thực tế** `data/c7/eval.jsonl` (355 đề, không train) là **thước đo chính từ nay**. **Bộ train** 3 000 đề (user hạ từ 10K xuống 3K để kịp đo trong ngày).
- **cx3c = cx3b + SFT 700 bước** (C7 35% + ôn tool-use / code / văn phòng, lr 5e-5). Loss C7 giảm 0,80 → 0,41.

| Bản | Bộ đo C7 (355) | Excel C7 | Python nghiệp vụ | HE | MBPP | Tool |
|---|---|---|---|---|---|---|
| cx3b | 0,6% | 1/39 | 0/116 | 13,4 | 23,3 | 55 |
| cx3c | 2,5% | 6/39 | 0/116 | **15,9** | **26,5** | 50 |

**Bẫy gặp khi chạy C7:**
- Thư viện `formulas` treo vô hạn trong lúc giữ khóa chung. Sửa: tính Excel trong tiến trình con có hạn giờ.
- TP 2 GPU lỗi custom all-reduce trong container. Sửa: chạy 2 vLLM, mỗi GPU một bản.
- Kernel FP8 trên A100 cần `LD_LIBRARY_PATH` trỏ tới `nvidia/cu13/lib`.
- Thầy chép mã kịch bản vào đề. Sửa: lọc bằng `strip_tag`.
- `pkill -f` trong cùng lệnh ssh giết luôn phiên ssh. Sửa: chép file script lên box rồi chạy bằng `setsid`.

## 15. C8: LoRA khối trộn của thân (22–23/10)

**Giả thuyết:** trần nằm ở thân đóng băng. Expert FFN không chạm tới phần đọc và nối các ý của đề, vốn nằm ở Mamba và attention.

**Cách làm:**
- `CodeLoRA` (`expert_ffn_1p3b.py`) cắm lên `wq/wk/wv/wo` và `in_proj/out_proj` từ khối 3 trở lên, tổng 210 lớp chiếu.
- Học cùng expert code: `trainGEN --lora-r`, lr 2e-4.
- So sánh sạch với cx3c: cùng gốc cx3b, cùng trộn dữ liệu, cùng 700 bước.

| Bản | LoRA | Bộ đo C7 | Excel | Python nghiệp vụ | SQL | Tool | Văn phòng |
|---|---|---|---|---|---|---|---|
| cx3c | — | 2,5% | 6/39 | 0/116 | 1/43 | 50 | — |
| cx3d | r32 (+49M) | 3,4% | 8/39 | 1/116 | 1/43 | 55 (fix 80) | chờ |
| cx3e | r128 (+195M) | 3,7% | 7/39 | 3/116 | 2/43 | 55 | Excel 15/27 · easy20 15/20 |

**Kết luận:**
- LoRA chỉ thêm 3–4 đề, nằm trong ngưỡng nhiễu. r128 không hơn r32, nên dung lượng LoRA không phải nút thắt.
- Chỉnh nhẹ thân bằng 3K đề cũng không vượt được trần.
- Với thân 2,5B học 20 tỷ token (8% code), hướng còn lại là **C6**: học sâu vào thân với 10–20 tỷ token code + ôn tiếng Việt, rồi SFT lại P3. Đề xuất tham số và chi phí đã trình user ngày 23/10. Cần chốt: số token, loại máy, có thêm token code vào tokenizer không.
