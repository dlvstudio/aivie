#!/usr/bin/env python3
"""Sinh data/text_corpus/styled_humor_history.jsonl — hài gắn entity lịch sử VN.

Mỗi thân: Setup (fact thật) → Pivot (thế nhưng/rốt cuộc) → Payoff (đời thường neo entity).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/text_corpus/styled_humor_history.jsonl"

# (prompt_label_style, body) — body đã gồm setup→pivot→payoff
PIECES: list[tuple[str, str]] = [
    # —— Ngô Quyền / Bạch Đằng ——
    (
        "Viết một đoạn văn hài hước về Ngô Quyền",
        "Ngô Quyền năm 938 đóng cọc nhọn xuống lòng sông Bạch Đằng rồi chờ thủy triều rút "
        "để đánh chìm thuyền Nam Hán. Thế nhưng nhìn cách ông bày trận, thời nay chắc cũng "
        "chỉ là một cú setup deadline cực cao: địch vào lúc nước lớn, mình thu hoạch lúc nước rút, "
        "báo cáo cuối quý nhìn mà lạnh sống lưng.",
    ),
    (
        "Viết hài hước về Ngô Quyền",
        "Ngô Quyền được xem là người mở đầu thời kỳ độc lập tự chủ sau hơn một nghìn năm Bắc thuộc. "
        "Rốt cuộc thành tích nghe oách, nhưng cách làm lại cực kỳ thực dụng: không cần slogan dài dòng, "
        "chỉ cần đúng chỗ đóng cọc và đúng giờ nước rút — kiểu project manager biết chờ sprint review.",
    ),
    (
        "Kể vui về trận Bạch Đằng năm 938",
        "Trận Bạch Đằng năm 938: thủy triều lên thì nhử, thủy triều xuống thì thu lưới. "
        "Thế nhưng Gen Z nghe xong chắc hỏi ngay: 'Ủa vậy có cần app đếm giờ triều không ạ?' "
        "Ông bà mình chơi lớn thật, còn mình thì nhắc lịch Google còn sót deadline.",
    ),
    (
        "Viết hài hước về Bạch Đằng",
        "Cắm cọc dưới sông Bạch Đằng nghe đơn giản, làm thì phải canh từng nhịp nước. "
        "Thế nhưng đó đúng là nghệ thuật ambush marketing thời cổ: để đối phương tự lao vào "
        "sản phẩm của mình, rồi mới lộ giá — à không, lộ cọc.",
    ),
    (
        "Hài hước về Ngô Quyền xưng vương",
        "Năm 939 Ngô Quyền xưng vương, mở nhà Ngô. "
        "Rốt cuộc nhìn timeline: đánh xong một trận lớn, năm sau đã launch brand mới. "
        "Startup thời nay gọi vốn còn lâu mới kịp tốc độ branding kiểu đó.",
    ),
    (
        "Viết hài hước về sông Bạch Đằng",
        "Sông Bạch Đằng trong sử không chỉ là phong cảnh, mà là cái bẫy thủy lực có chủ đích. "
        "Thế nhưng mỗi lần đi ngang sông lớn thời nay, mình chỉ nghĩ đến… phí đỗ xe ven bờ. "
        "Ông bà dùng sông để thắng giặc; mình dùng sông để check-in Story.",
    ),
    # —— Trần Hưng Đạo ——
    (
        "Viết hài hước về Trần Hưng Đạo",
        "Trần Hưng Đạo tên thật Trần Quốc Tuấn, tổng chỉ huy hai cuộc kháng chiến chống Nguyên Mông. "
        "Thế nhưng nếu ông sống thời nay, chắc Hịch tướng sĩ sẽ thành một thread dài trên mạng: "
        "đầu tiên khích lệ tinh thần, giữa kỳ nhắc KPI, cuối kỳ hỏi 'anh em còn theo không'.",
    ),
    (
        "Viết một đoạn văn hài hước về Trần Hưng Đạo",
        "Trần Hưng Đạo soạn Hịch tướng sĩ để khích lệ binh sĩ trước họa ngoại xâm. "
        "Rốt cuộc đọc lại thấy đúng là bản kickoff meeting hoàn hảo: rõ thù, rõ việc, rõ hậu quả nếu chùn bước. "
        "Team building thời nay hay hát karaoke; ông thì viết văn đánh thức cả đạo quân.",
    ),
    (
        "Kể vui về Trần Hưng Đạo và quân Nguyên",
        "Hai lần đẩy lui quân Nguyên Mông nghe như highlight reel bất bại. "
        "Thế nhưng bí quyết không phải 'cứ đánh là thắng', mà là biết lúc nào tránh, lúc nào dồn, "
        "lúc nào để đối phương tự mệt — y như biết lúc nào mute group chat để còn sống sót đến Friday.",
    ),
    (
        "Viết hài hước về Hịch tướng sĩ",
        "Hịch tướng sĩ không phải văn chương để treo tường cho đẹp. "
        "Thế nhưng nếu nộp bài văn đó vào lớp ngày nay, giáo viên chắc gạch đỏ: "
        "'Em viết cảm xúc mạnh quá, nên thêm dẫn chứng số liệu.' "
        "Thời Trần thì số liệu nằm ở số quân Nguyên phải quay về.",
    ),
    (
        "Viết hài hước về Binh thư yếu lược",
        "Trần Hưng Đạo còn soạn Binh thư yếu lược — đúng nghĩa handbook nội bộ. "
        "Rốt cuộc thời nay công ty nào cũng có quy trình, nhưng ít ai viết được quy trình "
        "mà đọc xong vẫn muốn xông ra làm việc. Đa số quy trình đọc xong chỉ muốn… xin nghỉ phép.",
    ),
    (
        "Hài hước về nhà Trần đoàn kết",
        "Nhà Trần nổi tiếng cả nhà một lòng đánh giặc. "
        "Thế nhưng thử convene họp gia đình chọn quán ăn cuối tuần xem: "
        "đánh Nguyên còn dễ hơn chốt được một địa chỉ không ai dỗi.",
    ),
    # —— Nhà Trần / kháng chiến ——
    (
        "Viết hài hước về Nhà Trần",
        "Nhà Trần ba lần chống Nguyên mà vẫn giữ được khí thế. "
        "Rốt cuộc nhìn lại giống một tập đoàn biết crisis management: "
        "có plan A rút về căn cứ, plan B phản công, plan C vẫn không bán logo quốc gia.",
    ),
    (
        "Kể vui về kháng chiến chống Nguyên Mông",
        "Quân Nguyên Mông từng làm mưa làm gió khắp lục địa, vậy mà sang Đại Việt lại vấp. "
        "Thế nhưng bài học mang về không chỉ là 'yêu nước', mà còn là địa hình + thời tiết + đoàn kết "
        "ăn đứt quân đông nhưng lạc lối — giống team nhỏ hiểu product hơn tập đoàn lớn copy feature.",
    ),
    (
        "Viết hài hước về vua Trần",
        "Vua Trần nhường ngôi, làm thái thượng hoàng, vẫn quanh quẩn chuyện nước. "
        "Thế nhưng kiểu succession planning đó thời nay gọi là 'founder vẫn ngồi hội đồng', "
        "nhân viên cười cười nhưng ai cũng biết: quyết định lớn vẫn không thoát được group chat ông ấy.",
    ),
    # —— Lý / Thăng Long ——
    (
        "Viết hài hước về Lý Thái Tổ",
        "Lý Thái Tổ dời đô về Thăng Long vì thấy rồng bay lên — tầm nhìn địa linh. "
        "Rốt cuộc quyết định relocation của cả quốc gia dựa trên một tín hiệu trời đất, "
        "còn mình chuyển nhà còn phải so bảng Excel tiền thuê ba tháng.",
    ),
    (
        "Viết hài hước về Thăng Long",
        "Thăng Long nghĩa là rồng bay lên, nghe đã thấy pitch deck nghìn năm. "
        "Thế nhưng nếu đặt tên khu đô thị thời nay kiểu đó, "
        "chắc sales bảo: 'Em ơi tên đẹp nhưng khách hỏi tiện ích trường học chỗ nào?'",
    ),
    (
        "Kể vui về Nhà Lý",
        "Nhà Lý mở kỷ nguyên Thăng Long, chùa chiền và khoa cử dần định hình. "
        "Thế nhưng nhìn lịch sử dài dòng ấy, học sinh thời nay vẫn chỉ nhớ "
        "có bài kiểm tra 'năm nào dời đô' — đúng là quốc gia nhớ bằng… đề thi.",
    ),
    (
        "Viết hài hước về dời đô Thăng Long",
        "Dời đô không phải chuyển văn phòng cuối tuần. "
        "Rốt cuộc Lý Thái Tổ làm một cú migrate toàn hệ thống: chính trị, kinh tế, biểu tượng. "
        "IT thời nay migrate database còn run book dày bằng sử ký.",
    ),
    # —— Lê Lợi / Lam Sơn ——
    (
        "Viết hài hước về Lê Lợi",
        "Lê Lợi khởi nghĩa Lam Sơn mười năm gian khổ mới đẩy được quân Minh. "
        "Thế nhưng nghe 'mười năm' là biết dự án không có sprint ngắn: "
        "toàn bộ roadmap là kiên nhẫn + liên kết lực lượng + đúng thời cơ — "
        "chứ không phải update status mỗi sáng rồi tưởng xong việc.",
    ),
    (
        "Kể vui về khởi nghĩa Lam Sơn",
        "Khởi nghĩa Lam Sơn lúc đầu lực mỏng, phải nương núi rừng. "
        "Rốt cuộc đó là giai đoạn stealth mode kinh điển: chưa launch ầm ĩ, "
        "chưa PR rầm rộ, chỉ âm thầm đủ mạnh rồi mới lộ diện. "
        "Startup nào cũng nói stealth; ít ai stealth được mười năm.",
    ),
    (
        "Viết hài hước về Bình Ngô đại cáo",
        "Bình Ngô đại cáo đọc như bản tuyên bố sản phẩm sau khi ship thành công. "
        "Thế nhưng khác chỗ: không có câu 'xin lỗi vì trải nghiệm chưa hoàn hảo', "
        "mà là khẳng định chủ quyền rõ ràng. PR thời nay nên học… và cũng nên sợ.",
    ),
    (
        "Hài hước về Nguyễn Trãi",
        "Nguyễn Trãi viết văn giúp Lê Lợi định hình chính nghĩa. "
        "Rốt cuộc vai trò copywriter chiến lược thời đó nặng ký thật: "
        "viết sai một câu là mất lòng quân, viết đúng thì cả nước nhớ mãi. "
        "Content creator thời nay trễ deadline chỉ mất… tương tác.",
    ),
    # —— Quang Trung / Tây Sơn ——
    (
        "Viết hài hước về Quang Trung",
        "Quang Trung hành quân thần tốc mùa Tết, đánh tan quân Thanh. "
        "Thế nhưng tốc độ đó nếu đem so với ship hàng Tết thời nay: "
        "ông tiến quân; mình thì chờ shipper và mã giảm giá hết hạn.",
    ),
    (
        "Kể vui về trận Ngọc Hồi - Đống Đa",
        "Ngọc Hồi – Đống Đa là cú knock-out Tết lịch sử. "
        "Rốt cuộc đối phương nghĩ mình đang nghỉ lễ, bên này nghĩ mình đang deadline. "
        "Chiến lược bất ngờ nhất đôi khi chỉ là… làm việc đúng lúc người ta chủ quan.",
    ),
    (
        "Viết hài hước về Tây Sơn",
        "Tây Sơn nổi lên từ nông dân, lớn nhanh như một phong trào viral. "
        "Thế nhưng viral thời xưa nghĩa là 'cả huyện theo'; thời nay nghĩa là "
        "'ba ngày trên for you rồi biến'. Tây Sơn giữ được lâu hơn một trend.",
    ),
    (
        "Hài hước về Nguyễn Huệ",
        "Nguyễn Huệ lên ngôi Quang Trung, vừa đánh vừa xây. "
        "Rốt cuộc multitasking mức độ quốc gia: chiến trường một bên, "
        "chính thể một bên. Mình mở hai tab Excel đã thấy… cần cà phê.",
    ),
    # —— Hai Bà Trưng ——
    (
        "Viết hài hước về Hai Bà Trưng",
        "Hai Bà Trưng khởi nghĩa chống ách đô hộ, cưỡi voi ra trận. "
        "Thế nhưng hình ảnh đó đủ mạnh đến mức nghìn năm sau vẫn là icon: "
        "không cần rebrand, không cần đổi logo — recognition ăn đứt nhiều chiến dịch marketing.",
    ),
    (
        "Kể vui về Trưng Trắc Trưng Nhị",
        "Trưng Trắc, Trưng Nhị đứng lên khi đất nước bị chèn ép. "
        "Rốt cuộc câu chuyện không chỉ là anh hùng ca, mà còn là "
        "biết nói 'đủ rồi' đúng lúc — skill mà group chat họ hàng vẫn chưa học được.",
    ),
    # —— Đinh / Tiền Lê ——
    (
        "Viết hài hước về Đinh Bộ Lĩnh",
        "Đinh Bộ Lĩnh dẹp loạn 12 sứ quân, thống nhất thiên hạ. "
        "Thế nhưng nhìn lại giống việc merge mười hai nhánh git đang conflict: "
        "ai cũng tuyên bố master, cuối cùng phải có người đủ cứng để… force merge lịch sử.",
    ),
    (
        "Kể vui về 12 sứ quân",
        "Thời 12 sứ quân, ai cũng xưng hùng một cõi. "
        "Rốt cuộc đúng là giai đoạn 'mỗi team một OKR riêng': "
        "không ai chịu align, đất nước thành dashboard đầy màu đỏ. "
        "Đinh Bộ Lĩnh đến như ông Scrum Master cầm gậy.",
    ),
    (
        "Viết hài hước về Lê Đại Hành",
        "Lê Đại Hành giữ nước sau Đinh, đánh Tống trên sông Bạch Đằng lần nữa. "
        "Thế nhưng Bạch Đằng như một template chiến thắng được reuse: "
        "cùng một sông, cùng kiểu bẫy nước — chứng tỏ best practice nào hay thì giữ, "
        "đừng reinvent bánh xe mỗi sprint.",
    ),
    # —— Lý Thường Kiệt ——
    (
        "Viết hài hước về Lý Thường Kiệt",
        "Lý Thường Kiệt chủ trương 'tiên phát chế nhân', đánh sang đất Tống trước. "
        "Rốt cuộc đó là tư duy proactive đúng nghĩa: không ngồi đợi ticket tới rồi mới fix. "
        "Team nào cũng nói proactive; ít ai proactive kiểu… xuất quân.",
    ),
    (
        "Kể vui về Nam quốc sơn hà",
        "Bài Nam quốc sơn hà khẳng định chủ quyền nghe ngắn mà đủ. "
        "Thế nhưng nếu đem so với terms of service thời nay: "
        "một bên bốn câu rõ ràng, một bên mười trang không ai đọc vẫn phải bấm Accept.",
    ),
    # —— Hồ / thuộc Minh / ——
    (
        "Viết hài hước về Hồ Quý Ly",
        "Hồ Quý Ly cải cách tiền tệ, hành chính — tham vọng lớn. "
        "Thế nhưng cải cách nhanh mà lòng người chưa theo thì "
        "giống roll-out feature lúc nửa đêm: kỹ thuật có thể đúng, "
        "user experience thì… nổi sóng.",
    ),
    (
        "Kể vui về thời thuộc Minh",
        "Thời thuộc Minh, ách đô hộ nặng nề. "
        "Rốt cuộc lịch sử nhắc rằng độc lập không phải default setting: "
        "mất rồi mới biết quý. Giống mất wifi năm phút là cả nhà thành nhà khảo cổ tìm router.",
    ),
    # —— Nguyễn / Gia Long / —— nhẹ nhàng
    (
        "Viết hài hước về thống nhất Đàng Trong Đàng Ngoài",
        "Đàng Trong – Đàng Ngoài chia cắt lâu, rồi mới về một mối. "
        "Thế nhưng nghe như merge hai codebase song song vài trăm năm: "
        "conflict đầy, ai cũng bảo branch mình sạch hơn. "
        "Cuối cùng vẫn phải có ngày release chung.",
    ),
    # —— Thêm biến thể Ngô / Trần cho cover retrieve ——
    (
        "Viết hài hước về Ngô Quyền và Nam Hán",
        "Quân Nam Hán kéo sang, tưởng thủy chiến là sở trường. "
        "Thế nhưng gặp cọc Bạch Đằng thì sở trường thành… sở hở. "
        "Đúng là đừng bao giờ overconfident vào đúng cái mình nghĩ mình giỏi nhất — "
        "nhất là khi đối phương đã đọc kỹ bản đồ và lịch triều.",
    ),
    (
        "Viết đoạn văn hài về độc lập tự chủ thời Ngô",
        "Độc lập tự chủ sau Bạch Đằng không phải món quà trên trời rơi xuống. "
        "Rốt cuộc là kết quả của một kế hoạch cụ thể: đúng chỗ, đúng giờ, đúng người chỉ huy. "
        "Khẩu hiệu suông không đóng được cọc; đóng cọc mới nâng được khẩu hiệu.",
    ),
    (
        "Hài hước về Trần Quốc Tuấn",
        "Trần Quốc Tuấn được đời sau nhớ với tên Hưng Đạo Vương. "
        "Thế nhưng ông để lại không chỉ chiến thắng, mà còn văn bản và binh thư — "
        "kiểu leader vừa deliver kết quả vừa để lại documentation. "
        "Hiếm. Rất hiếm. Nhất là trong mấy group dự án tan rã sau ngày release.",
    ),
    (
        "Viết hài hước về ba lần đánh Nguyên",
        "Ba lần chống Nguyên, mỗi lần một kịch bản. "
        "Rốt cuộc không copy nguyên xi lần trước: có lần giữ, có lần rút, có lần quyết. "
        "Agile thật sự là biết đổi chiến thuật, không phải đổi tên sprint cho vui.",
    ),
    (
        "Kể vui về cọc gỗ Bạch Đằng",
        "Cọc gỗ — vật dụng nghe dân dã — lại quyết định một trận thủy chiến. "
        "Thế nhưng bài học mang về: công nghệ cao không phải lúc nào cũng thắng "
        "công nghệ đúng chỗ. Thời nay cũng vậy: slide đẹp không cứu được sản phẩm sai bài toán.",
    ),
    (
        "Viết hài hước về tướng sĩ nhà Trần",
        "Tướng sĩ nhà Trần nghe Hịch xong là vào việc. "
        "Rốt cuộc văn khích lệ thời đó có conversion rate kinh khủng: "
        "đọc xong không phải thả tym, mà là cầm vũ khí. "
        "Content marketing nhìn vào mà học — và cũng mà… run.",
    ),
    # —— Hà Nội / Thăng Long mở rộng tone ——
    (
        "Viết hài hước về Hà Nội nghìn năm",
        "Hà Nội mang danh nghìn năm văn hiến, kế thừa Thăng Long. "
        "Thế nhưng sống ở đó thì nghìn năm ấy đôi khi gói trong một câu: "
        "kẹt xe cũng có bề dày lịch sử. Văn hiến là có thật; kiên nhẫn đèn đỏ cũng là có thật.",
    ),
    (
        "Kể vui về học sử Việt",
        "Học sử Việt mà chỉ nhồi năm tháng thì dễ ngủ gật. "
        "Rốt cuộc nhớ lâu nhất lại là mấy chi tiết 'đời thường hóa': "
        "cọc trên sông, hịch trong quân, hành quân mùa Tết — "
        "giống meme lịch sử nhưng lại là sự thật.",
    ),
    (
        "Viết hài hước về anh hùng dân tộc",
        "Anh hùng dân tộc trong sách giáo khoa thường đứng rất thẳng. "
        "Thế nhưng nếu kể lại bằng giọng đời thường, "
        "họ hóa ra cũng là những người biết tính toán, biết chờ thời, biết liên kết — "
        "không phải chỉ biết… tạo dáng trên tượng.",
    ),
    # —— Thêm Hỏi/Đáp format để khớp ingest 1k ——
]

# Thêm bản Hỏi/Đáp song song cho ~nửa đầu (để CO_ACT label↔body)
QA_EXTRA: list[tuple[str, str]] = []
for label, body in PIECES[:28]:
    q = label if label.endswith("?") else f"{label}?"
    if not q.lower().startswith("hãy") and not q.startswith("Viết") and not q.startswith("Kể"):
        q = f"Hãy {q[0].lower()}{q[1:]}" if q else q
    # Chuẩn hóa thành câu hỏi rõ
    if not q.startswith("Hỏi:"):
        ask = q if "?" in q else f"{q}?"
        QA_EXTRA.append((f"Hỏi: {ask}", body))


def main() -> None:
    rows: list[dict[str, str]] = []
    for label, body in PIECES:
        # format seed: "Viết …: body"
        text = f"{label}: {body}"
        rows.append({"text": text})
    for ask, body in QA_EXTRA:
        # format 1k
        text = f"{ask} Đáp: {body}"
        rows.append({"text": text})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} lines → {OUT}")


if __name__ == "__main__":
    main()
