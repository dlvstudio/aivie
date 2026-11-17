#!/usr/bin/env python3
"""
TK_028 — soạn data SỬ VIỆT NAM (nhà-sử-học), title-anchored, sự kiện được xác lập.

Mục tiêu: làm giàu entity substrate + fact đáp được cho lịch sử VN (bổ khuyết wiki-dump thưa/nhiễu,
vd ca "Triều Lý"→Korea). Nội dung AI-soạn từ kiến thức sử phổ thông — bám sự kiện/ngày lớn, ổn định.
Xuất `data/real_vi/history_vi.jsonl` cùng format statements_doc ({text, doc_id, para, title}).
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

ARTICLES: dict[str, list[str]] = {
    "Nhà Lý": [
        "Nhà Lý là một triều đại phong kiến trong lịch sử Việt Nam, tồn tại từ năm 1009 đến năm 1225.",
        "Nhà Lý do Lý Công Uẩn, tức Lý Thái Tổ, sáng lập vào năm 1009 sau khi lên ngôi thay nhà Tiền Lê.",
        "Năm 1010, triều Lý dời đô từ Hoa Lư về thành Đại La và đổi tên kinh đô thành Thăng Long.",
        "Năm 1054, dưới thời Lý Thánh Tông, nhà Lý đặt quốc hiệu là Đại Việt.",
        "Năm 1070, nhà Lý cho xây Văn Miếu ở Thăng Long để thờ Khổng Tử và các bậc hiền tài.",
        "Năm 1075, triều Lý mở khoa thi đầu tiên để tuyển chọn nhân tài cho đất nước.",
        "Năm 1076, nhà Lý lập Quốc Tử Giám, được xem là trường đại học đầu tiên của Việt Nam.",
        "Lý Thường Kiệt là danh tướng nhà Lý, đã lãnh đạo cuộc kháng chiến chống quân Tống xâm lược.",
        "Năm 1077, Lý Thường Kiệt đánh bại quân Tống trên phòng tuyến sông Như Nguyệt.",
        "Nhà Lý sùng đạo Phật và cho xây dựng nhiều chùa, trong đó có chùa Một Cột ở Thăng Long.",
        "Triều Lý trải qua chín đời vua và kéo dài hơn hai trăm năm.",
        "Nhà Lý kết thúc năm 1225 khi Lý Chiêu Hoàng nhường ngôi cho chồng là Trần Cảnh, mở ra nhà Trần.",
    ],
    "Lý Thái Tổ": [
        "Lý Thái Tổ tên thật là Lý Công Uẩn, là vị vua đầu tiên của nhà Lý.",
        "Lý Công Uẩn sinh năm 974, quê ở vùng Kinh Bắc, được nuôi dạy trong chùa từ nhỏ.",
        "Năm 1009, Lý Công Uẩn được các quan tôn lên làm vua, lập ra nhà Lý.",
        "Năm 1010, Lý Thái Tổ ban Chiếu dời đô, quyết định dời kinh đô về Đại La.",
        "Ông đổi tên Đại La thành Thăng Long, nghĩa là rồng bay lên.",
        "Lý Thái Tổ trị vì gần hai mươi năm và đặt nền móng vững chắc cho triều Lý.",
        "Ông mất năm 1028, con là Lý Thái Tông lên nối ngôi.",
    ],
    "Lý Thường Kiệt": [
        "Lý Thường Kiệt là một danh tướng và nhà quân sự lớn của nhà Lý.",
        "Ông sinh năm 1019 tại Thăng Long và phục vụ dưới ba đời vua nhà Lý.",
        "Năm 1075, Lý Thường Kiệt chủ động đem quân đánh phủ đầu vào đất Tống ở châu Ung, châu Khâm và châu Liêm.",
        "Năm 1077, ông chỉ huy quân Đại Việt đánh bại quân Tống trên sông Như Nguyệt.",
        "Bài thơ Nam quốc sơn hà được cho là gắn với Lý Thường Kiệt và được xem như bản tuyên ngôn độc lập đầu tiên.",
        "Lý Thường Kiệt còn có công đánh Chiêm Thành ở phía nam, giữ yên bờ cõi.",
        "Ông mất năm 1105, được sử sách ca ngợi là bậc danh tướng tài ba.",
    ],
    "Nhà Trần": [
        "Nhà Trần là triều đại phong kiến Việt Nam tồn tại từ năm 1225 đến năm 1400.",
        "Nhà Trần được lập nên khi Trần Cảnh, tức Trần Thái Tông, lên ngôi năm 1225.",
        "Trần Thủ Độ là người có công lớn trong việc đưa nhà Trần lên thay nhà Lý.",
        "Nhà Trần ba lần đánh bại quân Nguyên Mông vào các năm 1258, 1285 và 1288.",
        "Trần Hưng Đạo là vị tổng chỉ huy quân đội nhà Trần trong các cuộc kháng chiến chống Nguyên Mông.",
        "Năm 1288, quân dân nhà Trần đánh tan quân Nguyên trên sông Bạch Đằng.",
        "Trần Nhân Tông là vị vua sáng lập Thiền phái Trúc Lâm Yên Tử.",
        "Nhà Trần kết thúc năm 1400 khi Hồ Quý Ly cướp ngôi, lập ra nhà Hồ.",
    ],
    "Trần Hưng Đạo": [
        "Trần Hưng Đạo tên thật là Trần Quốc Tuấn, là danh tướng kiệt xuất của nhà Trần.",
        "Ông là tổng chỉ huy quân đội Đại Việt trong hai cuộc kháng chiến chống quân Nguyên Mông năm 1285 và 1288.",
        "Trần Hưng Đạo viết Hịch tướng sĩ để khích lệ tinh thần binh lính.",
        "Ông cũng soạn Binh thư yếu lược, một tác phẩm về nghệ thuật quân sự.",
        "Năm 1288, Trần Hưng Đạo chỉ huy trận Bạch Đằng, tiêu diệt đạo quân Nguyên do Ô Mã Nhi cầm đầu.",
        "Ông được nhân dân tôn kính và thờ phụng là Đức Thánh Trần.",
    ],
    "Nhà Hậu Lê": [
        "Nhà Hậu Lê là triều đại phong kiến Việt Nam do Lê Lợi sáng lập năm 1428.",
        "Nhà Hậu Lê ra đời sau khi cuộc khởi nghĩa Lam Sơn đánh đuổi quân Minh thành công.",
        "Lê Lợi lên ngôi lấy hiệu là Lê Thái Tổ và đặt kinh đô ở Thăng Long.",
        "Thời Lê Thánh Tông là giai đoạn cực thịnh của nhà Hậu Lê với bộ luật Hồng Đức.",
        "Nhà Hậu Lê chia làm hai giai đoạn là Lê sơ và Lê trung hưng.",
        "Nhà Hậu Lê tồn tại trên danh nghĩa đến năm 1789.",
    ],
    "Lê Lợi": [
        "Lê Lợi là người lãnh đạo cuộc khởi nghĩa Lam Sơn chống quân Minh.",
        "Ông phát động khởi nghĩa tại Lam Sơn, Thanh Hóa vào năm 1418.",
        "Sau mười năm chiến đấu gian khổ, nghĩa quân Lam Sơn giành thắng lợi năm 1427.",
        "Nguyễn Trãi là quân sư giúp Lê Lợi trong cuộc khởi nghĩa.",
        "Năm 1428, Lê Lợi lên ngôi hoàng đế, lập ra nhà Hậu Lê.",
        "Truyền thuyết kể rằng Lê Lợi trả gươm thần cho rùa vàng ở hồ Hoàn Kiếm.",
    ],
    "Nguyễn Trãi": [
        "Nguyễn Trãi là nhà chính trị, nhà văn hóa và anh hùng dân tộc thời Hậu Lê.",
        "Ông sinh năm 1380 và là con của Nguyễn Phi Khanh.",
        "Nguyễn Trãi là quân sư của Lê Lợi trong cuộc khởi nghĩa Lam Sơn.",
        "Ông viết Bình Ngô đại cáo, được coi là bản tuyên ngôn độc lập thứ hai của dân tộc.",
        "Nguyễn Trãi bị hàm oan trong vụ án Lệ Chi Viên và bị xử tử năm 1442.",
        "Năm 1980, Nguyễn Trãi được UNESCO công nhận là danh nhân văn hóa thế giới.",
    ],
    "Nhà Tây Sơn": [
        "Nhà Tây Sơn là triều đại tồn tại từ năm 1778 đến năm 1802.",
        "Phong trào Tây Sơn do ba anh em Nguyễn Nhạc, Nguyễn Huệ và Nguyễn Lữ khởi xướng.",
        "Nguyễn Huệ, tức vua Quang Trung, là thủ lĩnh quân sự tài ba nhất của Tây Sơn.",
        "Năm 1789, Quang Trung đại phá hai mươi chín vạn quân Thanh trong trận Ngọc Hồi Đống Đa.",
        "Nhà Tây Sơn có công thống nhất đất nước và đánh bại các thế lực ngoại xâm.",
        "Nhà Tây Sơn sụp đổ năm 1802 khi Nguyễn Ánh đánh bại và lập ra nhà Nguyễn.",
    ],
    "Quang Trung": [
        "Quang Trung tên thật là Nguyễn Huệ, là vị vua thứ hai của nhà Tây Sơn.",
        "Nguyễn Huệ nổi tiếng là một thiên tài quân sự trong lịch sử Việt Nam.",
        "Năm 1785, ông đánh tan quân Xiêm trong trận Rạch Gầm Xoài Mút.",
        "Năm 1786, Nguyễn Huệ đem quân ra Bắc lật đổ chúa Trịnh.",
        "Cuối năm 1788, ông lên ngôi hoàng đế lấy hiệu là Quang Trung.",
        "Đầu năm 1789, Quang Trung hành quân thần tốc ra Bắc và đại phá quân Thanh ở Ngọc Hồi Đống Đa.",
        "Vua Quang Trung mất năm 1792 khi mới ba mươi chín tuổi.",
    ],
    "Nhà Nguyễn": [
        "Nhà Nguyễn là triều đại phong kiến cuối cùng của Việt Nam, tồn tại từ năm 1802 đến năm 1945.",
        "Nhà Nguyễn do Nguyễn Ánh, tức vua Gia Long, sáng lập năm 1802.",
        "Gia Long đặt kinh đô ở Phú Xuân, tức thành phố Huế ngày nay.",
        "Năm 1804, quốc hiệu Việt Nam chính thức được đặt dưới thời nhà Nguyễn.",
        "Vua Minh Mạng là vị vua thứ hai, có nhiều cải cách hành chính.",
        "Từ năm 1858, thực dân Pháp bắt đầu xâm lược và dần biến Việt Nam thành thuộc địa.",
        "Nhà Nguyễn chấm dứt năm 1945 khi vua Bảo Đại thoái vị.",
    ],
    "Ngô Quyền": [
        "Ngô Quyền là vị vua sáng lập nhà Ngô, được xem là người mở đầu thời kỳ độc lập tự chủ.",
        "Năm 938, Ngô Quyền lãnh đạo quân dân đánh bại quân Nam Hán trên sông Bạch Đằng.",
        "Ông cho đóng cọc nhọn xuống lòng sông Bạch Đằng để đánh chìm thuyền giặc.",
        "Chiến thắng Bạch Đằng năm 938 chấm dứt hơn một nghìn năm Bắc thuộc.",
        "Năm 939, Ngô Quyền xưng vương và đóng đô ở Cổ Loa.",
    ],
    "Đinh Bộ Lĩnh": [
        "Đinh Bộ Lĩnh là người có công dẹp loạn mười hai sứ quân, thống nhất đất nước.",
        "Năm 968, ông lên ngôi hoàng đế, lấy hiệu là Đinh Tiên Hoàng.",
        "Đinh Bộ Lĩnh đặt quốc hiệu là Đại Cồ Việt và đóng đô ở Hoa Lư.",
        "Ông là vị hoàng đế đầu tiên của Việt Nam sau thời Bắc thuộc.",
        "Đinh Tiên Hoàng bị ám sát năm 979.",
    ],
    "Hai Bà Trưng": [
        "Hai Bà Trưng là Trưng Trắc và Trưng Nhị, hai nữ anh hùng của dân tộc Việt Nam.",
        "Năm 40, Hai Bà Trưng phát động khởi nghĩa chống ách đô hộ của nhà Hán.",
        "Cuộc khởi nghĩa giành thắng lợi và Trưng Trắc lên làm vua, đóng đô ở Mê Linh.",
        "Năm 43, quân Hán do Mã Viện chỉ huy đàn áp cuộc khởi nghĩa.",
        "Hai Bà Trưng được nhân dân tôn kính và lập đền thờ ở nhiều nơi.",
    ],
    "Văn Miếu Quốc Tử Giám": [
        "Văn Miếu Quốc Tử Giám là quần thể di tích lịch sử ở Hà Nội.",
        "Văn Miếu được xây dựng năm 1070 dưới thời vua Lý Thánh Tông để thờ Khổng Tử.",
        "Năm 1076, vua Lý Nhân Tông cho lập Quốc Tử Giám bên cạnh Văn Miếu.",
        "Quốc Tử Giám được coi là trường đại học đầu tiên của Việt Nam.",
        "Trong Văn Miếu có các bia tiến sĩ khắc tên những người đỗ đạt qua các khoa thi.",
        "Bia tiến sĩ ở Văn Miếu đã được UNESCO công nhận là di sản tư liệu thế giới.",
    ],
    "Hoàng thành Thăng Long": [
        "Hoàng thành Thăng Long là khu di tích trung tâm quyền lực của Việt Nam qua nhiều triều đại.",
        "Hoàng thành được xây dựng từ thời nhà Lý khi Lý Thái Tổ dời đô về Thăng Long năm 1010.",
        "Các triều đại Lý, Trần, Lê đều đặt trung tâm chính trị tại Hoàng thành Thăng Long.",
        "Năm 2010, Hoàng thành Thăng Long được UNESCO công nhận là di sản văn hóa thế giới.",
    ],
    "Cố đô Hoa Lư": [
        "Hoa Lư là kinh đô đầu tiên của nhà nước Đại Cồ Việt.",
        "Hoa Lư nằm ở tỉnh Ninh Bình ngày nay.",
        "Đinh Bộ Lĩnh chọn Hoa Lư làm kinh đô vào năm 968.",
        "Các triều Đinh và Tiền Lê đều đóng đô ở Hoa Lư.",
        "Năm 1010, nhà Lý dời đô từ Hoa Lư về Thăng Long.",
    ],
    "Chiến thắng Bạch Đằng": [
        "Sông Bạch Đằng là nơi diễn ra ba trận thủy chiến lớn trong lịch sử Việt Nam.",
        "Năm 938, Ngô Quyền đánh bại quân Nam Hán trên sông Bạch Đằng.",
        "Năm 981, Lê Hoàn đánh bại quân Tống trên sông Bạch Đằng.",
        "Năm 1288, Trần Hưng Đạo tiêu diệt quân Nguyên Mông trên sông Bạch Đằng.",
        "Chiến thuật đóng cọc gỗ nhọn xuống lòng sông được dùng để phá thuyền giặc.",
    ],
    "Khởi nghĩa Lam Sơn": [
        "Khởi nghĩa Lam Sơn là cuộc khởi nghĩa chống quân Minh do Lê Lợi lãnh đạo.",
        "Cuộc khởi nghĩa bắt đầu năm 1418 tại Lam Sơn, Thanh Hóa.",
        "Nguyễn Trãi là quân sư quan trọng của nghĩa quân Lam Sơn.",
        "Năm 1427, nghĩa quân đánh tan viện binh quân Minh trong trận Chi Lăng Xương Giang.",
        "Cuộc khởi nghĩa thắng lợi năm 1428, mở ra nhà Hậu Lê.",
    ],
    "Trận Ngọc Hồi Đống Đa": [
        "Trận Ngọc Hồi Đống Đa là chiến thắng của quân Tây Sơn trước quân Thanh.",
        "Trận đánh diễn ra vào đầu xuân năm Kỷ Dậu 1789.",
        "Vua Quang Trung chỉ huy quân Tây Sơn hành quân thần tốc ra Bắc.",
        "Quân Tây Sơn đánh tan hai mươi chín vạn quân Thanh do Tôn Sĩ Nghị chỉ huy.",
        "Chiến thắng này giải phóng Thăng Long và bảo vệ nền độc lập của đất nước.",
    ],
    "Nhà Đinh": [
        "Nhà Đinh là triều đại do Đinh Bộ Lĩnh sáng lập năm 968.",
        "Nhà Đinh đặt quốc hiệu là Đại Cồ Việt và đóng đô ở Hoa Lư.",
        "Đinh Tiên Hoàng là vị vua đầu tiên của nhà Đinh.",
        "Nhà Đinh tồn tại ngắn ngủi và kết thúc năm 980.",
        "Sau nhà Đinh, Lê Hoàn lên ngôi lập ra nhà Tiền Lê.",
    ],
    "Nhà Tiền Lê": [
        "Nhà Tiền Lê là triều đại do Lê Hoàn sáng lập năm 980.",
        "Lê Hoàn lên ngôi lấy hiệu là Lê Đại Hành.",
        "Năm 981, Lê Đại Hành đánh bại quân Tống xâm lược.",
        "Nhà Tiền Lê tiếp tục đóng đô ở Hoa Lư.",
        "Nhà Tiền Lê kết thúc năm 1009 khi Lý Công Uẩn lên ngôi lập nhà Lý.",
    ],
    "Lê Thánh Tông": [
        "Lê Thánh Tông là vị vua thứ năm của nhà Hậu Lê, trị vì từ năm 1460 đến 1497.",
        "Thời Lê Thánh Tông được coi là giai đoạn thịnh trị nhất của nhà Hậu Lê.",
        "Ông ban hành bộ luật Hồng Đức, một bộ luật tiến bộ của thời phong kiến.",
        "Lê Thánh Tông mở mang bờ cõi về phía nam.",
        "Ông lập hội Tao Đàn, khuyến khích văn học và giáo dục.",
    ],
    "Trần Nhân Tông": [
        "Trần Nhân Tông là vị vua thứ ba của nhà Trần.",
        "Ông trực tiếp lãnh đạo hai cuộc kháng chiến chống quân Nguyên Mông năm 1285 và 1288.",
        "Sau khi nhường ngôi, ông xuất gia tu hành ở núi Yên Tử.",
        "Trần Nhân Tông sáng lập Thiền phái Trúc Lâm, một dòng thiền của người Việt.",
        "Ông được tôn là Phật hoàng Trần Nhân Tông.",
    ],
    "Lý Thánh Tông": [
        "Lý Thánh Tông là vị vua thứ ba của nhà Lý.",
        "Năm 1054, ông đặt quốc hiệu là Đại Việt.",
        "Năm 1070, Lý Thánh Tông cho xây Văn Miếu ở Thăng Long.",
        "Ông là vị vua có công phát triển Nho học và mở mang bờ cõi.",
        "Lý Thánh Tông trị vì từ năm 1054 đến năm 1072.",
    ],
    "Bà Triệu": [
        "Bà Triệu tên thật là Triệu Thị Trinh, là nữ anh hùng chống quân Ngô.",
        "Năm 248, Bà Triệu phát động khởi nghĩa ở vùng Thanh Hóa.",
        "Bà nổi tiếng với câu nói thể hiện chí khí không chịu khuất phục quân giặc.",
        "Cuộc khởi nghĩa của Bà Triệu tuy thất bại nhưng để lại tiếng vang lớn.",
        "Bà Triệu được nhân dân lập đền thờ và tôn kính.",
    ],
    "Chùa Một Cột": [
        "Chùa Một Cột là ngôi chùa nổi tiếng ở Hà Nội.",
        "Chùa được xây dựng năm 1049 dưới thời vua Lý Thái Tông.",
        "Chùa có kiến trúc độc đáo với một cột đá đỡ toàn bộ ngôi chùa như bông sen nở trên mặt nước.",
        "Chùa Một Cột là biểu tượng văn hóa của thủ đô Hà Nội.",
    ],
    "Chiếu dời đô": [
        "Chiếu dời đô là bài chiếu do vua Lý Thái Tổ ban hành năm 1010.",
        "Bài chiếu nêu lý do dời kinh đô từ Hoa Lư về thành Đại La.",
        "Lý Thái Tổ cho rằng Đại La là nơi trung tâm, đất rộng và bằng phẳng, thuận lợi cho muôn dân.",
        "Sau khi dời đô, Đại La được đổi tên thành Thăng Long.",
        "Chiếu dời đô được coi là một áng văn chính luận có giá trị lịch sử lớn.",
    ],
    "Kháng chiến chống Nguyên Mông": [
        "Kháng chiến chống Nguyên Mông là ba cuộc chiến của quân dân nhà Trần chống đế quốc Mông Nguyên.",
        "Cuộc kháng chiến lần thứ nhất diễn ra năm 1258.",
        "Cuộc kháng chiến lần thứ hai diễn ra năm 1285.",
        "Cuộc kháng chiến lần thứ ba diễn ra năm 1288 với chiến thắng Bạch Đằng.",
        "Trần Hưng Đạo là tổng chỉ huy trong hai cuộc kháng chiến sau.",
        "Nhà Trần tổ chức hội nghị Diên Hồng để hỏi ý kiến các bô lão về việc đánh giặc.",
    ],
    "Nhà Hồ": [
        "Nhà Hồ là triều đại do Hồ Quý Ly sáng lập năm 1400.",
        "Hồ Quý Ly cướp ngôi nhà Trần để lập ra nhà Hồ.",
        "Nhà Hồ đổi quốc hiệu thành Đại Ngu và dời đô về Thành Tây Đô ở Thanh Hóa.",
        "Hồ Quý Ly thực hiện nhiều cải cách về tiền tệ và ruộng đất.",
        "Năm 1407, nhà Hồ sụp đổ trước cuộc xâm lược của quân Minh.",
    ],
}


def main() -> None:
    out = _ROOT / "data" / "real_vi" / "history_vi.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for i, (title, sents) in enumerate(ARTICLES.items()):
            for para, s in enumerate(sents):
                f.write(json.dumps(
                    {"text": s, "doc_id": f"hist:{i}", "para": para, "title": title},
                    ensure_ascii=False,
                ) + "\n")
                n += 1
    print(f"XONG: {len(ARTICLES)} bài, {n} câu → {out}")


if __name__ == "__main__":
    main()
