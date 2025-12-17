from inexus_parser import AddressParser
import argparse
import json
import re
import time
from pathlib import Path

parser = AddressParser()

if __name__ == "__main__":
    tests = [
        {
            "mst_address": "320/1 Quốc Lộ 1A, Ấp Bình Cang 1, Xã Bình Thạnh, Huyện Thủ Thừa, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Khu Vực 4, Thị Trấn Đức Hòa, Huyện Đức Hoà, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "P706 - nhà B, chung cư An Sinh, Thị Trấn Cầu Diễn, Huyện Từ Liêm, Thành phố Hà Nội, Việt Nam"
        },
        {
            "mst_address": "39 Đinh Tiên Hoàng, Thị Trấn Cam Đức, Huyện Cam Lâm, Tỉnh Khánh Hòa, Việt Nam"
        },
        {
            "mst_address": "Thôn Trường Thọ Tây, Thị Trấn Sơn Tịnh, Huyện Sơn Tịnh, Tỉnh Quảng Ngãi, Việt Nam"
        },
        {
            "mst_address": "Số 79 Nguyễn Văn Tiếp, Khu phố 3, Thị Trấn Bến Lức, Huyện Bến Lức, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Thôn Chánh Hoá, Xã Cát Thành, Huyện Phù Cát, Tỉnh Bình Định, Việt Nam"
        },
        {
            "mst_address": "xóm Tân Thành, Xã Bảo Cường, Huyện Định Hoá, Tỉnh Thái Nguyên, Việt Nam"
        },
        {
            "mst_address": "Thôn 1, Xã Đồng Trạch, Huyện Bố Trạch, Tỉnh Quảng Bình, Việt Nam"
        },
        {
            "mst_address": "Số 504 Quốc lộ 20, Thị Trấn Liên Nghĩa, Huyện Đức Trọng, Tỉnh Lâm Đồng, Việt Nam"
        },
        {
            "mst_address": "Tổ 03, Ấp Tân Thành, Xã Tân Lập, Huyện Tịnh Biên, Tỉnh An Giang, Việt Nam"
        },
        {
            "mst_address": "Thôn Quang Biểu, Xã Quang Châu, Huyện Việt Yên, Tỉnh Bắc Giang, Việt Nam"
        },
        {
            "mst_address": "208 Tỉnh lộ 8, Thị Trấn Củ Chi, Huyện Củ Chi, Thành phố Hồ Chí Minh, Việt Nam"
        },
        {
            "mst_address": "Xóm Bắc Hợp, Thôn Duệ Đông, Thị Trấn Lim, Huyện Tiên Du, Tỉnh Bắc Ninh, Việt Nam"
        },
        {
            "mst_address": "Thị Trấn Yên Bình, Huyện Yên Bình, Tỉnh Yên Bái, Việt Nam"
        },
        {
            "mst_address": "Thôn Phụng Sơn, Xã Phước Sơn, Huyện Tuy Phước, Tỉnh Bình Định, Việt Nam"
        },
        {
            "mst_address": "Tổ dân phố 2 , Thị Trấn Trà Xuân, Huyện Trà Bồng, Tỉnh Quảng Ngãi, Việt Nam"
        },
        {
            "mst_address": "Tổ dân phố Trạch Thượng 1, Thị Trấn Phong Điền, Huyện Phong Điền, Tỉnh Thừa Thiên Huế, Việt Nam"
        },
        {
            "mst_address": "Xóm 3, Thôn Đề Hạ, Thị Trấn Kim Bài, Huyện Thanh Oai, Thành phố Hà Nội, Việt Nam"
        },
        {
            "mst_address": "91, Ấp 8, Xã Tân Phước, Huyện Gò Công Đông, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ 4, ấp Rẩy Mới, Xã Bình An, Huyện Kiên Lương, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Số 258, Tổ 8, Ấp Bình Hòa A, Xã Tam Bình, Huyện Cai Lậy, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Số 99, tổ 3, ấp Phú Hội, Xã Tân Hội, Huyện Tân Hiệp, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Ấp Cá, Thị Trấn Tân Hiệp, Huyện Châu Thành, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Thôn 1, Thị Trấn Buôn Trấp, Huyện Krông Ana, Tỉnh Đắk Lắk, Việt Nam"
        },
        {
            "mst_address": "Thôn Mỹ Quế, Xã Gia Tường, Huyện Nho quan, Tỉnh Ninh Bình, Việt Nam"
        },
        {
            "mst_address": "., Thị trấn Nghi Xuân, Huyện Nghi Xuân, Hà Tĩnh"
        },
        {
            "mst_address": "Số 427, Tổ 23, Khu 2, Thị Trấn Cái Bè, Huyện Cái Bè, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Thôn Trại (tại nhà ông Nguyễn Văn Hà), Xã Thuỷ Đường, Huyện Thuỷ Nguyên, Thành phố Hải Phòng, Việt Nam"
        },
        {
            "mst_address": "Thôn Đồng Phú, Xã Kỳ Đồng, Huyện Kỳ Anh, Hà Tĩnh"
        },
        {
            "mst_address": "Quốc lộ 50 Ấp Thạnh Yên, Xã Thạnh Trị, Huyện Gò Công Tây, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Khối 3, Thị Trấn Yên Thành, Huyện Yên Thành, Tỉnh Nghệ An, Việt Nam"
        },
        {
            "mst_address": "Số 2400 đường Trần Văn Trà, ấp Thị Cầu, Xã Phú Đông, Huyện Nhơn Trạch, Đồng Nai"
        },
        {
            "mst_address": "Khu phố 3 - Thị trấn Yên ninh, , Huyện Yên Khánh, Ninh Bình"
        },
        {
            "mst_address": "Phố Nam Giang, Thị Trấn Nho Quan, Huyện Nho quan, Tỉnh Ninh Bình, Việt Nam"
        },
        {
            "mst_address": "Tiểu khu 2, thị trấn Mộc châu, , Huyện Mộc Châu, Sơn La"
        },
        {
            "mst_address": "TDP Cây Châm, Thị Trấn Đu, Huyện Phú Lương, Tỉnh Thái Nguyên, Việt Nam"
        },
        {
            "mst_address": "11 TL835B, Ấp Phú Ân\t, Xã Phước Lý, Huyện Cần Giuộc, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Số 256 Quốc lộ 50, khu phố 4, Thị Trấn Cần Giuộc, Huyện Cần Giuộc, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Số 126 Đào Duy Từ, Thị Trấn Liên Nghĩa, Huyện Đức Trọng, Tỉnh Lâm Đồng, Việt Nam"
        },
        {
            "mst_address": "Số 18, Tôn Đức Thắng, Ấp Rạch Bùi , Thị Trấn Vĩnh Hưng, Huyện Vĩnh Hưng, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Tổ dân phố Viễn Trình, Thị Trấn Phú Đa, Huyện Phú Vang, Tỉnh Thừa Thiên Huế, Việt Nam"
        },
        {
            "mst_address": "Số 201 Ô 1, Khu 3, Thị Trấn Chợ Gạo, Huyện Chợ Gạo, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Số 34, tổ 18, khu phố Lò Bom, Thị Trấn Kiên Lương, Huyện Kiên Lương, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Cụm 2, TDP Hòa Sơn, Thị trấn Chúc Sơn, Huyện Chương Mỹ, Hà Nội"
        },
        {
            "mst_address": "Ấp Mỹ Lợi, Xã Mỹ Long, Huyện Cai Lậy, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "8-10-12/D Nguyễn Văn Tre, Khóm 4, Thị Trấn Mỹ An, Huyện Tháp Mười, Tỉnh Đồng Tháp, Việt Nam"
        },
        {
            "mst_address": "Ấp Hòa Quí, Xã Hòa Khánh, Huyện Cái Bè, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Đường Lê Thị Khuông, Thị Trấn Phù Mỹ, Huyện Phù Mỹ, Tỉnh Bình Định, Việt Nam"
        },
        {
            "mst_address": "Cụm công nghiệp Bình Dương, Thị Trấn Bình Dương, Huyện Phù Mỹ, Tỉnh Bình Định, Việt Nam"
        },
        {
            "mst_address": "Thôn Trung Tín 1, Thị Trấn Tuy Phước, Huyện Tuy Phước, Tỉnh Bình Định, Việt Nam"
        },
        {
            "mst_address": "Thôn Phả Lại (NR: Ông Nguyễn Văn Nghề), Xã Đức Long, Huyện Quế Võ, Tỉnh Bắc Ninh, Việt Nam"
        },
        {
            "mst_address": "ấp Vĩnh Thành, thị trấn Cái Dầu, Thị trấn Cái Dầu, Huyện Châu Phú, An Giang"
        },
        {
            "mst_address": "Lô C07 (07-06), khu công nghiệp Tịnh Phong, , Huyện Sơn Tịnh, Quảng Ngãi"
        },
        {
            "mst_address": "Tổ 13, ấp Bình Hòa, Xã Bình Giang, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Đường số 8, KCN Nhơn Trạch II-Nhơn Phú, Xã Phú Hội, Huyện Nhơn Trạch, Đồng Nai"
        },
        {
            "mst_address": "Số 151 đường Đình Thôn, Xã Mỹ Đình, Huyện Từ Liêm, Thành phố Hà Nội, Việt Nam"
        },
        {
            "mst_address": "Đường DT20, Thôn Xuân Tiến, Thị Trấn Phong Nha, Huyện Bố Trạch, Tỉnh Quảng Bình, Việt Nam"
        },
        {
            "mst_address": "Cụm làng nghề, Thị Trấn Ái Tử, Huyện Triệu Phong, Tỉnh Quảng Trị, Việt Nam"
        },
        {
            "mst_address": "Số 21B đường Trần Phú, Tổ dân phố 2, Thị Trấn Phước An, Huyện Krông Pắc, Tỉnh Đắk Lắk, Việt Nam"
        },
        {
            "mst_address": "UBND Thị Trấn Cành Nàng, , Huyện Bá Thước, Thanh Hoá"
        },
        {
            "mst_address": "Lô N4, Đường số 6, Khu Công Nghiệp Phúc Long, Xã Long Hiệp, Huyện Bến Lức, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Đông Sơn 2, Thị trấn Núi Sập, Huyện Thoại Sơn, An Giang"
        },
        {
            "mst_address": "Tổ dân phố 5, Thị Trấn Krông Kmar, Huyện Krông Bông, Tỉnh Đắk Lắk, Việt Nam"
        },
        {
            "mst_address": "Khu phố Rạch Bùi, Thị Trấn Vĩnh Hưng, Huyện Vĩnh Hưng, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Khu phố Rọc Bùi, Thị Trấn Vĩnh Hưng, Huyện Vĩnh Hưng, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Số 81, Khu Bắc Sơn, Thị Trấn Chúc Sơn, Huyện Chương Mỹ, Thành phố Hà Nội, Việt Nam"
        },
        {
            "mst_address": "Thôn Nguyệt Áng, Xã Tân Ninh, Huyện Quảng Ninh, Tỉnh Quảng Bình, Việt Nam"
        },
        {
            "mst_address": "Thị trấn Chợ Chu, , Huyện Định Hoá, Thái Nguyên"
        },
        {
            "mst_address": "81 Đinh Tiên Hoàng, Thị trấn Vạn Giã, Huyện Vạn Ninh, Khánh Hòa"
        },
        {
            "mst_address": "Khu 5, Thị Trấn Phố Mới, Huyện Quế Võ, Tỉnh Bắc Ninh, Việt Nam"
        },
        {
            "mst_address": "Ấp Phú Quí, Xã Vĩnh Hựu, Huyện Gò Công Tây, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Thôn Dực Liễn, Xã Thuỷ Sơn, Huyện Thuỷ Nguyên, Thành phố Hải Phòng, Việt Nam"
        },
        {
            "mst_address": "Tổ 5, ấp Thành Trung, Xã Đông Thái, Huyện An Biên, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Khối 6, Thị Trấn Kim Sơn, Huyện Quế Phong, Tỉnh Nghệ An, Việt Nam"
        },
        {
            "mst_address": "Đuờng 25B, Tổ 4, Ấp Đất Mới, Xã Phú Hội, Huyện Nhơn Trạch, Tỉnh Đồng Nai, Việt Nam"
        },
        {
            "mst_address": "Xuân Phương, Xã Phước Sơn, Huyện Tuy Phước, Tỉnh Bình Định, Việt Nam"
        },
        {
            "mst_address": "Số nhà 74, Tổ dân phố Vĩnh Giang, Thị Trấn Vĩnh Lộc, Huyện Chiêm Hoá, Tỉnh Tuyên Quang, Việt Nam"
        },
        {
            "mst_address": "Tổ Trung Tâm 1, Thị trấn Vĩnh Lộc, Huyện Chiêm Hoá, Tuyên Quang"
        },
        {
            "mst_address": "ấp An Biên, Xã An Nông, Huyện Tịnh Biên, Tỉnh An Giang, Việt Nam"
        },
        {
            "mst_address": "Ấp Trung, Xã Tân Hòa, Huyện Tân Thạnh, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Thôn Ngò, Xã Đồng Kỳ, Huyện Yên Thế, Tỉnh Bắc Giang, Việt Nam"
        },
        {
            "mst_address": "53 Ngô Gia Tự, tổ dân phố 9, Thị Trấn Vạn Giã, Huyện Vạn Ninh, Tỉnh Khánh Hòa, Việt Nam"
        },
        {
            "mst_address": "Đường Hai Bà Trung, Thị trấn Tô Hạp, Huyện Khánh Sơn, Khánh Hòa"
        },
        {
            "mst_address": "Khu 6, Thị Trấn Trạm Trôi, Huyện Hoài Đức, Thành phố Hà Nội, Việt Nam"
        },
        {
            "mst_address": "Thôn An Tân, Xã Hoà Phong, Huyện Hoà Vang, Thành phố Đà Nẵng, Việt Nam"
        },
        {
            "mst_address": "Tổ dân phố Tân Tiến, Thị trấn Tân Yên, Huyện Hàm Yên, Tuyên Quang"
        },
        {
            "mst_address": "Khu công nghiệp phía Nam, Xã Phú Thịnh, Huyện Yên Bình, Tỉnh Yên Bái, Việt Nam"
        },
        {
            "mst_address": "Số 24 Dương Đình Nghệ, Thị Trấn Liên Nghĩa, Huyện Đức Trọng, Tỉnh Lâm Đồng, Việt Nam"
        },
        {
            "mst_address": "Đường Tôn Đức Thắng, Tổ 14B, Ấp Xóm Hố, Xã Phú Hội, Huyện Nhơn Trạch, Tỉnh Đồng Nai, Việt Nam"
        },
        {
            "mst_address": "Số 19, Lê Văn Tám, Khu phố 6, Thị Trấn Liên Nghĩa, Huyện Đức Trọng, Tỉnh Lâm Đồng, Việt Nam"
        },
        {
            "mst_address": "Cả Đá, Xã Tân Thành, Huyện Mộc Hoá, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "TT Hát Lót, , Huyện Mai Sơn, Sơn La"
        },
        {
            "mst_address": "Bản Nà Tiến, Xã Hát Lót, Huyện Mai Sơn, Sơn La"
        },
        {
            "mst_address": "Thửa Đất Số 699, Tờ bản đồ số 14, Xã Mỹ Yên, Huyện Bến Lức, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "27 Tổ 1, Ấp 4, Xã Tân Hưng, Huyện Cái Bè, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Số 23 đường Lê Duẩn, Thị Trấn Phước An, Huyện Krông Pắc, Tỉnh Đắk Lắk, Việt Nam"
        },
        {
            "mst_address": "Số 3, Khu vưc 5, Thị Trấn Đức Hòa, Huyện Đức Hoà, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Số 04, Ấp Xóm Mới, Xã Tân Lân, Huyện Cần Đước, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Tổ 3, ấp 10 Huỳnh, Xã Đông Hưng, Huyện An Minh, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Tầng 2, số nhà 21, ngõ 344 đường mùng 10 tháng 6, khu 5, Thị Trấn Quán Lào, Huyện Yên Định, Tỉnh Thanh Hoá, Việt Nam"
        },
        {
            "mst_address": "6A, Nguyễn Du, Thị trấn Thạnh Mỹ, Huyện Đơn Dương, Lâm Đồng"
        },
        {
            "mst_address": "Bản Nà Tòng, Thị Trấn Ít Ong, Huyện Mường La, Tỉnh Sơn La, Việt Nam"
        },
        {
            "mst_address": "Số 578, Tổ 17, Áp Tân Thuận A, Xã Bình Đức, Huyện Châu Thành, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Số 16 đường 198, Thị Trấn Chờ, Huyện Yên Phong, Tỉnh Bắc Ninh, Việt Nam"
        },
        {
            "mst_address": "Số 16, Ngõ B8 Đường Kiên Thành, Thị Trấn Trâu Quỳ, Huyện Gia Lâm, Thành phố Hà Nội, Việt Nam"
        },
        {
            "mst_address": "Thị trấn Nước hai - Hoà An, , Huyện Hoà An, Cao Bằng"
        },
        {
            "mst_address": "Khu Dã Hương 1, Thị Trấn Nước Hai, Huyện Hoà An, Tỉnh Cao Bằng, Việt Nam"
        },
        {
            "mst_address": "Số 140, Đường 838, Khu phố 4, Thị Trấn Đông Thành, Huyện Đức Huệ, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Thị Trấn Thanh Nhật, , Huyện Hạ Lang, Cao Bằng"
        },
        {
            "mst_address": "Thôn Thủy An, Xã Thuỷ Đường, Huyện Thuỷ Nguyên, Thành phố Hải Phòng, Việt Nam"
        },
        {
            "mst_address": "Khối 10, Thị Trấn Cầu Giát, Huyện Quỳnh Lưu, Tỉnh Nghệ An, Việt Nam"
        },
        {
            "mst_address": "Tổ 3, ấp Phước Hưng 1, Thị Trấn Gò Quao, Huyện Gò Quao, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "C15, Đường 4, Khu xưởng Kizuna 2, Lô B2-9-1-10, KCN Tân Kim, Thị Trấn Cần Giuộc, Huyện Cần Giuộc, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Tổ 5, ấp Hòn Sóc, Xã Thổ Sơn, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "130/1, Ấp Thuận Đạo, Thị Trấn Bến Lức, Huyện Bến Lức, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Tân Dân, thị trấn Thứa, , Huyện Lương Tài, Bắc Ninh"
        },
        {
            "mst_address": "Khối 2, thị trấn Quỳ Châu, , Huyện Quỳ Châu, Nghệ An"
        },
        {
            "mst_address": "Số 396 Quốc lộ 20, Thị Trấn Liên Nghĩa, Huyện Đức Trọng, Tỉnh Lâm Đồng, Việt Nam"
        },
        {
            "mst_address": "Xóm Nội, Thị Trấn Chúc Sơn, Huyện Chương Mỹ, Thành phố Hà Nội, Việt Nam"
        },
        {
            "mst_address": "Số 08, Đường Thoại Ngọc Hầu, Khóm Nam Sơn, Thị Trấn Núi Sập, Huyện Thoại Sơn, Tỉnh An Giang, Việt Nam"
        },
        {
            "mst_address": "Số 68/2 đường Nguyễn Văn Cừ, Thị Trấn Liên Nghĩa, Huyện Đức Trọng, Tỉnh Lâm Đồng, Việt Nam"
        },
        {
            "mst_address": "Khu Dịch vụ Cảng Cá, Khóm 11, Thị Trấn Sông Đốc, Huyện Trần Văn Thời, Tỉnh Cà Mau, Việt Nam"
        },
        {
            "mst_address": "Khu dân cư 3A2, thôn Liên Hiệp I, Thị Trấn Sơn Tịnh, Huyện Sơn Tịnh, Tỉnh Quảng Ngãi, Việt Nam"
        },
        {
            "mst_address": "Nhà ông Vũ Nguyên Khải, Quán Dốc, Xã Triệu Lộc, Huyện Hậu Lộc, Tỉnh Thanh Hoá, Việt Nam"
        },
        {
            "mst_address": "Thôn Phú Vinh Trung, Thị Trấn Chợ Chùa, Huyện Nghĩa Hành, Tỉnh Quảng Ngãi, Việt Nam"
        },
        {
            "mst_address": "Số 85, tổ 1, khu phố Đầu Voi, Thị Trấn Hòn Đất, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Xã Đề Thám, , Huyện Tràng Định, Lạng Sơn"
        },
        {
            "mst_address": "Khu Phố Lúng, Thị Trấn Yên Cát, Huyện Như Xuân, Tỉnh Thanh Hoá, Việt Nam"
        },
        {
            "mst_address": "Khu Công Nghiệp Nhơn Trạch II, Ấp Xóm Hố, Xã Phú Hội, Huyện Nhơn Trạch, Tỉnh Đồng Nai, Việt Nam"
        },
        {
            "mst_address": "Khối 4, Thị trấn La Hà, , Huyện Tư Nghĩa, Quảng Ngãi"
        },
        {
            "mst_address": "Thôn Hồng Hà, Xã Nga Quán, Huyện Trấn Yên, Yên Bái"
        },
        {
            "mst_address": "TDP Tân Hải, Thị trấn Cam Đức, Huyện Cam Lâm, Khánh Hòa"
        },
        {
            "mst_address": "KDC Đông Nam thị trấn Châu Ổ, Thị Trấn Châu Ổ, Huyện Bình Sơn, Tỉnh Quảng Ngãi, Việt Nam"
        },
        {
            "mst_address": "Đường Khuất Quang Chiến, Thị trấn Phố Lu, Huyện Bảo Thắng, Lào Cai"
        },
        {
            "mst_address": "Tổ 1, ấp An Phước, Xã Bình An, Huyện Châu Thành, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ 1, ấp Bình Lợi, Xã Minh Hòa, Huyện Châu Thành, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ 21, ấp Thạnh Hiệp, Xã Thạnh Lộc, Huyện Giồng Riềng, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ 9, ấp Phước Trung I, Thị Trấn Gò Quao, Huyện Gò Quao, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "UBND xã Phong Thu, , Huyện Phong Điền, Thừa Thiên - Huế"
        },
        {
            "mst_address": "Nhà ông Hà Trọng Sơn, tiểu khu Phúc Sơn, Thị Trấn Bút Sơn, Huyện Hoằng Hoá, Tỉnh Thanh Hoá, Việt Nam"
        },
        {
            "mst_address": "Thôn Nam Lãnh, Xã Quảng Phú, Huyện Quảng Trạch, Tỉnh Quảng Bình, Việt Nam"
        },
        {
            "mst_address": "Số 22, khu phố Hiệp Thương, Thị Trấn Định Quán, Huyện Định Quán, Tỉnh Đồng Nai, Việt Nam"
        },
        {
            "mst_address": "Số 43 ngõ 21, phố Nguyễn Khiêm Ích, khu 31ha, Thị Trấn Trâu Quỳ, Huyện Gia Lâm, Thành phố Hà Nội, Việt Nam"
        },
        {
            "mst_address": "Số 13, khu phố Đông An, Thị Trấn Tân Hiệp, Huyện Tân Hiệp, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ 7, ấp An Ninh, Xã Bình An, Huyện Châu Thành, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Thửa đất số 1082 + 1083 tờ bản đồ số 3 tổ 2 ấp An Phước, Xã Bình An, Huyện Châu Thành, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Đông Sơn 2, TT Núi Sập, , Huyện Thoại Sơn, An Giang"
        },
        {
            "mst_address": "Số 648 khu phố Thị Tứ, Thị Trấn Sóc Sơn, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ 12, ấp số 8, Xã Sơn Kiên, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Phố 1 - Thị trấn Cành Nàng, , Huyện Bá Thước, Thanh Hoá"
        },
        {
            "mst_address": "Số 226, ấp Ranh Hạt, Xã Bình Giang, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Số 117, ấp Bến Đá, Xã Thổ Sơn, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Khu phố Cá, Thị Trấn Tân Hiệp, Huyện Châu Thành, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "152 Đường Huyện Lộ 18, Tổ 4,Ấp Thạnh Phú, Xã Đồng Thạnh, Huyện Gò Công Tây, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Số 10, tổ 10, khu phố Đường Hòn, Thị Trấn Hòn Đất, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ 9, khu phố Đường Hòn, Thị Trấn Hòn Đất, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ 11, Khu phố Đầu Doi, Thị Trấn Hòn Đất, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Nhà ông Nguyễn Thanh Sơn Đường 19/5, Thị Trấn Phố Lu, Huyện Bảo Thắng, Tỉnh Lào Cai, Việt Nam"
        },
        {
            "mst_address": "Số 50, Ấp Tân Thuận, Xã Bình Đức, Huyện Châu Thành, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Số nhà 18, lô 7, trung tâm thương mại Hòn Đất, khu phố Tri Tôn, Thị Trấn Hòn Đất, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ 5, khu phố Tri Tôn, Thị Trấn Hòn Đất, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Số 161, tổ 4, ấp Vĩnh Thành, Xã Vĩnh Hòa, Huyện U Minh Thượng, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "TK 4, Thị trấn Hát Lót, Huyện Mai Sơn, Sơn La"
        },
        {
            "mst_address": "thôn 5, Xã Thuỷ Sơn, Huyện Thuỷ Nguyên, Hải Phòng"
        },
        {
            "mst_address": "Đường Lê Quý Đôn, khóm 1, Thị trấn Mỹ An, Huyện Tháp Mười, Đồng Tháp"
        },
        {
            "mst_address": "Số 151A, ấp Hưng Giang, Xã Mỹ Lâm, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "ấp An Hoà, Xã An Cư, Huyện Cái Bè, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Tổ dân phố 3, Thị Trấn La Hà, Huyện Tư Nghĩa, Tỉnh Quảng Ngãi, Việt Nam"
        },
        {
            "mst_address": "Ấp Mỹ Lộc, Xã Thạnh Mỹ, Huyện Tân Phước, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "266-268, quốc lộ 1A, khu phố 9, Thị Trấn Bến Lức, Huyện Bến Lức, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Ấp An Phú, Xã An Thạnh Thủy, Huyện Chợ Gạo, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "Khu phố 2 - TT Madaguôi, Thị trấn Ma Đa Guôi, Huyện Đạ Huoai, Lâm Đồng"
        },
        {
            "mst_address": "31A Tổ 1, Ấp Phước Lý, Xã Phước Lý, Huyện Cần Giuộc, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Số 345, Ấp Xẻo Nhàu A, Xã Tân Thạnh, Huyện An Minh, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Số 17, ấp Tàu Hơi A, Xã Thạnh Trị, Huyện Tân Hiệp, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "34 Xuân Diệu, Thị Trấn Cam Đức, Huyện Cam Lâm, Tỉnh Khánh Hòa, Việt Nam"
        },
        {
            "mst_address": "Khu 9, Thị Trấn Cái Rồng, Huyện Vân Đồn, Tỉnh Quảng Ninh, Việt Nam"
        },
        {
            "mst_address": "Khu 5, thị trấn Cái Rồng, Thị trấn Cái Rồng, Huyện Vân Đồn, Quảng Ninh"
        },
        {
            "mst_address": "80 Trần Nhân Tông, Thị Trấn Liên Nghĩa, Huyện Đức Trọng, Tỉnh Lâm Đồng, Việt Nam"
        },
        {
            "mst_address": "Phố Hoằng Bó , Thị Trấn Nước Hai, Huyện Hoà An, Tỉnh Cao Bằng, Việt Nam"
        },
        {
            "mst_address": "Số 628, khu phố Ngã Ba, Thị Trấn Kiên Lương, Huyện Kiên Lương, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Thửa đất số 477, tờ bản đồ số 1, ấp Tây Phú, Xã Long Phụng, Huyện Cần Giuộc, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Số 05, khu vực 4, Thị Trấn Giồng Riềng, Huyện Giồng Riềng, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Số 401 Khu 3, Thị Trấn Cái Bè, Huyện Cái Bè, Tỉnh Tiền Giang, Việt Nam"
        },
        {
            "mst_address": "026 Tân Chánh, Xã Tân Tập, Huyện Cần Giuộc, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Tổ 5, ấp Mương Đào C, Xã Vân Khánh, Huyện An Minh, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Cụm công nghiệp Kiến Thành, Xã Long Cang, Huyện Cần Đước, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Số 56, quốc lộ 80, Xã Mỹ Lâm, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Thửa đất số 7, 31, 32 Tờ bản đồ số 59, Xã Thạnh Lợi, Huyện Bến Lức, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Nhà bà Dương Thị Thúy, khối 6, Thị Trấn Anh Sơn, Huyện Anh Sơn, Tỉnh Nghệ An, Việt Nam"
        },
        {
            "mst_address": "âp Đá Biên, Xã Thạnh Phước, Huyện Thạnh Hoá, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Số 373, ấp Thuận Tiến, Xã Bình Sơn, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "128 ấp Tân Hưng, Xã Mỹ Lâm, Huyện Hòn Đất, Tỉnh Kiên Giang, Việt Nam"
        },
        {
            "mst_address": "Ấp 3, Xã Tân Thành, Huyện Tân Thạnh, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "Khu công nghiệp Phú Đa, Thị trấn Phú Đa, Huyện Phú Vang, Thừa Thiên - Huế"
        },
        {
            "mst_address": "Số nhà 048 - Khu V, Thị Trấn Phố Ràng, Huyện Bảo Yên, Tỉnh Lào Cai, Việt Nam"
        },
        {
            "mst_address": "ấp 19/5, Xã Khánh Bình, Huyện Trần Văn Thời, Cà Mau"
        },
        {
            "mst_address": "Số 19/2, Ấp An Hòa 1, Xã Bình An, Huyện Thủ Thừa, Tỉnh Long An, Việt Nam"
        },
        {
            "mst_address": "DT 818 Khu Phố 11, Thị Trấn Thủ Thừa, Huyện Thủ Thừa, Tỉnh Long An, Việt Nam"
        }
        ]
        
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Only run first N cases")
    ap.add_argument(
        "--show",
        type=int,
        default=5,
        help="Show up to N failing cases (0 = none)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Write failures to JSONL file",
    )
    args = ap.parse_args()

    district_prefix_re = re.compile(
        r"(?:^|[,;\n])\s*(?:huy[ệe]n|qu[ậa]n|thi\s*x[ãa]|tx\.?|h\.|q\.)\b",
        flags=re.IGNORECASE,
    )

    addresses = [t.get("mst_address") for t in tests if isinstance(t, dict)]
    addresses = [a for a in addresses if isinstance(a, str) and a.strip()]
    if args.limit and args.limit > 0:
        addresses = addresses[: args.limit]

    start = time.time()
    failures = []
    with_district_prefix = 0

    for i, addr in enumerate(addresses, start=1):
        res = parser.process(addr)
        has_district_prefix = bool(district_prefix_re.search(addr))
        if has_district_prefix:
            with_district_prefix += 1
            if res.get("is_new") is True:
                failures.append(
                    {
                        "i": i,
                        "mst_address": addr,
                        "reason": "is_new=True but district prefix present",
                        "parsed": res,
                    }
                )

    elapsed_s = time.time() - start
    summary = {
        "cases": len(addresses),
        "cases_with_district_prefix": with_district_prefix,
        "failures": len(failures),
        "elapsed_s": round(elapsed_s, 3),
    }
    print(json.dumps(summary, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.show and args.show > 0 and failures:
        for row in failures[: args.show]:
            parsed = row.get("parsed") or {}
            prov = (parsed.get("province") or {}).get("name")
            dist = (parsed.get("district") or {}).get("name")
            ward = (parsed.get("ward") or {}).get("name")
            print("-" * 40)
            print(f"#{row['i']} {row['reason']}")
            print(row["mst_address"])
            print(
                json.dumps(
                    {
                        "province": prov,
                        "district": dist,
                        "ward": ward,
                        "is_new": parsed.get("is_new"),
                    },
                    ensure_ascii=False,
                )
            )
