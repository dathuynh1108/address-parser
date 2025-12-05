from inexus_parser import AddressParser
parser = AddressParser()

if __name__ == "__main__":
    tests = [
        "269 Tăng Bạt Hổ, Khu Vực 05, Phường Lý Thường Kiệt, Thành phố Quy Nhơn, Tỉnh Bình Định, Việt Nam",
        #"Số 269 Văn Cao, Phường Hải An, TP Hải Phòng, Việt Nam",
        # "Số nhà 14 ngách 222/44 đường 19/5 Văn Quán, Phường Hà Đông, TP Hà Nội, Việt Nam",
        # "Thôn Tân Lập, xã Ea Bông, , Huyện Krông Pắk, Đắk Lắk"
        # "74/3 đường Hiệp Thành 17, tổ 23, khu phố 2, Phường Hiệp Thành, Quận 12, Thành phố Hồ Chí Minh, Việt Nam",
        # "Số 25/19, Đường số 13, Khu phố Bình Đường 1, Phường An Bình, Thành phố Dĩ An, Tỉnh Bình Dương, Việt Nam",
        # "46/1L Ấp Xuân Thới Đông 2, Xã Xuân Thới Đông, Huyện Hóc Môn, Thành phố Hồ Chí Minh, Việt Nam",
        # "Tiểu khu K1-G3, Đường D1, Khu công nghệ cao, Phường Tân Phú, Quận 9, Thành phố Hồ Chí Minh, Việt Nam",
        # "70 Vũ Tông Phan, Phường An Phú, Quận 2, Thành phố Hồ Chí Minh, Việt Nam",
        # "1/3/3 Đường 160, Phường Tăng Nhơn Phú A, Quận 9, Thành phố Hồ Chí Minh, Việt Nam"
        # "83 Triệu Nữ Vương, Phường Hải Châu Ii, Quận Hải Châu, Thành phố Đà Nẵng, Việt Nam",
        # "Số 4 đường 102, Khu phố 1, Phường Tăng Nhơn Phú A, Thành phố Thủ Đức, Thành phố Hồ Chí Minh, Việt Nam",
        # "Thôn Phương Nhị, Xã Liên Ninh, Huyện Thanh Trì, Hà Nội",
        # "308 Trần Hưng Đạo P.NCT, , Quận 1, TP Hồ Chí Minh",
        # "Gian L1-01A, Trung tâm thương mại Vincom Plaza Thái Bình, số 460 đường Lý Bôn, Phường Đề Thám, Thành phố Thái Bình, Tỉnh Thái Bình, Việt Nam",
        # "Thương Cảng Vũng Tàu, Số 973, Đường 30/4, Phường 11, Thành phố Vũng Tàu, Tỉnh Bà Rịa - Vũng Tàu, Việt Nam",
        # "116A Đường Tạ Quang Bửu, Phường 3, Quận 8, Thành phố Hồ Chí Minh, Việt Nam",
        # "1027 Nguyễn Tất Thành, Phường Xuân Hà, Quận Thanh Khê, Thành phố Đà Nẵng, Việt Nam",
        # "2/16 Đường số 7, Cư Xá Đô Thành, Phường Bàn Cờ, TP Hồ Chí Minh, Việt Nam",
        # "Thửa đất 101, tờ bản đồ số 88, tổ 2, đường 30/4, khu phố 1, Đặc khu Phú Quốc, Tỉnh AN Giang, Việt Nam",
        # "Lô 09 Bến xe Trung tâm, Phường Hoà An, Quận Cẩm Lệ, Thành phố Đà Nẵng, Việt Nam"
        # "171 Đinh Bộ Lĩnh, Phường 26, Quận Bình Thạnh, Thành phố Hồ Chí Minh, Việt Nam",
        # "Tầng 9, số 271 đường Nguyễn Văn Linh, Tòa nhà Bưu Điện Thành Phố Đà Nẵng, Phường Vĩnh Trung, Quận Thanh Khê, Thành phố Đà Nẵng, Việt Nam"
        # "Lô 213 Khu dân cư Kho Lào, Phường Hoà Hiệp Nam, Quận Liên Chiểu, Thành phố Đà Nẵng, Việt Nam",
        # "Thửa đất 101, tờ bản đồ số 88, tổ 2, đường 30/4, khu phố 1, Đặc khu Phú Quốc, Tỉnh AN Giang, Việt Nam",
        # "Phường An Hải Tây, Đà Nẵng",
        # "Ấp 5, Xã Vị Thanh 1, TP Cần Thơ, Việt Nam",
        # "41 - 43 Nguyễn Duy Dương, Phường 08, Quận 5, Thành phố Hồ Chí Minh",
        # "41 - 43 Nguyễn Duy Dương, Phường 08, Quận 5, Thành phố Hồ Chí Minh, Việt Nam",
        # "Phòng 202, Lầu 2, 70 Lý Tự Trọng, Phường Bến Thành, Quận 1, Thành phố Hồ Chí Minh, Việt Nam",
        # "115-117 Thuận Kiều, Phường 4, Quận 11, Thành phố Hồ Chí Minh, Việt Nam",
        # "Lô D, KCN Quế Võ, Phường Nam Sơn, Tỉnh Bắc Ninh, Việt Nam",
        # "Tầng 4, Tòa nhà Dương Tuấn, đường Lê Thái Tổ, Phường Võ Cường, Tỉnh Bắc Ninh, Việt Nam",
        # "Lô A4-3, đường 6, Khu Công nghiệp Công nghệ cao Long Thành, Xã Long Thành, Tỉnh Đồng Nai, Việt Nam",
        # "Số 1505/60/2, đường Bùi Hữu Nghĩa, khu phố Tân Hạnh 3, Phường Biên Hòa, Tỉnh Đồng Nai, Việt Nam",
        # "76B/34 Nguyễn Nhạc, Phường Thống Nhất, Tỉnh Gia Lai, Việt Nam",
        # "Tổ 2, Phường Thống Nhất, Tỉnh Gia Lai, Việt Nam",
        # "81 Tôn Thất Tùng, Phường Pleiku, Tỉnh Gia Lai, Việt Nam",
        # "Số 35 Ngô Gia Tự, Phường Nguyễn Văn Cừ, Thành phố Quy Nhơn, Tỉnh Bình Định, Việt Nam",
        # "Số 97 đường Lê Lợi, Phường Trần Hưng Đạo, Thành phố Quy Nhơn, Tỉnh Bình Định, Việt Nam",
        # "Tổ 18, ấp Bàu Trâm , Xã Bàu Trâm, Thành phố Long khánh, Tỉnh Đồng Nai, Việt Nam",
        # "Số 89, tổ 4, KP Tân Phong, Phường Xuân Tân, Thành phố Long khánh, Tỉnh Đồng Nai, Việt Nam",
        # "44 Trần Phú, Phường Lý Thường Kiệt, Thành phố Quy Nhơn, Bình Định",
        # "Số 11 Huỳnh Văn Thống, Phường Nhơn Bình, Thành phố Quy Nhơn, Tỉnh Bình Định, Việt Nam",
        # "Lô 57 Phan Tứ, Phường Mỹ An, Quận Ngũ Hành Sơn, Thành phố Đà Nẵng, Việt Nam",
        # "260/20B Hải Phòng, Phường Tân Chính, Quận Thanh Khê, Thành phố Đà Nẵng, Việt Nam",
        # "Số nhà E4, Lô 35, đường Vũ Miên, Thôn Miếu Bông, Xã Hoà Phước, Huyện Hoà Vang, Thành phố Đà Nẵng, Việt Nam",
        # "25/16A, Lý Thường Kiệt, Phường Thạch Thang, Quận Hải Châu, Thành phố Đà Nẵng, Việt Nam",
        # "10 Tôn Quang Phiệt, Phường Nại Hiên Đông, Quận Sơn Trà, Thành phố Đà Nẵng, Việt Nam",
        # "263 Hoàng Diệu, Quận Hải Châu, Thành phố Đà Nẵng, Việt Nam",
        # "Số nhà 11, ngõ 229, khu 10, phố Bình Lộc, Phường Tân Bình, Thành phố Hải Dương, Tỉnh Hải Dương, Việt Nam",
    ]
    
    for test in tests:
        result = parser.process(test)
        print(f"Input: {test}")
        print(f"Parsed: {result}")
        print("-" * 40)
    
    # parser.search_province()