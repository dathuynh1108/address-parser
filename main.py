# NOTE: you CAN change this cell
# If you want to use your own database, download it here
# !gdown ...

# NOTE: you CAN change this cell
# Add more to your needs
# you must place ALL pip install here

# NOTE: you CAN change this cell
# import your library here
import time
import json
import unicodedata
import re
from collections import Counter
from rapidfuzz.fuzz import partial_ratio, ratio
from rapidfuzz import process as rf_process

MAX_EXECUTION_TIME = 0.01

# NOTE: you MUST change this cell
# New methods / functions must be written under class Solution.



class Solution:
    def __init__(self):     
        self.reference_province_path = 'list_province.txt'
        self.reference_district_path = 'list_district.txt'
        self.reference_ward_path = 'list_ward.txt'
        
        self.standard_address_list_path = 'list_address.json'

        self.addressNodeList = []
        self.invertNgramToIndexFullNameDict = {}

        # Tunables to cap worst-case latency
        self.TOPK_CANDIDATES = 100       # bound number of candidates from inverted index
        self.DICE_GATE = 0.5            # only compute partial ratio when Dice >= this
        self.PARTIAL_CUTOFF = 50       # minimum acceptable partial ratio

        self.PreProcessAddress() # Pre-process address data once when initializing the Solution object

    def process(self, inputString: str):
        # write your process string here

        # Chuẩn hóa và tạo n-gram cho input
        startTime = time.perf_counter_ns()


        inputStringStandard = self.StandardizeName(inputString, True) 
        inputStringNgramList = self.GenerateNGrams(inputStringStandard)
                    
        # log_path = "input_standardize_log.txt"
        # with open(log_path, "a", encoding="utf-8") as f:  # "a" = append, không ghi đè
        #     f.write(inputString + "\n")
        #     f.write("=>  " + inputStringStandard + "\n\n")

        inputNgramSet = set(inputStringNgramList)

        address = self.addressNode("", "", "")

        # Tìm địa chỉ

        ngramAddressPieceList = self.NgramAddressPieceList(inputStringNgramList, self.TOPK_CANDIDATES, startTime)

        addressCandidate = self.AddressCandidateList(inputStringStandard, inputNgramSet, ngramAddressPieceList)

        if addressCandidate:
            address = self.addressNodeList[addressCandidate[0][0]] 

        return {
            "province": address.provinceName,
            "district": address.districtName,
            "ward": address.wardName,
        }

    class addressNode:
        def __init__(self, provinceName : str, districtName : str, wardName : str):
            self.fullName = wardName + " " + districtName + " " + provinceName
            self.fullName = re.sub(r"\s+", " ", self.fullName).strip()
            self.standardizedFullName = ""
            self.provinceName = provinceName
            self.districtName = districtName
            self.wardName = wardName
            self.ngramList = []  # List of n-grams for fuzzy matching

    def PreProcessAddress(self):
        # Đọc file JSON
        with open(self.standard_address_list_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Duyệt từng tỉnh
        for province_name, districts in data.items():

            node = self.addressNode(province_name, "", "")
            node.standardizedFullName = self.StandardizeName(node.fullName)
            node.ngramList = set(self.GenerateNGrams(node.standardizedFullName))
            self.addressNodeList.append(node)

            # Duyệt từng huyện
            for district_name, wards in districts.items():

                node = self.addressNode("", district_name, "")
                node.standardizedFullName = self.StandardizeName(node.fullName)
                node.ngramList = set(self.GenerateNGrams(node.standardizedFullName))
                self.addressNodeList.append(node)

                node = self.addressNode(province_name, district_name, "")
                node.standardizedFullName = self.StandardizeName(node.fullName)
                node.ngramList = set(self.GenerateNGrams(node.standardizedFullName))
                self.addressNodeList.append(node)

                # Duyệt từng xã
                for ward_name in wards:

                    node = self.addressNode("", "", ward_name)
                    node.standardizedFullName = self.StandardizeName(node.fullName)
                    node.ngramList = set(self.GenerateNGrams(node.standardizedFullName))
                    self.addressNodeList.append(node)

                    node = self.addressNode("", district_name, ward_name)
                    node.standardizedFullName = self.StandardizeName(node.fullName) 
                    node.ngramList = set(self.GenerateNGrams(node.standardizedFullName))
                    self.addressNodeList.append(node)

                    node = self.addressNode(province_name, "", ward_name)
                    node.standardizedFullName = self.StandardizeName(node.fullName)
                    node.ngramList = set(self.GenerateNGrams(node.standardizedFullName))
                    self.addressNodeList.append(node)

                    node = self.addressNode(province_name, district_name, ward_name)
                    node.standardizedFullName = self.StandardizeName(node.fullName)
                    node.ngramList = set(self.GenerateNGrams(node.standardizedFullName))
                    self.addressNodeList.append(node)
        
        # # Loại các node trùng nhau
        # unique_nodes = []
        # seen = set()
        # for node in self.addressNodeList:
        #     key = (node.provinceName, node.districtName, node.wardName, node.fullName)
        #     if key not in seen:
        #         seen.add(key)
        #         unique_nodes.append(node)
        # self.addressNodeList = unique_nodes

        # Tạo các từ điển hỗ trợ tìm kiếm nhanh
        for index, node in enumerate(self.addressNodeList, start = 0):
            # Tạo từ điển ngram đảo ngược
            self.GenerateNGramInvertedIndex(node.ngramList, index, self.invertNgramToIndexFullNameDict)

    def StandardizeName(self, name: str, advancedProcess: bool = False) -> str:
        if not name:
            return ""
        

        # --- Bước 1: Đưa về chữ thường ---
        s = name.lower()

        # --- Bước 1.1: Loại bỏ dấu chấm và dấu phẩy ở đầu và cuối chuỗi ---
        s = re.sub(r'^[\.,]+', '', s)   # xóa tất cả . hoặc , ở đầu
        s = re.sub(r'[\.,]+$', '', s)   # xóa tất cả . hoặc , ở cuối
        # --- Bước 1.2: Xóa hẳn ký tự "/" ---
        s = s.replace("/", "")
        # # --- Bước 1.3: Thay các dấu "." và "-" bằng space ---
        # s = s.replace(".", " ").replace("-", " ")

        if advancedProcess:

            s = re.sub(
                r"\b(t.t.h)\b",
                "thua thien hue",
                s,
                flags=re.IGNORECASE
            )

            s = re.sub(
                r"\b(h.c.m|h.c.minh)\b",
                " ho chi minh ",
                s,
                flags=re.IGNORECASE
            )

            # --- Bước 2: Thay cụm từ thừa bằng space (thay chính xác 100%) ---
            redundant_phrases = [
                "thành phố", "thành phô", "thành fhố", "thanh fho", "thanh pho ", "thành. phố", "thành.phố", "tp.", "t.p", "tp ", "t.phố", "t. phố", "tỉnh", "tinh", "tt.", "t.", " t ",
                "quận", "qận", "qun", "q.", "q ", "huyện", "h.", " h ",  ".h ", "thị xã", "thị.xã", "tx.", "t.xã", "tx ", "thị trấn", "thị.trấn", "tt ",
                "xã", "x.", "x ", "phường", "kp.", "p.", " p ", ".p ", "phường.", "phường ", 
                "f", "j", "z", "w"
            
            ]

            for phrase in redundant_phrases:
                s = s.replace(phrase, " ")

            s = re.sub(
                r"\b("
                r"|tiểu\s*khu(\s*\d+\w*)?"      # tiểu khu 3, tiểu khu12a               
                r"|khu\s*pho(\s*\d+\w*)?"          # khu phố, khu phố 3
                r"|khu\s*phố(\s*\d+\w*)?"          # khu phố, khu phố 3
                r"|khu\s*vuc(\s*\d+\w*)?"          # khu vực, khu vực 2
                r"|khu\s*vực(\s*\d+\w*)?"          # khu vực, khu vực 2
                r"|khu(\s*\d+\w*)?"                 # khu, khu 3, khu12a
                r"|kp(\s*\d+\w*)?"                  # kp2, kp 3
                r"|tổ\s*dân\s*phố(\s*\d+\w*)?"     # tổ dân phố 5, tổ dân phố12a
                r"|tổ(\s*\d+\w*)?"                  # tổ 1
                r"|thôn(\s*\d+\w*)?"                # thôn 3
                r"|xóm(\s*\d+\w*)?"                 # xóm 2
                r"|cụm(\s*\d+\w*)?"                 # cụm 3
                r"|phố(\s*\d+\w*)?"                 # phố 5
                r"|khóm(\s*\d+\w*)?"                # khóm 2
                r"|số\s*nhà(\s*\d+\w*)?"            # số nhà 12
                r"|số(\s*\d+\w*)?"                   # số 12
                r"|nhà(\s*\d+\w*)?"                   # nhà 12
                r"|ấp(\s*\d+\w*)?"              # ấp 1, ấp2
                r"|ngách\s*\d+\w*"      # ngách 12, ngách12a
                r"|ngõ\s*\d+\w*"        # ngõ 12, ngõ12a   
                r"|hẻm\s*\d+\w*"
                r")\b",
                "",
                s,
                flags=re.IGNORECASE
            )

            # --- Bước 3: Loại các cụm "tp" dính liền chữ, ví dụ "tpbao loc" → "bao loc" ---
            s = re.sub(r"\btp([a-z0-9]+)", r"\1", s)

        # --- Bước 4: Chuẩn hóa Unicode & bỏ dấu ---
        s = s.replace("đ", "d")
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

        # --- Bước 5: Giữ lại a-z, 0-9, space ---
        s = re.sub(r"[^a-z0-9\s]+", " ", s)

        if advancedProcess:

            # Chuẩn hóa các biến thể của "ho chi minh"
            s = re.sub(
                r"\b(hochiminh|hochi\s*minh|ho\s*chiminh|hcm|hcminh)\b",
                "ho chi minh",
                s,
                flags=re.IGNORECASE
            )
            if re.search(r"\bho chi minh\b", s, flags=re.IGNORECASE):
                mapping = {
                    "bc": "binh chanh",
                    "tb": "tan binh",
                    "bt": "binh thanh",
                    "gv": "go vap",
                    "pn": "phu nhuan",
                    "cc": "cu chi",
                    "hm": "hoc mon",
                    "nb": "nha be",
                }

                # Thay từng viết tắt bằng tên đầy đủ (chỉ thay khi là từ riêng biệt)
                for abbr, full in mapping.items():
                    s = re.sub(rf"\b{abbr}\b", full, s, flags=re.IGNORECASE)


            # --- Bước 7: Loại bỏ các chuỗi chứa từ 3 chữ số trở lên ---

            # Bỏ số 0 ở đầu của mọi cụm số
            s = re.sub(r"\b0+(\d+)\b", r"\1", s)
            # Tức là "abc123xyz" hoặc "123" đều bị loại bỏ phần chứa "123"
            s = re.sub(r"\d{3,}", "", s)

            # --- Bước 8: Bỏ 'p' hoặc 'q' trước số (vd: p1 → 1, q10 → 10) ---
            s = re.sub(r"\b[pq](\d+)\b", r"\1", s)

            # --- Bước X: Loại bỏ các cụm địa chỉ thừa ---
            

        # --- Bước 9: Gom space ---
        s = re.sub(r"\s+", " ", s).strip()
        # print(s)
        return s

    def GenerateNGrams(self, s: str, n : int = 4) -> list:
        s = f" {s} "  # Thêm khoảng trắng ở đầu và cuối để tạo n-gram chính xác
        ngrams = [s[i:i+n] for i in range(len(s)-n+1)]
        return ngrams

    def GenerateNGramInvertedIndex(self, ngramList: list, index: int, invertNgramToIndexDict: dict):
        for ngram in ngramList:
            if ngram not in invertNgramToIndexDict:
                invertNgramToIndexDict[ngram] = set()
            invertNgramToIndexDict[ngram].add(index)

    def NgramAddressPieceList(self, inputNgramList: list, topk: int, startTime: float) -> list:
        counter = Counter()
        invert_dict = self.invertNgramToIndexFullNameDict

        # Iterate unique ngrams to avoid redundant counting
        for ngram in set(inputNgramList):
            if ngram in invert_dict:
                counter.update(invert_dict[ngram])  # ✅ xử lý hàng loạt
            
            if (time.perf_counter_ns() - startTime)/1000000000 >= MAX_EXECUTION_TIME:
                return counter.most_common(topk)

        # Return only top-K candidates to cap cost (heap-based in CPython)
        return counter.most_common(topk)

    def AddressCandidateList(self, inputStringStandard: str, inputNgramSet: set, ngramAddressPieceList: list) -> list:
        # Stage 1: filter by Dice; collect IDs whose Dice >= gate
        A = inputNgramSet
        lenA = len(A)
        filtered_ids = []

        index = 0
        for idx_count in ngramAddressPieceList:
            idx = idx_count[0]
            B = self.addressNodeList[idx].ngramList
            inter = 0
            # Fast overlap count without building set
            for g in A:
                if g in B:
                    inter += 1
            dice_score = (2 * inter) / (lenA + len(B))

            index += 1

            if dice_score >= self.DICE_GATE:
                filtered_ids.append(idx)
            else:
                # Counter is ordered by frequency; dice will only go down
                if index >= 50:
                    break

        if not filtered_ids:
            return []

        # Stage 2: one vectorized RapidFuzz call over filtered strings
        choices = [self.addressNodeList[i].standardizedFullName for i in filtered_ids] 
        res = rf_process.extractOne( 
            inputStringStandard, 
            choices, 
            scorer=ratio, 
            score_cutoff=self.PARTIAL_CUTOFF, 
        )

        if res is None:
            return []

        choice_str, score, rel_idx = res
        best_abs_idx = filtered_ids[rel_idx]
        return [(best_abs_idx, float(score), self.addressNodeList[best_abs_idx].fullName)]

groups_province = {}
groups_district = {'hòa bình': ['Hoà Bình', 'Hòa Bình'], 'kbang': ['Kbang', 'KBang'], 'quy nhơn': ['Qui Nhơn', 'Quy Nhơn']}
groups_ward = {'ái nghĩa': ['ái Nghĩa', 'Ái Nghĩa'], 'ái quốc': ['ái Quốc', 'Ái Quốc'], 'ái thượng': ['ái Thượng', 'Ái Thượng'], 'ái tử': ['ái Tử', 'Ái Tử'], 'ấm hạ': ['ấm Hạ', 'Ấm Hạ'], 'an ấp': ['An ấp', 'An Ấp'], 'ẳng cang': ['ẳng Cang', 'Ẳng Cang'], 'ẳng nưa': ['ẳng Nưa', 'Ẳng Nưa'], 'ẳng tở': ['ẳng Tở', 'Ẳng Tở'], 'an hòa': ['An Hoà', 'An Hòa'], 'ayun': ['Ayun', 'AYun'], 'bắc ái': ['Bắc ái', 'Bắc Ái'], 'bảo ái': ['Bảo ái', 'Bảo Ái'], 'bình hòa': ['Bình Hoà', 'Bình Hòa'], 'châu ổ': ['Châu ổ', 'Châu Ổ'], 'chư á': ['Chư á', 'Chư Á'], 'chư rcăm': ['Chư Rcăm', 'Chư RCăm'], 'cộng hòa': ['Cộng Hoà', 'Cộng Hòa'], 'cò nòi': ['Cò  Nòi', 'Cò Nòi'], 'đại ân 2': ['Đại Ân  2', 'Đại Ân 2'], 'đak ơ': ['Đak ơ', 'Đak Ơ'], "đạ m'ri": ["Đạ M'ri", "Đạ M'Ri"], 'đông hòa': ['Đông Hoà', 'Đông Hòa'], 'đồng ích': ['Đồng ích', 'Đồng Ích'], 'hải châu i': ['Hải Châu  I', 'Hải Châu I'], 'hải hòa': ['Hải Hoà', 'Hải Hòa'], 'hành tín đông': ['Hành Tín  Đông', 'Hành Tín Đông'], 'hiệp hòa': ['Hiệp Hoà', 'Hiệp Hòa'], 'hòa bắc': ['Hoà Bắc', 'Hòa Bắc'], 'hòa bình': ['Hoà Bình', 'Hòa Bình'], 'hòa châu': ['Hoà Châu', 'Hòa Châu'], 'hòa hải': ['Hoà Hải', 'Hòa Hải'], 'hòa hiệp trung': ['Hoà Hiệp Trung', 'Hòa Hiệp Trung'], 'hòa liên': ['Hoà Liên', 'Hòa Liên'], 'hòa lộc': ['Hoà Lộc', 'Hòa Lộc'], 'hòa lợi': ['Hoà Lợi', 'Hòa Lợi'], 'hòa long': ['Hoà Long', 'Hòa Long'], 'hòa mạc': ['Hoà Mạc', 'Hòa Mạc'], 'hòa minh': ['Hoà Minh', 'Hòa Minh'], 'hòa mỹ': ['Hoà Mỹ', 'Hòa Mỹ'], 'hòa phát': ['Hoà Phát', 'Hòa Phát'], 'hòa phong': ['Hoà Phong', 'Hòa Phong'], 'hòa phú': ['Hoà Phú', 'Hòa Phú'], 'hòa phước': ['Hoà Phước', 'Hòa Phước'], 'hòa sơn': ['Hoà Sơn', 'Hòa Sơn'], 'hòa tân': ['Hoà Tân', 'Hòa Tân'], 'hòa thuận': ['Hoà Thuận', 'Hòa Thuận'], 'hòa tiến': ['Hoà Tiến', 'Hòa Tiến'], 'hòa trạch': ['Hoà Trạch', 'Hòa Trạch'], 'hòa vinh': ['Hoà Vinh', 'Hòa Vinh'], 'hương hòa': ['Hương Hoà', 'Hương Hòa'], 'ích hậu': ['ích Hậu', 'Ích Hậu'], 'ít ong': ['ít Ong', 'Ít Ong'], 'khánh hòa': ['Khánh Hoà', 'Khánh Hòa'], 'krông á': ['Krông Á', 'KRông á'], 'lộc hòa': ['Lộc Hoà', 'Lộc Hòa'], 'minh hòa': ['Minh Hoà', 'Minh Hòa'], 'mường ải': ['Mường ải', 'Mường Ải'], 'mường ẳng': ['Mường ẳng', 'Mường Ẳng'], 'nậm ét': ['Nậm ét', 'Nậm Ét'], 'nam hòa': ['Nam Hoà', 'Nam Hòa'], 'na ư': ['Na ư', 'Na Ư'], 'ngã sáu': ['Ngã sáu', 'Ngã Sáu'], 'nghi hòa': ['Nghi Hoà', 'Nghi Hòa'], 'nguyễn úy': ['Nguyễn Uý', 'Nguyễn úy', 'Nguyễn Úy'], 'nhân hòa': ['Nhân Hoà', 'Nhân Hòa'], 'nhơn hòa': ['Nhơn Hoà', 'Nhơn Hòa'], 'nhơn nghĩa a': ['Nhơn nghĩa A', 'Nhơn Nghĩa A'], 'phúc ứng': ['Phúc ứng', 'Phúc Ứng'], 'phước hòa': ['Phước Hoà', 'Phước Hòa'], 'sơn hóa': ['Sơn Hoá', 'Sơn Hóa'], 'tạ an khương đông': ['Tạ An Khương  Đông', 'Tạ An Khương Đông'], 'tạ an khương nam': ['Tạ An Khương  Nam', 'Tạ An Khương Nam'], 'tăng hòa': ['Tăng Hoà', 'Tăng Hòa'], 'tân hòa': ['Tân Hoà', 'Tân Hòa'], 'tân hòa thành': ['Tân Hòa  Thành', 'Tân Hòa Thành'], 'tân khánh trung': ['Tân  Khánh Trung', 'Tân Khánh Trung'], 'tân lợi': ['Tân lợi', 'Tân Lợi'], 'thái hòa': ['Thái Hoà', 'Thái Hòa'], 'thiết ống': ['Thiết ống', 'Thiết Ống'], 'thuận hòa': ['Thuận Hoà', 'Thuận Hòa'], 'thượng ấm': ['Thượng ấm', 'Thượng Ấm'], 'thụy hương': ['Thuỵ Hương', 'Thụy Hương'], 'thủy xuân': ['Thuỷ Xuân', 'Thủy Xuân'], 'tịnh ấn đông': ['Tịnh ấn Đông', 'Tịnh Ấn Đông'], 'tịnh ấn tây': ['Tịnh ấn Tây', 'Tịnh Ấn Tây'], 'triệu ái': ['Triệu ái', 'Triệu Ái'], 'triệu ẩu': ['Triệu ẩu', 'Triệu Ẩu'], 'trung hòa': ['Trung Hoà', 'Trung Hòa'], 'trung ý': ['Trung ý', 'Trung Ý'], 'tùng ảnh': ['Tùng ảnh', 'Tùng Ảnh'], 'úc kỳ': ['úc Kỳ', 'Úc Kỳ'], 'ứng hòe': ['ứng Hoè', 'Ứng Hoè'], 'vĩnh hòa': ['Vĩnh Hoà', 'Vĩnh Hòa'], 'vũ hòa': ['Vũ Hoà', 'Vũ Hòa'], 'xuân ái': ['Xuân ái', 'Xuân Ái'], 'xuân áng': ['Xuân áng', 'Xuân Áng'], 'xuân hòa': ['Xuân Hoà', 'Xuân Hòa'], 'xuất hóa': ['Xuất Hoá', 'Xuất Hóa'], 'ỷ la': ['ỷ La', 'Ỷ La']}
groups_ward.update({1: ['1', '01'], 2: ['2', '02'], 3: ['3', '03'], 4: ['4', '04'], 5: ['5', '05'], 6: ['6', '06'], 7: ['7', '07'], 8: ['8', '08'], 9: ['9', '09']})
def to_same(groups):
    same = {ele: k for k, v in groups.items() for ele in v}
    return same
same_province = to_same(groups_province)
same_district = to_same(groups_district)
same_ward = to_same(groups_ward)
def normalize(text, same_dict):
    return same_dict.get(text, text)

TEAM_NAME = 'CTG'
EXCEL_FILE = f'{TEAM_NAME}.xlsx'

import json
import time
with open("test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

summary_only = True
df = []
solution = Solution()

inputString = "TT Đạ Tẻh, , T. Lâm Đồng"
solution.process(inputString)

timer = []
correct = 0
for test_idx, data_point in enumerate(data):
    address = data_point["text"]

    ok = 0
    try:
        answer = data_point["result"]
        answer["province_normalized"] = normalize(answer["province"], same_province)
        answer["district_normalized"] = normalize(answer["district"], same_district)
        answer["ward_normalized"] = normalize(answer["ward"], same_ward)

        start = time.perf_counter_ns()
        result = solution.process(address)
        finish = time.perf_counter_ns()
        timer.append(finish - start)
        result["province_normalized"] = normalize(result["province"], same_province)
        result["district_normalized"] = normalize(result["district"], same_district)
        result["ward_normalized"] = normalize(result["ward"], same_ward)

        province_correct = int(answer["province_normalized"] == result["province_normalized"])
        district_correct = int(answer["district_normalized"] == result["district_normalized"])
        ward_correct = int(answer["ward_normalized"] == result["ward_normalized"])
        ok = province_correct + district_correct + ward_correct

        df.append([
            test_idx,
            address,
            answer["province"],
            result["province"],
            answer["province_normalized"],
            result["province_normalized"],
            province_correct,
            answer["district"],
            result["district"],
            answer["district_normalized"],
            result["district_normalized"],
            district_correct,
            answer["ward"],
            result["ward"],
            answer["ward_normalized"],
            result["ward_normalized"],
            ward_correct,
            ok,
            timer[-1] / 1_000_000_000,
        ])
    except Exception as e:
        print(f"{answer = }")
        print(f"{result = }")
        df.append([
            test_idx,
            address,
            answer["province"],
            "EXCEPTION",
            answer["province_normalized"],
            "EXCEPTION",
            0,
            answer["district"],
            "EXCEPTION",
            answer["district_normalized"],
            "EXCEPTION",
            0,
            answer["ward"],
            "EXCEPTION",
            answer["ward_normalized"],
            "EXCEPTION",
            0,
            0,
            0,
        ])
        # any failure count as a zero correct
        pass
    correct += ok


    if not summary_only:
        # responsive stuff
        print(f"Test {test_idx:5d}/{len(data):5d}")
        print(f"Correct: {ok}/3")
        print(f"Time Executed: {timer[-1] / 1_000_000_000:.4f}")


print(f"-"*30)
total = len(data) * 3
score_scale_10 = round(correct / total * 10, 2)
if len(timer) == 0:
    timer = [0]
max_time_sec = round(max(timer) / 1_000_000_000, 4)
avg_time_sec = round((sum(timer) / len(timer)) / 1_000_000_000, 4)

import pandas as pd

df2 = pd.DataFrame(
    [[correct, total, score_scale_10, max_time_sec, avg_time_sec]],
    columns=['correct', 'total', 'score / 10', 'max_time_sec', 'avg_time_sec',],
)

columns = [
    'ID',
    'text',
    'province',
    'province_student',
    'province_normalized',
    'province_student_normalized',
    'province_correct',
    'district',
    'district_student',
    'district_normalized',
    'district_student_normalized',
    'district_correct',
    'ward',
    'ward_student',
    'ward_normalized',
    'ward_student_normalized',
    'ward_correct',
    'total_correct',
    'time_sec',
]

df = pd.DataFrame(df)
df.columns = columns

print(f'{TEAM_NAME = }')
print(f'{EXCEL_FILE = }')
print(df2)

writer = pd.ExcelWriter(EXCEL_FILE, engine='xlsxwriter')
df2.to_excel(writer, index=False, sheet_name='summary')
df.to_excel(writer, index=False, sheet_name='details')
writer.close()

