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
from rapidfuzz.fuzz import partial_ratio

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

        self.PreProcessAddress() # Pre-process address data once when initializing the Solution object


    def process(self, inputString: str):
        # write your process string here

        # Chuẩn hóa và tạo n-gram cho input
        inputStringStandard = self.StandardizeName(inputString)
        inputStringNgramList = self.GenerateNGrams(inputStringStandard)

        address = self.addressNode("", "", "")

        # Tìm địa chỉ
        ngramAddressPieceList = self.NgramAddressPieceList(inputStringNgramList)

        addressCandidate = self.AddressCandidateList(inputStringStandard, ngramAddressPieceList)

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
            # Duyệt từng huyện
            for district_name, wards in districts.items():
                # Duyệt từng xã
                for ward_name in wards:
                    node = self.addressNode(province_name, district_name, ward_name)

                    # Chuẩn hóa tên
                    node.standardizedFullName = self.StandardizeName(node.fullName)

                    # Sinh ngram cho node (nếu bạn có hàm này)
                    node.ngramList = set(self.GenerateNGrams(node.standardizedFullName))

                    # Thêm node vào list
                    self.addressNodeList.append(node)

        # Tạo các từ điển hỗ trợ tìm kiếm nhanh
        for index, node in enumerate(self.addressNodeList, start = 0):
            # Tạo từ điển ngram đảo ngược
            self.GenerateNGramInvertedIndex(node.ngramList, index, self.invertNgramToIndexFullNameDict)

    def StandardizeName(self, name: str) -> str:
        if not name:
            return ""
        # Đưa về chữ thường
        s = name.lower()

        # Chuyển đ -> d
        s = s.replace("đ", "d")

        # Tách dấu Unicode (NFD) rồi loại bỏ các ký tự dấu (Mn = Mark, Nonspacing)
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

        # Giữ lại chỉ a-z, 0-9 và khoảng trắng
        s = re.sub(r"[^a-z0-9\s]+", " ", s)

        # Gom nhiều khoảng trắng thành 1, bỏ khoảng trắng đầu cuối
        s = re.sub(r"\s+", " ", s).strip()

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
    
    # def NgramAddressPieceList(self, inputNgramList: list) -> dict:
    #     ngramDict = {}
    #     for ngram in inputNgramList:
    #         if ngram in self.invertNgramToIndexFullNameDict:
    #             for index in self.invertNgramToIndexFullNameDict[ngram]:
    #                 if index not in ngramDict:
    #                     ngramDict[index] = 0
    #                 ngramDict[index] += 1
        
    #     # Sắp xếp giảm dần theo value
    #     ngramDict = sorted(
    #         ngramDict.items(), 
    #         key=lambda item: item[1],  # sắp theo value
    #         reverse=True               # giảm dần
    #     )

    #     return ngramDict

    def NgramAddressPieceList(self, inputNgramList: list) -> list:
        counter = Counter()
        invert_dict = self.invertNgramToIndexFullNameDict

        for ngram in inputNgramList:
            if ngram in invert_dict:
                counter.update(invert_dict[ngram])  # ✅ xử lý hàng loạt

        # Trả về list sorted (value giảm dần)
        return counter.most_common()

    def AddressCandidateList(self, inputStringStandard: str, ngramAddressPieceList: list) -> list:
        candidateList = []

        A  = set(self.GenerateNGrams(inputStringStandard))
        for addressIndex in ngramAddressPieceList:
            # Tính Dice
            B = self.addressNodeList[addressIndex[0]].ngramList
            inter = len(A & B)
            dice_score = (2 * inter) / (len(A) + len(B))
            if dice_score > 0.5:
                # partialRatio = self.PartialRatio(inputStringStandard, self.addressNodeList[addressIndex[0]].standardizedFullName)
                partialRatio = partial_ratio(inputStringStandard, self.addressNodeList[addressIndex[0]].standardizedFullName)
            else:
                break

            if partialRatio > 75.0:
                candidateList.append((addressIndex[0], partialRatio))
        
        candidateList.sort(key=lambda x: x[1], reverse=True)

        

        return candidateList

    def MinDistanceSubstring(self, pattern: str, text: str) -> int:
        """
        Correct Sellers (substring Levenshtein):
        - D[0][j] = 0 for all j  (free to start anywhere in `text`)
        - D[i][0] = i            (must consume all of `pattern`)
        - answer = min over last row (free to end anywhere in `text`)
        """

        m, n = len(pattern), len(text)
        if m == 0: 
            return 0
        if n == 0: 
            return m

        # FIRST ROW: all zeros (not 0..n!)
        prev = [0] * (n + 1)

        for i in range(1, m + 1):
            # FIRST COLUMN: i
            cur = [i] + [0] * n
            pi = pattern[i - 1]
            for j in range(1, n + 1):
                cost = 0 if pi == text[j - 1] else 1
                cur[j] = min(
                    prev[j] + 1,      # deletion in pattern
                    cur[j - 1] + 1,   # insertion in pattern
                    prev[j - 1] + cost
                )
            prev = cur

        return min(prev)  # best substring end anywhere
        
    def PartialRatio(self, a: str, b: str) -> float:
        """
        RapidFuzz-like PartialRatio in [0, 100], but always using your normalize().
        """
        if not a and not b: return 100.0
        if not a or not b:  return 0.0
        # shorter is the needle
        if len(a) <= len(b):
            pat, txt = a, b
        else:
            pat, txt = b, a
        dist = self.MinDistanceSubstring(pat, txt)
        return 100.0 * (1.0 - dist / max(1, len(pat)))



TEAM_NAME = 'CTG'  # This should be your team nameword
EXCEL_FILE = f'{TEAM_NAME}.xlsx'

with open('test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

summary_only = True
df = []
solution = Solution()
timer = []
correct = 0
for test_idx, data_point in enumerate(data):
    address = data_point["text"]

    ok = 0
    try:
        start = time.perf_counter_ns()
        result = solution.process(address)
        answer = data_point["result"]
        finish = time.perf_counter_ns()
        timer.append(finish - start)
        ok += int(answer["province"] == result["province"])
        ok += int(answer["district"] == result["district"])
        ok += int(answer["ward"] == result["ward"])
        df.append([
            test_idx,
            address,
            answer["province"],
            result["province"],
            int(answer["province"] == result["province"]),
            answer["district"],
            result["district"],
            int(answer["district"] == result["district"]),
            answer["ward"],
            result["ward"],
            int(answer["ward"] == result["ward"]),
            ok, 
            timer[-1] / 1_000_000_000,
        ])
    except Exception as e:
        df.append([
            test_idx,
            address,
            answer["province"],
            "EXCEPTION",
            0,
            answer["district"],
            "EXCEPTION",
            0,
            answer["ward"],
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
    'province_correct',
    'district',
    'district_student',
    'district_correct',
    'ward',
    'ward_student',
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
    