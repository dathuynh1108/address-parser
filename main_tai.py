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
from rapidfuzz import process as rf_process

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
        self.TOPK_CANDIDATES = 600       # bound number of candidates from inverted index
        self.DICE_GATE = 0.52            # only compute partial ratio when Dice >= this
        self.PARTIAL_CUTOFF = 75.0       # minimum acceptable partial ratio

        self.PreProcessAddress() # Pre-process address data once when initializing the Solution object


    def process(self, inputString: str):
        # write your process string here

        # Chuẩn hóa và tạo n-gram cho input
        inputStringStandard = self.StandardizeName(inputString)
        inputStringNgramList = self.GenerateNGrams(inputStringStandard)
        inputNgramSet = set(inputStringNgramList)

        address = self.addressNode("", "", "")

        # Tìm địa chỉ
        ngramAddressPieceList = self.NgramAddressPieceList(inputStringNgramList, self.TOPK_CANDIDATES)

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

        # Tạo các từ điển hỗ trợ tìm kiếm nhanh
        for index, node in enumerate(self.addressNodeList, start = 0):
            # Tạo từ điển ngram đảo ngược
            self.GenerateNGramInvertedIndex(node.ngramList, index, self.invertNgramToIndexFullNameDict)

    def StandardizeName(self, name: str) -> str:
        if not name:
            return ""

        # --- Bước 1: Đưa về chữ thường ---
        s = name.lower()

        # --- Bước 2: Thay cụm từ thừa bằng space (thay chính xác 100%) ---
        redundant_phrases = [
            "thành phố","thành. phố", "thành.phố", "tp.", "tp ", "t.phố", "t. phố", "tỉnh", "t.", "t ",
            "quận", "qận", "qun", "q.", "q ", "huyện", "h.", "h ", "thị xã", "thị.xã", "tx.", "tx ", "thị trấn", "thị.trấn", "tt.", "tt ",
            "xã", "x.", "x ", "phường", "p.", "p ", "phường.", "phường "
          
        ]

        for phrase in redundant_phrases:
            s = s.replace(phrase, " ")

        # --- Bước 3: Loại các cụm "tp" dính liền chữ, ví dụ "tpbao loc" → "bao loc" ---
        s = re.sub(r"\btp([a-z0-9]+)", r"\1", s)

        # --- Bước 4: Chuẩn hóa Unicode & bỏ dấu ---
        s = s.replace("đ", "d")
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

        # --- Bước 5: Giữ lại a-z, 0-9, space ---
        s = re.sub(r"[^a-z0-9\s]+", " ", s)

        # --- Bước 6: Gom space ---
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

    def NgramAddressPieceList(self, inputNgramList: list, topk: int) -> list:
        counter = Counter()
        invert_dict = self.invertNgramToIndexFullNameDict

        # Iterate unique ngrams to avoid redundant counting
        for ngram in set(inputNgramList):
            if ngram in invert_dict:
                counter.update(invert_dict[ngram])  # ✅ xử lý hàng loạt

        # Return only top-K candidates to cap cost (heap-based in CPython)
        return counter.most_common(topk)

    def AddressCandidateList(self, inputStringStandard: str, inputNgramSet: set, ngramAddressPieceList: list) -> list:
        # Stage 1: filter by Dice; collect IDs whose Dice >= gate
        A = inputNgramSet
        lenA = len(A)
        filtered_ids = []

        for idx_count in ngramAddressPieceList:
            idx = idx_count[0]
            B = self.addressNodeList[idx].ngramList
            inter = 0
            # Fast overlap count without building set
            for g in A:
                if g in B:
                    inter += 1
            dice_score = (2 * inter) / (lenA + len(B))
            if dice_score >= self.DICE_GATE:
                filtered_ids.append(idx)
            else:
                # Counter is ordered by frequency; dice will only go down
                break

        if not filtered_ids:
            return []

        # Stage 2: one vectorized RapidFuzz call over filtered strings
        choices = [self.addressNodeList[i].standardizedFullName for i in filtered_ids]
        res = rf_process.extractOne(
            inputStringStandard,
            choices,
            scorer=partial_ratio,
            score_cutoff=self.PARTIAL_CUTOFF,
        )

        if res is None:
            return []

        choice_str, score, rel_idx = res
        best_abs_idx = filtered_ids[rel_idx]
        return [(best_abs_idx, float(score), self.addressNodeList[best_abs_idx].fullName)]
