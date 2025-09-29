# NOTE: you MUST change this cell
# New methods / functions must be written under class Solution.
import re
import unicodedata
import ahocorasick                         # pip install pyahocorasick
from rapidfuzz.distance import Levenshtein as RFLev   # pip install rapidfuzz
# NOTE: you MUST change this cell
# New methods / functions must be written under class Solution.
import re
import unicodedata
import ahocorasick
from rapidfuzz.distance import Levenshtein as RFLev

class Solution:
    def __init__(self):
        self.province_path = 'list_province.txt'
        self.district_path = 'list_district.txt'
        self.ward_path = 'list_ward.txt'

        # --- precompute ---
        self._admin_prefixes = [
            "tinh","thanh pho","tp","quan","q","huyen","h",
            "thi xa","tx","thi tran","tt","xa","phuong","p"
        ]
        self._admin_prefix_re = re.compile(r'\b(?:' + "|".join(self._admin_prefixes) + r')\b')

        self._re_q_num = re.compile(r'\b(?:q|quan)\s*([0-9]{1,2})\b')
        self._re_p_num = re.compile(r'\b(?:p|phuong)\s*([0-9]{1,2})\b')
        self._re_initial_last = re.compile(r'\b([a-z])\s*[\.\-]?\s*([a-z]{2,})\b')

        self.prov_map, self.prov_maxlen, self.prov_abbrev = self._build_index(self.province_path, 'province')
        self.dist_map, self.dist_maxlen, _ = self._build_index(self.district_path, 'district')
        self.ward_map, self.ward_maxlen, _ = self._build_index(self.ward_path, 'ward')

        self._aho_prov = self._build_aho(self.prov_map)
        self._aho_dist = self._build_aho(self.dist_map)
        self._aho_ward = self._build_aho(self.ward_map)

        # Precompute key lists + token info for gated fuzzy
        self._prov_keys = list(self.prov_map.keys())
        self._dist_keys = list(self.dist_map.keys())
        self._ward_keys = list(self.ward_map.keys())

        self._prov_keyinfo = {k: (k.split(), k.split()[-1]) for k in self._prov_keys}
        self._dist_keyinfo = {k: (k.split(), k.split()[-1]) for k in self._dist_keys}
        self._ward_keyinfo = {k: (k.split(), k.split()[-1]) for k in self._ward_keys}

        self._prov_syn = {
            self._canon("hochiminh"): "Hồ Chí Minh",
            self._canon("tp ho chi minh"): "Hồ Chí Minh",
            self._canon("thanh pho ho chi minh"): "Hồ Chí Minh",
            self._canon("tphcm"): "Hồ Chí Minh",
            self._canon("tp.hcm"): "Hồ Chí Minh",
            self._canon("hcm"): "Hồ Chí Minh",
            self._canon("ha noi"): "Hà Nội",
            self._canon("thanh pho ha noi"): "Hà Nội",
            self._canon("tp ha noi"): "Hà Nội",
            self._canon("tp.hn"): "Hà Nội",
            self._canon("hn"): "Hà Nội",
            self._canon("da nang"): "Đà Nẵng",
            self._canon("thanh pho da nang"): "Đà Nẵng",
            self._canon("tp da nang"): "Đà Nẵng",
        }

    # -------------------- Public API --------------------
    def process(self, s: str):
        s_norm = self._canon(s)
        tokens = s_norm.split()
        tokens = self._drop_admin_tokens(tokens)
        
        province, p_span = self._find_province(tokens, s_norm)
        district, d_span = self._find_district(tokens, s_norm, p_span, province)
        ward, w_span = self._find_ward(tokens, s_norm, p_span, d_span, province, district)

        return {"province": province or "", "district": district or "", "ward": ward or ""}

    # -------------------- Finders --------------------
    def _find_province(self, tokens, s_norm):
        n = len(tokens)
        syn = self._prov_syn.get(s_norm)
        if syn:
            return syn, (0, n-1)

        for m in self._re_initial_last.finditer(s_norm):
            c0, last = m.group(1), m.group(2)
            cand = self.prov_abbrev.get((c0, last))
            if cand:
                j = self._rfind_token(tokens, last)
                i = max(0, (j or 0)-1)
                return cand, (i, j if j is not None else n-1)

        hit = self._aho_longest_right(tokens, self._aho_prov)
        if hit:
            i, j, name = hit
            return self._clean_display(name), (i, j)

        fuzz = self._fuzzy_right(tokens, self.prov_maxlen, self.prov_map, self._prov_keys, self._prov_keyinfo,
                                 max_ed_first=1, max_ed_second=2, max_windows=3)
        if fuzz:
            i, j, name = fuzz
            return self._clean_display(name), (i, j)

        for key, disp in self._prov_syn.items():
            if key in " ".join(tokens):
                return disp, (0, n-1)
        return None, None

    def _find_district(self, tokens, s_norm, p_span, province):
        ranges = [(0, len(tokens)-1)]
        if p_span and p_span[0] > 0:
            ranges = [(0, p_span[0]-1)]

        if province in ("Hồ Chí Minh", "Hà Nội"):
            m = self._re_q_num.search(s_norm)
            if m:
                return m.group(1), self._span_by_number(tokens, int(m.group(1)))

        for lo, hi in ranges:
            hit = self._aho_longest_right(tokens[lo:hi+1], self._aho_dist)
            if hit:
                i, j, name = hit
                return self._clean_display(name), (lo+i, lo+j)

        for lo, hi in ranges:
            fuzz = self._fuzzy_right(tokens[lo:hi+1], self.dist_maxlen, self.dist_map,
                                     self._dist_keys, self._dist_keyinfo,
                                     max_ed_first=1, max_ed_second=2, max_windows=3)
            if fuzz:
                i, j, name = fuzz
                return self._clean_display(name), (lo+i, lo+j)
        return None, None

    def _find_ward(self, tokens, s_norm, p_span, d_span, province, district):
        ranges = [(0, len(tokens)-1)]
        if d_span and d_span[0] > 0:
            ranges = [(0, d_span[0]-1)]
        elif p_span and p_span[0] > 0:
            ranges = [(0, p_span[0]-1)]

        if province in ("Hồ Chí Minh", "Hà Nội"):
            m = self._re_p_num.search(s_norm)
            if m:
                return m.group(1), self._span_by_number(tokens, int(m.group(1)))

        for lo, hi in ranges:
            hit = self._aho_longest_right(tokens[lo:hi+1], self._aho_ward)
            if hit:
                i, j, name = hit
                return self._clean_display(name), (lo+i, lo+j)

        for lo, hi in ranges:
            fuzz = self._fuzzy_right(tokens[lo:hi+1], self.ward_maxlen, self.ward_map,
                                     self._ward_keys, self._ward_keyinfo,
                                     max_ed_first=1, max_ed_second=2, max_windows=2)
            if fuzz:
                i, j, name = fuzz
                return self._clean_display(name), (lo+i, lo+j)
        return None, None

    # -------------------- Exact (Aho) --------------------
    def _build_index(self, path, level):
        norm_map, max_len, abbrev = {}, 1, {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except FileNotFoundError:
            lines = []

        for raw in lines:
            disp = raw
            n1 = self._canon(raw)                 # keep admin words
            n2 = self._remove_admin_words(n1)     # remove admin words

            for key in {n1, n2}:
                if key:
                    norm_map[key] = disp
                    max_len = max(max_len, len(key.split()))

            if level == 'province':
                toks = (n2 or n1).split()
                if len(toks) >= 2:
                    abbrev[(toks[0][0], toks[-1])] = disp
        return norm_map, max_len, abbrev if level == 'province' else None

    def _build_aho(self, dictionary):
        A = ahocorasick.Automaton()
        for key, disp in dictionary.items():
            A.add_word(key, (key, disp))
        A.make_automaton()
        return A

    def _aho_longest_right(self, tokens, automaton):
        print("tokens =", tokens, "automaton =", automaton)
        """
        Prefer rightmost end token, then the longest span.
        Only accept matches aligned to token boundaries.
        """
        if not tokens:
            return None
        tstring = " ".join(tokens)
        char2tok = self._char_to_token_map(tokens)

        best = None  # (end_tok, span_len, i, j, disp)
        nchars = len(tstring)
        for end_char, (key, disp) in automaton.iter(tstring):
            start_char = end_char - len(key) + 1

            # ---- NEW: token-boundary guards ----
            left_ok  = (start_char == 0) or (tstring[start_char - 1] == " ")
            right_ok = (end_char   == nchars - 1) or (tstring[end_char + 1] == " ")
            if not (left_ok and right_ok):
                continue
            # ------------------------------------

            i = char2tok[start_char]
            j = char2tok[end_char]
            span_len = j - i + 1
            cand = (j, span_len, i, j, disp)  # rightmost, then longest
            if (best is None) or (cand > best):
                best = cand

        if best:
            _, _, i, j, disp = best
            return (i, j, disp)
        return None


    def _char_to_token_map(self, tokens):
        s = " ".join(tokens)
        char2tok = [0]*len(s)
        pos = 0
        for ti, tok in enumerate(tokens):
            for _ in tok:
                char2tok[pos] = ti
                pos += 1
            if pos < len(s):
                char2tok[pos] = ti  # right-bias spaces
                pos += 1
        return char2tok

    # -------------------- Fuzzy (gated) --------------------
    def _fuzzy_right(self, tokens, max_len, dictionary, keys, keyinfo,
                     max_ed_first=1, max_ed_second=2, max_windows=3):
        if not tokens:
            return None

        # Build rightmost windows (prefer longer)
        windows = []
        n = len(tokens); seen = 0
        for j in range(n-1, -1, -1):
            mw = min(max_len, j+1)
            for w in range(mw, 0, -1):
                i = j - w + 1
                q = " ".join(tokens[i:j+1])
                q_toks = tokens[i:j+1]
                windows.append((i, j, q, q_toks))
                seen += 1
                if seen >= max_windows:
                    break
            if seen >= max_windows:
                break

        # Two passes: ed<=1, then ed<=2
        for ed in (max_ed_first, max_ed_second):
            best = None  # (dist, -span_len, i, j, disp)
            for i, j, q, q_toks in windows:
                qlen = len(q)
                q_last = q_toks[-1]
                q_set = set(q_toks)

                # length gating (char-level)
                lo, hi = qlen - ed, qlen + ed

                # candidate loop (token/last-token gated)
                for key in keys:
                    if not (lo <= len(key) <= hi):
                        continue
                    k_toks, k_last = keyinfo[key]

                    # --- hard guards to prevent drift ---
                    # 1) last token must be very close (<=1 edit)
                    if RFLev.distance(q_last, k_last, score_cutoff=1) > 1:
                        continue
                    # 2) require at least 1 exact token overlap
                    if len(q_set.intersection(k_toks)) == 0:
                        continue

                    d = RFLev.distance(q, key, score_cutoff=ed)
                    if d <= ed:
                        span_len = j - i + 1
                        disp = dictionary[key]
                        cand = (d, -span_len, i, j, disp)
                        if (best is None) or (cand < best):
                            best = cand
            if best:
                _, _, i, j, disp = best
                return (i, j, disp)
        return None

    # -------------------- Utils --------------------
    def _span_by_number(self, tokens, number):
        num = str(number)
        try:
            idx = len(tokens) - 1 - tokens[::-1].index(num)
            return (idx, idx)
        except ValueError:
            return None

    def _rfind_token(self, tokens, tok):
        try:
            return len(tokens) - 1 - tokens[::-1].index(tok)
        except ValueError:
            return None

    def _clean_display(self, name):
        if not name:
            return name
        x = name.strip()
        for p in ["Xã ","Phường ","Thị trấn ","Thị Trấn ","Thị xã ","Thị Xã ",
                  "Quận ","Huyện ","Thành phố ","TP. ","TP ","Tx. ","TX. ","TT. "]:
            if x.startswith(p):
                return x[len(p):].strip()
        return x

    def _remove_admin_words(self, canon_text):
        return self._admin_prefix_re.sub(" ", canon_text).strip()

    def _canon(self, s: str) -> str:
        if not s:
            return ""
        import re, unicodedata
        s = s.lower().replace("đ", "d")
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

        # --- NEW: unglue common admin bigrams stuck together ---
        # e.g. "thanhpho" -> "thanh pho", "thixa" -> "thi xa", "thitran" -> "thi tran"
        s = re.sub(r"\bthanhpho\b", "thanh pho", s)
        s = re.sub(r"\bthixa\b", "thi xa", s)
        s = re.sub(r"\bthitran\b", "thi tran", s)

        # If a token starts with 'pho' and previous token is 'thanh', split it
        # e.g. "thanh phohchiminh" -> "thanh pho hchiminh"
        s = re.sub(r"\bthanh\s+pho(?=[a-z0-9])", "thanh pho ", s)

        # --- NEW: normalize Ho Chi Minh synonyms even when glued ---
        # tphcm / tp.hcm / hcm / hochiminh -> "tp ho chi minh" (so synonyms logic picks it)
        s = re.sub(r"\btp\.?\s*hcm\b", "tp ho chi minh", s)
        s = re.sub(r"\bhochiminh\b", "ho chi minh", s)
        s = re.sub(r"\bhcm\b", "ho chi minh", s)

        return s

    def _drop_admin_tokens(self, tokens):
        """
        Xóa token hành chính khỏi input (cả 1-từ và 2-từ).
        Ví dụ: 'thanh pho', 'tp', 'quan', 'huyen', 'thi xa', 'thi tran', 'phuong', 'xa', 'tx', 'tt', 'q', 'h', 'p'.
        """
        admins1 = {"tp","quan","q","huyen","h","phuong","p","xa","tx","tt"}
        admins2 = {("thanh","pho"), ("thi","xa"), ("thi","tran")}
        out = []
        i = 0
        n = len(tokens)
        while i < n:
            if i+1 < n and (tokens[i], tokens[i+1]) in admins2:
                i += 2  # bỏ cụm 2 từ
                continue
            if tokens[i] in admins1:
                i += 1  # bỏ từ đơn
                continue
            out.append(tokens[i])
            i += 1
        return out
solution = Solution()
import time
input = "Xã Tân Kiên Hbình Chánh Thành phôHôChíMinh"
start_time = time.perf_counter_ns()
result = solution.process(input)
end_time = time.perf_counter_ns()
print(f"Processing time: {(end_time - start_time) / 1_000_000} ms")
print("Processed", input, "->", result)