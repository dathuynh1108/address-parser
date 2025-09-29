from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from collections import defaultdict
import unicodedata, re, os
from rapidfuzz import fuzz, process
# ============== VN Normalization & Helpers =================

_WS_RE = re.compile(r"\s+")
_PUNCT_AS_SPACE = re.compile(r"[^\w\s]", flags=re.UNICODE)

# Designator tokens that may appear in the QUERY (never in your lists)
_DESIGNATOR_TOKENS = {
    # province/city
    "tinh","thanh","pho","thanhpho","tp","t","t p","t p","tx","t x","thi","thi_xa","thi xa",
    # district-level
    "quan","q","huyen","h","thi_tran","thi tran","tt",
    # ward-level
    "phuong","p","xa","x",
}

_ALIAS_PATTERNS = {
    # --- Numeric expansions ---
    r"\bq\s*0*([1-9][0-9]?)\b": r"quan \1",        # Q3, Q.03 → quan 3
    r"\bp\s*0*([1-9][0-9]?)\b": r"phuong \1",      # P1, P.01 → phuong 1
    r"\btx\b": "thi xa",
    r"\btt\b": "thi tran",
    r"\btp\b": "thanh pho",

    # --- Hồ Chí Minh ---
    r"\b(tp\s*\.?\s*hcm|tphcm|hcm|sai\s*gon)\b": "ho chi minh",

    # --- Thừa Thiên Huế ---
    r"\b(thuathienhue|tt\s*hue|t\s*thien\s*hue|tth|t\s*t\s*h)\b": "thua thien hue",

    # --- Hà Nội ---
    r"\b(hn|tp\s*hn)\b": "ha noi",
}

def _strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def _normalize_core(s: str) -> str:
    s = s.replace(".", " ")
    s = _strip_diacritics(s).lower()
    s = _PUNCT_AS_SPACE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _expand_aliases(s: str) -> str:
    for pat, repl in _ALIAS_PATTERNS.items():
        s = re.sub(pat, repl, s)
    return s

def _remove_designators(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _DESIGNATOR_TOKENS]

def _normalize_query_for_match(s: str) -> str:
    s = _normalize_core(s)
    s = _expand_aliases(s)           # <--- regex replacements
    tokens = s.split()
    toks = _remove_designators(tokens)
    print(toks)
    return " ".join(toks)

def _normalize_catalog_name(s: str) -> str:
    return _normalize_core(s)

def _initials(s: str) -> str:
    return "".join(tok[0] for tok in s.split() if tok)

def _ngrams(s: str, n: int = 3) -> List[str]:
    s2 = f" {s} "
    if len(s2) < n:
        return [s2]
    return [s2[i:i+n] for i in range(len(s2) - n + 1)]

# =========================
# N-gram index
# =========================

@dataclass
class _Entry:
    id: int
    name: str
    norm: str
    initials: str

class _NGramIndex:
    def __init__(self, n: int = 3):
        self.n = n
        self.entries: List[_Entry] = []
        self.inv: Dict[str, List[int]] = defaultdict(list)
        self._cached_ngrams: List[Optional[set]] = []
        self._norm_list: List[str] = []

    def add(self, name: str) -> int:
        sid = len(self.entries)
        ns = _normalize_catalog_name(name)
        ins = _initials(ns)
        self.entries.append(_Entry(sid, name, ns, ins))
        self._norm_list.append(ns)
        grams = set(_ngrams(ns, self.n))
        for g in grams:
            self.inv[g].append(sid)
        self._cached_ngrams.append(grams)
        return sid

    def shortlist(self, nq: str, limit: int, dice_min: float) -> List[int]:
        qgrams = set(_ngrams(nq, self.n))
        counts = defaultdict(int)
        for g in qgrams:
            for sid in self.inv.get(g, []):
                counts[sid] += 1
        if not counts:
            return list(range(min(len(self.entries), limit)))

        pre = []
        for sid in counts.keys():
            sgrams = self._cached_ngrams[sid]
            inter = len(qgrams & sgrams)
            dice = 0.0 if inter == 0 else (2.0 * inter) / (len(qgrams) + len(sgrams))
            if dice >= dice_min:
                pre.append((sid, dice))
        if not pre:
            return list(range(min(len(self.entries), limit)))
        pre.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in pre[:limit]]

    
    def rf_extract_one(self, nq: str, choices: list[str], cutoff: float):
        if not choices:
            return None  # nothing to match
        
        res = process.extractOne(nq, choices, scorer=fuzz.partial_ratio, score_cutoff=cutoff)
        if res is None:
            return None
        choice, score, idx = res
        return choice, float(score), idx

    def extract_one(self, nq: str, pr_min: float, shortlist_sids: list[int]):
        if not shortlist_sids:
            return None
        
        choices = [self._norm_list[sid] for sid in shortlist_sids]

        # 1st try: strict cutoff
        got = self.rf_extract_one(nq, choices, pr_min)
        if got is None:
            return None

        choice, base, idx = got
        sid = shortlist_sids[idx]
        e = self.entries[sid]
        return (sid, e.name, e.norm, base)

# =========================
# Token removal
# =========================


def _remove_by_span(q_norm: str, cand_norm: str, *, cutoff: float = 60.0) -> str:
    """
    Remove the exact matched substring of `cand_norm` inside `q_norm`
    using RapidFuzz alignment. Falls back to token-multiset removal if
    no substring alignment meets cutoff.
    """
    if not q_norm or not cand_norm:
        return q_norm

    # 1) Try substring alignment (precise start/end indices in the QUERY)
    #    Works great for "tp hcm" vs "ho chi minh" AFTER alias expansion.
    try:
        result = fuzz.partial_ratio_alignment(q_norm, cand_norm, score_cutoff=cutoff)
        # If we’re here, score >= cutoff and we have a concrete span [qs:qe)
        new_q = (q_norm[:result.src_start] + " " + q_norm[result.src_end:]).strip()
        return re.sub(r"\s+", " ", new_q)
    except Exception as e:
        print(e)
        pass  # RF v2/v3 both have this; the try/except is just defensive

    # 2) Fallback: remove by token multiset ONCE (order-insensitive)
    q_tokens = q_norm.split()
    need = {}
    for t in cand_norm.split():
        need[t] = need.get(t, 0) + 1

    out = []
    for t in q_tokens:
        if need.get(t, 0) > 0:
            need[t] -= 1
            continue
        out.append(t)
    return " ".join(out)

# =========================
# Solution class
# =========================

class Solution:
    def __init__(self):
        self.province_path = 'list_province.txt'
        self.district_path = 'list_district.txt'
        self.ward_path     = 'list_ward.txt'

        self.province_idx = _NGramIndex(n=3)
        self.district_idx = _NGramIndex(n=3)
        self.ward_idx     = _NGramIndex(n=3)

        def _load_lines(path: str) -> List[str]:
            if not os.path.exists(path):
                return []
            out, seen = [], set()
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if not name: continue
                    key = _normalize_catalog_name(name)
                    if key in seen: continue
                    seen.add(key)
                    out.append(name)
            return out

        for name in _load_lines(self.province_path):
            self.province_idx.add(name)
        for name in _load_lines(self.district_path):
            self.district_idx.add(name)
        for name in _load_lines(self.ward_path):
            self.ward_idx.add(name)

        self.province_min = 68.0
        self.district_min = 64.0
        self.ward_min     = 60.0

    def _pick_one(self, idx: _NGramIndex, nq: str, pr_min: float, dice_min: float, short_lim: int):
        sids = idx.shortlist(nq, limit=short_lim, dice_min=dice_min)
        return idx.extract_one(nq, pr_min=pr_min, shortlist_sids=sids)

    def process(self, s: str):
        nq0 = _normalize_query_for_match(s)
        print("Normalized query:", nq0)

        # Province first (end of string)
        pickP = self._pick_one(self.province_idx, nq0, self.province_min, 0.15, 200)
        pname, nq1 = "", nq0
        if pickP:
            _, pname, pnorm, _ = pickP
            nq1 = _remove_by_span(nq0, pnorm, cutoff=self.province_min)
            print("  After province removal:", nq1)

        # District second (middle)
        pickD = self._pick_one(self.district_idx, nq1, self.district_min, 0.15, 400)
        dname, nq2 = "", nq1
        if pickD:
            _, dname, dnorm, _ = pickD
            nq2 = _remove_by_span(nq1, dnorm, cutoff=self.district_min)
            print("  After district removal:", nq2)

        # Ward last (front)
        pickW = self._pick_one(self.ward_idx, nq2, self.ward_min, 0.12, 600)
        wname = ""
        if pickW:
            _, wname, _, _ = pickW
            print("  After ward removal: (not needed, just for debug)")

        return {"province": pname or "", "district": dname or "", "ward": wname or ""}
        
    
solution = Solution()
import time
start_time = time.perf_counter_ns()
result = solution.process("TT Tân Bình Huyện Yên Sơn, Tuyên Quang")
end_time = time.perf_counter_ns()
print(f"Processing time: {(end_time - start_time) / 1_000_000} ms, result = {result}")