from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import unicodedata, re, time

# -----------------------------
# Normalization (VN-friendly)
# -----------------------------
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").lower()
    s = re.sub(r"[^\w\s]", " ", s)       # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ngrams(s: str, n: int = 3) -> List[str]:
    s2 = f" {s} "  # boundary padding
    if len(s2) < n:
        return [s2]
    return [s2[i:i+n] for i in range(len(s2) - n + 1)]

# --- RapidFuzz-style partial ratio (VN friendly) -----------------------------

def seller_distance(pattern: str, text: str) -> int:
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

def partial_ratio(a: str, b: str) -> float:
    """
    RapidFuzz-like partial_ratio in [0, 100], but always using your normalize().
    """
    if not a and not b: return 100.0
    if not a or not b:  return 0.0
    # shorter is the needle
    if len(a) <= len(b):
        pat, txt = a, b
    else:
        pat, txt = b, a
    dist = seller_distance(pat, txt)
    return 100.0 * (1.0 - dist / max(1, len(pat)))


# -----------------------------
# N-gram index with metadata
# -----------------------------
@dataclass
class Entry:
    id: int
    name: str           # original
    norm: str           # normalized
    meta: dict          # e.g., {"province_id": 79}

class NGramIndex:
    def __init__(self, n: int = 3):
        self.n = n
        self.inv: Dict[str, List[int]] = defaultdict(list)   # gram -> [sid]
        self.entries: List[Entry] = []
        self._cached_ngrams: List[Optional[Set[str]]] = []

    def add(self, name: str, meta: Optional[dict] = None) -> int:
        sid = len(self.entries)
        ns = normalize(name)
        self.entries.append(Entry(sid, name, ns, meta or {}))
        grams = set(ngrams(ns, self.n))
        for g in grams:
            self.inv[g].append(sid)
        self._cached_ngrams.append(grams)
        return sid

    def _dice(self, A: Set[str], B: Set[str]) -> float:
        inter = len(A & B)
        return 0.0 if inter == 0 else (2.0 * inter) / (len(A) + len(B))

    def query(
        self,
        q: str,
        shortlist: int = 300,
        k: int = 10,
        dice_min: float = 0.25,
        max_dist: int = 2,
        pr_min: float = 60.0,          # ### CHANGED: partial ratio minimum
        allowed_ids: Optional[Set[int]] = None,
        allowed_parent_key: Optional[str] = None,
        allowed_parent_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, str, int, float]]:
        """
        Returns a list of (sid, original_name, dist_proxy, dice_score)
        dist_proxy is derived from partial_ratio: round((1 - pr) * len(candidate))
        """
        nq = normalize(q)
        if not nq:
            return []

        # stricter on tiny queries
        if len(nq) <= 2:
            dice_min = max(dice_min, 0.50)
            pr_min = max(pr_min, 80.0)

        qgrams = set(ngrams(nq, self.n))
        # print("Checking grams:", qgrams)  # debug

        # 1) Collect candidates by gram overlap (+ constraints)
        counts: Dict[int, int] = defaultdict(int)
        for g in qgrams:
            for sid in self.inv.get(g, []):
                if allowed_ids is not None and sid not in allowed_ids:
                    continue
                if allowed_parent_key and allowed_parent_ids is not None:
                    if self.entries[sid].meta.get(allowed_parent_key) not in allowed_parent_ids:
                        continue
                counts[sid] += 1
        if not counts:
            return []

        # 2) Dice prefilter
        pre = []
        for sid in counts.keys():
            sgrams = self._cached_ngrams[sid]
            dsc = self._dice(qgrams, sgrams)
            if dsc >= dice_min:
                pre.append((sid, dsc))
        if not pre:
            return []

        pre.sort(key=lambda x: x[1], reverse=True)
        pre = pre[:shortlist]

        # 3) ### CHANGED: substring-aware similarity (partial_ratio)
        finals = []
        for sid, dsc in pre:
            e = self.entries[sid]
            pr = partial_ratio(e.norm, nq)  # 0..100
            print("Checking", e.norm, "<>", nq,  "pr=", pr, "dice=", dsc)  # debug
            # print(f"  {e.name}: pr={pr:.1f} dice={dsc:.2f}")  # debug
            if pr < pr_min:
                continue
            dist_proxy = int(round((1.0 - pr / 100.0) * max(1, len(e.norm))))
            is_pref = e.norm.startswith(nq) or nq.startswith(e.norm)
            finals.append((sid, e.name, dist_proxy, dsc, len(e.norm), is_pref, pr))
        print(finals)
        if not finals:
            return []

        # Sort: higher pr, then lower dist, then higher dice, then prefix first, then shorter
        finals.sort(key=lambda x: (-x[6], x[2], -x[3], not x[5], x[4]))
        return [(sid, name, dist_proxy, dsc) for sid, name, dist_proxy, dsc, _, _, _ in finals[:k]]

# ------------------------------------------
# Demo data (tiny, illustrative, not full)
# ------------------------------------------
PROVINCES = [
    (79, "TP Hồ Chí Minh"),
    (82, "Tiền Giang"),
    (46, "Thừa Thiên Huế"),
    (49, "Quảng Nam"),
]

DISTRICTS = [
    (760, "Quận 1", 79),
    (761, "Quận 12", 79),
    (815, "Châu Thành", 82),
    (816, "Cai Lậy", 82),
    (490, "Thành phố Huế", 46),
    (502, "Hội An", 49),
]

WARDS = [
    (26734, "Phường Bến Nghé", 760),
    (26842, "Phường Thạnh Xuân", 761),
    (28252, "Xã Vĩnh Kim", 815),
    (28258, "Xã Tân Lý Đông", 815),
    
    (20113, "Phường Phú Hội", 490),
    (20545, "Phường Minh An", 502),
]

# ------------------------------------------
# Build indices
# ------------------------------------------
province_idx = NGramIndex(n=3)
district_idx = NGramIndex(n=3)
ward_idx     = NGramIndex(n=3)

prov_id_to_sid: Dict[int, int] = {}
dist_id_to_sid: Dict[int, int] = {}
ward_id_to_sid: Dict[int, int] = {}

for pid, pname in PROVINCES:
    sid = province_idx.add(pname, meta={"province_id": pid})
    prov_id_to_sid[pid] = sid

for did, dname, pid in DISTRICTS:
    sid = district_idx.add(dname, meta={"district_id": did, "province_id": pid})
    dist_id_to_sid[did] = sid

for wid, wname, did in WARDS:
    sid = ward_idx.add(wname, meta={"ward_id": wid, "district_id": did})
    ward_id_to_sid[wid] = sid

# ------------------------------------------
# Hierarchical search
# ------------------------------------------
@dataclass
class Match:
    province_id: Optional[int]
    district_id: Optional[int]
    ward_id: Optional[int]
    score: float
    pieces: Dict[str, Tuple[str, int, float]]  # level -> (name, dist_proxy, dice)

def combined_score(dist_len_pairs: List[Tuple[int, int]], weights: List[float]) -> float:
    """
    Lower is better. Each component is (dist_proxy / (len + 1)) weighted.
    dist_proxy comes from partial_ratio, so it scales with candidate length.
    """
    num = 0.0
    den = max(1e-9, sum(weights))
    for (dist_proxy, L), w in zip(dist_len_pairs, weights):
        num += w * (dist_proxy / (L + 1))
    return num / den

def search_address(
    query: str,
    topP: int = 5,
    topD_per_P: int = 5,
    topW_per_D: int = 5,
) -> List[Match]:
    """
    Beam search: Province -> District (constrained) -> Ward (constrained)
    Uses substring-aware NGramIndex.query so long free-text works.
    """
    results: List[Match] = []

    # Step 1: Province candidates
    P = province_idx.query(query, k=topP, dice_min=0.20, max_dist=3, pr_min=55.0)

    if not P:
        P = []  # still try district-first

    for sidP, pname, pdist, pdice in P:
        prov = province_idx.entries[sidP]
        pid = prov.meta["province_id"]

        # Step 2: District candidates constrained by province
        D = district_idx.query(
            query,
            k=topD_per_P,
            dice_min=0.18, max_dist=3, pr_min=55.0,
            allowed_parent_key="province_id",
            allowed_parent_ids={pid},
        )

        if not D:
            score = combined_score([(pdist, len(prov.norm))], weights=[1.0])
            results.append(Match(pid, None, None, score, pieces={
                "province": (pname, pdist, pdice)
            }))
            continue

        for sidD, dname, ddist, ddice in D:
            dist_e = district_idx.entries[sidD]
            did = dist_e.meta["district_id"]

            # Step 3: Ward candidates constrained by district
            W = ward_idx.query(
                query,
                k=topW_per_D,
                dice_min=0.15, max_dist=3, pr_min=55.0,
                allowed_parent_key="district_id",
                allowed_parent_ids={did},
            )

            if not W:
                score = combined_score(
                    [(pdist, len(prov.norm)), (ddist, len(dist_e.norm))],
                    weights=[0.5, 0.5]
                )
                results.append(Match(pid, did, None, score, pieces={
                    "province": (pname, pdist, pdice),
                    "district": (dname, ddist, ddice),
                }))
                continue

            for sidW, wname, wdist, wdice in W:
                ward_e = ward_idx.entries[sidW]
                score = combined_score(
                    [
                        (pdist, len(prov.norm)),
                        (ddist, len(dist_e.norm)),
                        (wdist, len(ward_e.norm))
                    ],
                    weights=[0.4, 0.35, 0.25]
                )
                results.append(Match(pid, did, ward_e.meta["ward_id"], score, pieces={
                    "province": (pname, pdist, pdice),
                    "district": (dname, ddist, ddice),
                    "ward":     (wname, wdist, wdice),
                }))

    # District-first start (helps when query mostly names a district/ward)
    if not results:
        D = district_idx.query(query, k=topD_per_P, dice_min=0.18, max_dist=3, pr_min=55.0)
        for sidD, dname, ddist, ddice in D:
            dist_e = district_idx.entries[sidD]
            did = dist_e.meta["district_id"]
            pid = dist_e.meta["province_id"]

            W = ward_idx.query(
                query,
                k=topW_per_D,
                dice_min=0.15, max_dist=3, pr_min=55.0,
                allowed_parent_key="district_id",
                allowed_parent_ids={did},
            )
            if not W:
                score = combined_score([(ddist, len(dist_e.norm))], weights=[1.0])
                results.append(Match(pid, did, None, score, pieces={
                    "district": (dname, ddist, ddice)
                }))
            else:
                for sidW, wname, wdist, wdice in W:
                    ward_e = ward_idx.entries[sidW]
                    score = combined_score(
                        [(ddist, len(dist_e.norm)), (wdist, len(ward_e.norm))],
                        weights=[0.6, 0.4]
                    )
                    results.append(Match(pid, did, ward_e.meta["ward_id"], score, pieces={
                        "district": (dname, ddist, ddice),
                        "ward":     (wname, wdist, wdice),
                    }))

    results.sort(key=lambda m: (m.score, repr(m.pieces)))  # lower is better
    return results[:10]

# ------------------------------------------
# Demo
# ------------------------------------------
if __name__ == "__main__":
    tests = [
        "105 Vĩnh Hoà, Vĩnh Kim, Châu Thành, Tiềz Giang",
        "Tân Lý Đông, Cheo Thành, Tiền Gian"
    ]
    for q in tests:
        print(f"\nQuery: {q}")
        rs = search_address(q)
        if not rs:
            print("  (no results)")
            continue
        for m in rs[:3]:
            def safe_name(piece): return piece[0] if piece else None
            p = safe_name(m.pieces.get("province"))
            d = safe_name(m.pieces.get("district"))
            w = safe_name(m.pieces.get("ward"))
            t0 = time.perf_counter()
            print(f"  -> score={m.score:.4f} | Province={p} | District={d} | Ward={w}")
            t1 = time.perf_counter()
            print(f"     Time taken: {t1 - t0:.6f}s")
