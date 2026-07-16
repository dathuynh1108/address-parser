import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Callable, Dict, List, Optional


def _normalize_value(value: str) -> str:
    return value.lower().strip() if value else ""


def _normalize_result(entry: Dict) -> Dict:
    normalized = {
        "ctryname": _normalize_value(entry.get("ctryname")),
        "ctrysubdivname": _normalize_value(entry.get("ctrysubdivname")),
        "ctrysubsubdivname": [],
    }
    for item in entry.get("ctrysubsubdivname") or []:
        normalized["ctrysubsubdivname"].append(_normalize_value(item))
    return normalized


def _load_reference_parser(repo_root: str):
    maso_root = os.path.join(
        repo_root, "..", "..", "iNexus", "chatbot", "crawler", "masothue"
    )
    if maso_root not in sys.path:
        sys.path.insert(0, maso_root)
    try:
        from masothue.utils import parse_full_address  # type: ignore

        return parse_full_address
    except ModuleNotFoundError:
        return None


def _load_new_parser():
    from address_parser import AddressParser

    return AddressParser()


def _adapt_new_result(result: Dict) -> Dict:
    province = (result.get("province") or {}).get("name") or ""
    district = (result.get("district") or {}).get("name") or ""
    ward = (result.get("ward") or {}).get("name") or ""
    street = result.get("street_address") or ""
    is_new = result.get("format") == "new" or result.get("is_new") is True

    def norm(value: str) -> str:
        return value.lower().strip()

    province = norm(province) if province else ""
    district = norm(district) if district else ""
    ward = norm(ward) if ward else ""
    street = norm(street) if street else ""

    if is_new:
        return {
            "ctryname": province,
            "ctrysubdivname": ward,
            "ctrysubsubdivname": [street] if street else [],
        }

    subdiv: List[str] = []
    if street:
        subdiv.append(street)
    if ward:
        subdiv.append(ward)
    return {
        "ctryname": province,
        "ctrysubdivname": district,
        "ctrysubsubdivname": subdiv,
    }


def evaluate(
    dataset_path: str,
    repo_root: str,
    max_samples: int,
    *,
    baseline: str,
    baseline_field: str,
) -> Dict:
    ref_parser = _load_reference_parser(repo_root) if baseline == "masothue" else None
    new_parser = _load_new_parser()
    summary = Counter()
    mismatches = []
    processed = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and processed >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            address = payload.get("mst_address") or ""
            if ref_parser is not None:
                ref = _normalize_result(ref_parser(address))
            else:
                baseline_value = payload.get(baseline_field)
                if not isinstance(baseline_value, dict):
                    raise ValueError(
                        f"Dataset mode requires '{baseline_field}' dict field, missing at id={payload.get('id')}"
                    )
                ref = _normalize_result(baseline_value)
            new = _normalize_result(_adapt_new_result(new_parser.process(address)))

            diff = False
            for field in ("ctryname", "ctrysubdivname", "ctrysubsubdivname"):
                if ref[field] != new[field]:
                    summary[field] += 1
                    diff = True
            if diff:
                mismatches.append(
                    {
                        "id": payload.get("id"),
                        "address": address,
                        "baseline": ref,
                        "new": new,
                    }
                )
            processed += 1
    return {
        "total": processed,
        "diff_counts": dict(summary),
        "mismatches": mismatches,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    default_dataset = os.path.join(
        "first_1000_parsed_addresses.jsonl",
    )
    parser = argparse.ArgumentParser(
        description="Compare inexus parser against Masothue baseline."
    )
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help="Path to the benchmark JSONL file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of records to evaluate.",
    )
    parser.add_argument(
        "--baseline",
        choices=("dataset", "masothue"),
        default="dataset",
        help="Which baseline to compare against.",
    )
    parser.add_argument(
        "--baseline-field",
        default="old_parsed",
        help="When --baseline=dataset, use this field as baseline dict.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=20,
        help="How many mismatch samples to print.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    report = evaluate(
        args.dataset,
        repo_root,
        args.limit,
        baseline=args.baseline,
        baseline_field=args.baseline_field,
    )
    print(f"Total records: {report['total']}")
    print(f"Field diff counts: {report['diff_counts']}")

    mismatches = report["mismatches"][: args.max_mismatches]
    if not mismatches:
        print("No mismatches found.")
        return

    print("\nSample mismatches:")
    for sample in mismatches:
        print(json.dumps(sample, ensure_ascii=False))


if __name__ == "__main__":
    main()
