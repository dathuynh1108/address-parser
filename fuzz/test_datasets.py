from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from test_new_address import main as run_new
from test_old_address import main as run_old


def _run_script(entrypoint, argv: List[str]) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        return int(entrypoint())
    finally:
        sys.argv = old_argv


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="Run both old and new dataset checks to prevent regressions."
    )
    ap.add_argument("--limit-old", type=int, default=0, help="Limit old cases (0 = all)")
    ap.add_argument("--limit-new", type=int, default=0, help="Limit new cases (0 = all)")
    ap.add_argument("--show", type=int, default=5, help="Show up to N failures per suite")
    ap.add_argument(
        "--strict-new-prefix",
        action="store_true",
        help="Fail if any new-dataset address contains district prefix",
    )
    args = ap.parse_args()

    old_args: List[str] = ["--show", str(args.show)]
    if args.limit_old:
        old_args += ["--limit", str(args.limit_old)]

    new_args: List[str] = ["--show", str(args.show)]
    if args.limit_new:
        new_args += ["--limit", str(args.limit_new)]
    if args.strict_new_prefix:
        new_args += ["--strict-prefix"]

    print(json.dumps({"suite": "old"}, ensure_ascii=False))
    old_rc = _run_script(run_old, old_args)

    print(json.dumps({"suite": "new"}, ensure_ascii=False))
    new_rc = _run_script(run_new, new_args)

    combined = {"old_exit": old_rc, "new_exit": new_rc}
    print(json.dumps(combined, ensure_ascii=False))
    return 0 if old_rc == 0 and new_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

