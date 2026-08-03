# AGENTS

## Scope

This repo parses Vietnamese addresses across two administrative shapes:

- old format: `street + ward + district + province`
- new format: `street + ward + province`

The parser is sensitive to collisions between:

- raw names vs normalized names
- canonical names vs legacy aliases
- old registry vs new registry

Do not treat old-to-new mapping as permission to overwrite the canonical parse result.

## Environment

Use the repo virtualenv, not system Python:

```bash
./.venv/bin/python ...
```

## Primary Files

- `address_parser/parser.py`: main parser logic
- `address_parser/contracts.py`: public input and result contracts
- `address_parser/search_engine.py`: typed BM25 search implementation
- `test_parser_regression.py`: curated regression cases
- `test_parser_new_format.py`: focused new-format behavior
- `test_street_prefix_guard.py`: street/locality false-positive guards
- `test_wheel_package.py`: isolated native/fallback wheel build and install smoke
- `full_dataset_regression_cases.py`: exhaustive synthetic-case builder
- `test_full_dataset_regression.py`: full old/new dataset sweep

## Verification Workflow

Install development-only type and lint tools when needed:

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
```

Run the strict type gate for library and corpus-builder code:

```bash
./.venv/bin/mypy
```

Run this first for fast feedback:

```bash
./.venv/bin/python -m unittest test_parser_contract.py test_parser_regression.py test_parser_new_format.py test_street_prefix_guard.py
```

Run this before closing parser changes:

```bash
./.venv/bin/python -m unittest test_wheel_package.py
./.venv/bin/python -m unittest test_full_dataset_regression.py
```

Current exhaustive sweep size is 14,082 cases. It takes about 1.5 minutes with
the native kernel and 2 minutes with the typed Python fallback on this machine.

If you want to regenerate the synthetic corpus explicitly:

```bash
./.venv/bin/python full_dataset_regression_cases.py --output /tmp/address_parser_full_dataset_cases.json
```

## Guardrails

- Exact raw ward names must beat normalized collisions such as `Van Lang` vs `Van Lang`, `Tam Dan` vs `Tam Dan`, etc.
- Do not infer `province` or `district` just because a bare ward matches a unique registry entry. If the input has no region hint, preserve `None` when that is the current contract.
- `KP` and `Khu pho` should be treated as street/locality fragments first. They must not silently become ward text without supporting context.
- Prefix detection must only read administrative tokens at the start of the segment. Do not let a name token like `Phuong` or `Phuong Vien` get misread as `phuong`.
- When explicit district-level hints exist (`Huyen`, `Quan`, `Thi xa`, etc.), old-format parsing should win unless there is a very strong reason otherwise.
- New-format province IDs/codes must come from the new registry, not from an old province record with the same display name.

## When Full Sweep Fails

Do not patch only the first handwritten test and stop.

1. Reproduce the failing case directly with `parser.process(...)`.
2. Check whether the miss comes from:
   - raw-name collision
   - alias leakage
   - old/new registry confusion
   - over-eager province or district inference
   - prefix stripping or unit-token detection
3. Fix the parser.
4. Re-run the focused suite.
5. Re-run the exhaustive sweep.

## Notes

- `demo.py` is useful for ad hoc repros, but it is not the source of truth for verification.
- Keep changes compatible with both curated regressions and the exhaustive dataset sweep.
