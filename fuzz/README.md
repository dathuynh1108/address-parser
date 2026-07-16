# Vietnamese Address Parser

Parses Vietnamese addresses in both administrative formats:

- old: `street + ward + district + province`
- new: `street + ward + province`

## Usage

```python
from address_parser import AddressParser

parser = AddressParser()
result = parser.process(
    "Số 27, Ngõ 92 Nguyễn Khánh Toàn, Phường Quan Hoa, Quận Cầu Giấy, Hà Nội"
)
```

`process()` accepts a string and returns a six-field `ParseResult`:

```python
{
    "province": component_or_none,
    "district": component_or_none,
    "ward": component_or_none,
    "street_address": "...",
    "format": "old" | "new" | "unknown",
    "is_new": False | True | None,
}
```

The discriminator fields are correlated: `old` maps to `False`, `new` maps to
`True`, and `unknown` maps to `None`. A component always contains `name`; its
`id`, `code`, `full_name`, `aliases`, and `legacy_names` fields are optional.

Administrative code inputs accept `str | int`. Use strings when leading zeroes
must be preserved. Boolean, float, and arbitrary object values are rejected.

`standardize_name(name, mode)` accepts exactly `"basic"`, `"search"`, or
`"aggressive"`; `name` must be a string. Search queries accept `str | None`,
and registry selectors such as `is_new_format`, `include_new`, and `include_old`
must be real booleans rather than truthy values.

## Development

Use the repository virtual environment:

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/mypy
./.venv/bin/python -m unittest \
  test_parser_contract.py \
  test_parser_regression.py \
  test_parser_new_format.py \
  test_street_prefix_guard.py
./.venv/bin/python -m unittest test_wheel_package.py
./.venv/bin/python -m unittest test_full_dataset_regression.py
```
