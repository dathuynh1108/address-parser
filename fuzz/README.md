# Vietnamese Address Parser

Parses Vietnamese addresses in both administrative formats:

- old: `street + ward + district + province`
- new: `street + ward + province`

## Installation

After publishing a release to PyPI:

```bash
python -m pip install vn-address-parser
```

Install the current package directly from this GitHub repository:

```bash
python -m pip install \
  "vn-address-parser @ git+https://github.com/dathuynh1108/address-parser.git@main#subdirectory=fuzz"
```

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

## Native acceleration

`AddressParser` has the same input and `ParseResult` contract with both
backends. Native wheels compile only the packed n-gram ranking kernel in
`address_parser/_native_kernels.pyx` with Cython `annotation_typing=True`.
Normalization, registry resolution, and final result construction remain strict
typed Python.

The native kernel is selected automatically when its extension is installed. A
wheel without the extension uses the typed Python fallback. Import failures from
a present but broken or incompatible native extension are not hidden.

```python
from address_parser import (
    native_acceleration_available,
    require_native_acceleration,
)

if native_acceleration_available():
    print("native backend active")

# Recommended during service startup when native performance is mandatory.
require_native_acceleration()
```

`VN_ADDRESS_PARSER_NATIVE` is a wheel build setting, not a runtime switch:

- `required`: compile the extension and fail the build on any native error. Use
  this mode for staging and production artifacts.
- `optional`: try to compile the extension and allow a Python-only wheel when a
  compiler is unavailable. This is the default; a skipped optional extension
  still produces a platform-tagged wheel.
- `disabled`: deliberately build a Python-only wheel.

```bash
VN_ADDRESS_PARSER_NATIVE=required \
  ./.venv/bin/python -m build --wheel --outdir /tmp/address-parser-native

VN_ADDRESS_PARSER_NATIVE=disabled \
  ./.venv/bin/python -m build --wheel --outdir /tmp/address-parser-fallback
```

Installing a prebuilt wheel does not require Cython or a compiler. Build native
wheels on the target operating system and Python ABI used by the service.

## Development

Use the repository virtual environment:

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/mypy
./.venv/bin/python -m unittest \
  test_parser_contract.py \
  test_parser_regression.py \
  test_parser_new_format.py \
  test_street_prefix_guard.py \
  test_performance_kernels.py
./.venv/bin/python -m unittest test_wheel_package.py
./.venv/bin/python -m unittest test_full_dataset_regression.py
```

Run the deterministic steady-state parser benchmark with:

```bash
./.venv/bin/python -m benchmarks.benchmark_parser \
  --old-cases 150 \
  --new-cases 150 \
  --rounds 5 \
  --require-native
```

## License

The software is released under the MIT License. See
[DATA_SOURCES.md](https://github.com/dathuynh1108/address-parser/blob/main/fuzz/DATA_SOURCES.md)
for bundled administrative registry provenance and upstream attribution.
