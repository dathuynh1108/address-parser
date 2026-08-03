# Administrative data sources

The wheel contains normalized Vietnamese administrative registry data required
by the parser. Local normalization and corrections mean the packaged files are
not necessarily byte-for-byte copies of their upstream sources.

- `old_provinces.json`, `old_districts.json`, `old_wards.json`, `provinces.json`,
  and `wards.json` are adapted from the
  [Vietnamese Provinces Database](https://github.com/thanglequoc/vietnamese-provinces-database).
  That project derives its registry from Vietnam's General Statistics Office
  administrative-unit API and is distributed under the
  [MIT License](https://github.com/thanglequoc/vietnamese-provinces-database/blob/master/LICENSE),
  copyright 2021 Thang Le Quoc.
- `ward_mappings.json` is adapted from the
  [Vietnam Address Database](https://github.com/quangtam/vietnam-address-database)
  by Quang Tam. Its package metadata declares the dataset MIT-licensed and based
  on Resolution 202/2025/QH15 and the relevant administrative reorganization
  resolutions.
- `address_parser.preprocessed.v104.pkl` is a derived parser cache generated from
  the six JSON files listed above. It does not add a separate registry source.

The upstream projects do not endorse this package. Data is provided as-is; users
should verify administrative records against current official sources when they
need authoritative results.

The applicable upstream copyright and permission notices are reproduced in
`THIRD_PARTY_NOTICES.md` and are included in built distributions.
