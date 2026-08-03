"""Static contracts and schema keys shared by parser and search engine.

Runtime parsing stays on plain dictionaries and built-in containers so these
contracts do not add object-model overhead to the CPU-bound path or prevent later
Cython compilation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Final, Literal, TypeAlias, TypedDict

AddressCode: TypeAlias = str
AddressCodeInput: TypeAlias = str | int
AddressFormat: TypeAlias = Literal["old", "new", "unknown"]
RegistryFormat: TypeAlias = Literal["old", "new"]
NormalizationMode: TypeAlias = Literal["basic", "search", "aggressive"]
AdministrativeLevel: TypeAlias = Literal["province", "district", "ward"]
RegistrySource: TypeAlias = Literal["old", "new"]
MappingDirection: TypeAlias = Literal["old_to_new", "new_to_old"]

DatasetFileSignature: TypeAlias = tuple[str, float | None, int | None]
DatasetSignature: TypeAlias = tuple[DatasetFileSignature, ...]
DetectedComponents: TypeAlias = tuple[str | None, str | None, str | None]
NgramHit: TypeAlias = tuple[int, int]
CandidateHit: TypeAlias = tuple[int, float, str]
FuzzyChoiceProfile: TypeAlias = tuple[str, str, int, int, str, str]


class _AdministrativeRecordOptionalFields(TypedDict, total=False):
    """Optional fields carried by normalized administrative records."""

    full_name: str | None
    name_en: str | None
    full_name_en: str | None
    code_name: str | None
    parent_code: AddressCode | None
    province_code: AddressCode | None
    district_code: AddressCode | None
    administrative_region_id: int | None
    administrative_unit_id: int | None
    is_new_format: bool
    legacy_names: list[str]
    aliases: list[str]
    slug: str | None
    type: str | None
    path: str | None
    path_with_type: str | None
    province_name: str | None
    province_key: str | None
    province_id: AddressCode | None
    district_name: str | None
    district_key: str | None
    district_id: AddressCode | None
    ward_name: str | None
    ward_key: str | None


class AdministrativeRecord(_AdministrativeRecordOptionalFields):
    """Normalized province, district, or ward record keyed by canonical code."""

    code: AddressCode | None
    id: AddressCode | None
    name: str | None


ADMINISTRATIVE_RECORD_REQUIRED_KEYS: Final[frozenset[str]] = frozenset({"code", "id", "name"})
ADMINISTRATIVE_RECORD_OPTIONAL_STRING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "full_name",
        "name_en",
        "full_name_en",
        "code_name",
        "parent_code",
        "province_code",
        "district_code",
        "slug",
        "type",
        "path",
        "path_with_type",
        "province_name",
        "province_key",
        "province_id",
        "district_name",
        "district_key",
        "district_id",
        "ward_name",
        "ward_key",
    }
)
ADMINISTRATIVE_RECORD_OPTIONAL_INTEGER_KEYS: Final[frozenset[str]] = frozenset(
    {"administrative_region_id", "administrative_unit_id"}
)
ADMINISTRATIVE_RECORD_OPTIONAL_STRING_LIST_KEYS: Final[frozenset[str]] = frozenset(
    {"legacy_names", "aliases"}
)
ADMINISTRATIVE_RECORD_STRING_KEYS: Final[frozenset[str]] = (
    ADMINISTRATIVE_RECORD_REQUIRED_KEYS | ADMINISTRATIVE_RECORD_OPTIONAL_STRING_KEYS
)
ADMINISTRATIVE_RECORD_KEYS: Final[frozenset[str]] = (
    ADMINISTRATIVE_RECORD_REQUIRED_KEYS
    | ADMINISTRATIVE_RECORD_OPTIONAL_STRING_KEYS
    | ADMINISTRATIVE_RECORD_OPTIONAL_INTEGER_KEYS
    | ADMINISTRATIVE_RECORD_OPTIONAL_STRING_LIST_KEYS
    | {"is_new_format"}
)

SEARCH_ENGINE_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {"token_corpus", "field_tokens", "metadata", "token_sets"}
)
SEARCH_DOCUMENT_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "level",
        "source",
        "record",
        "province_code",
        "district_code",
        "unit_token",
    }
)
SEARCH_DOCUMENT_KEYS: Final[frozenset[str]] = SEARCH_DOCUMENT_REQUIRED_KEYS | {"code"}


AdministrativeRecordsByCode: TypeAlias = dict[AddressCode, AdministrativeRecord]


class _ParsedAddressComponentOptionalFields(TypedDict, total=False):
    id: AddressCode
    code: AddressCode
    full_name: str
    aliases: list[str]
    legacy_names: list[str]


class ParsedAddressComponent(_ParsedAddressComponentOptionalFields):
    """Administrative component emitted by ``AddressParser.process``."""

    name: str


class _ProjectedAddressComponentOptionalFields(TypedDict, total=False):
    path: str | None
    path_with_type: str | None
    parent_code: AddressCode | None


class ProjectedAddressComponent(_ProjectedAddressComponentOptionalFields):
    """Administrative record projected by an ID lookup."""

    id: AddressCode | None
    code: AddressCode | None
    name: str | None
    full_name: str | None
    slug: str | None
    type: str | None


class _ParseResultBase(TypedDict):
    province: ParsedAddressComponent | None
    ward: ParsedAddressComponent | None
    street_address: str


class OldAddressParseResult(_ParseResultBase):
    district: ParsedAddressComponent | None
    format: Literal["old"]
    is_new: Literal[False]


class NewAddressParseResult(_ParseResultBase):
    district: None
    format: Literal["new"]
    is_new: Literal[True]


class UnknownAddressParseResult(_ParseResultBase):
    district: ParsedAddressComponent | None
    format: Literal["unknown"]
    is_new: None


ParseResult: TypeAlias = OldAddressParseResult | NewAddressParseResult | UnknownAddressParseResult


class AddressComponentsResult(TypedDict):
    province: ProjectedAddressComponent | None
    district: ProjectedAddressComponent | None
    ward: ProjectedAddressComponent | None
    full_address: str | None


class ExternalWardMappingRow(TypedDict):
    old_province_code: AddressCode
    old_province_name: str
    old_district_code: AddressCode
    old_district_name: str
    old_ward_code: AddressCode
    old_ward_name: str | None
    new_province_code: AddressCode
    new_province_name: str
    new_ward_code: AddressCode
    new_ward_name: str


class WardMappingRow(TypedDict):
    city_id_old: AddressCode | None
    district_id_old: AddressCode | None
    ward_id_old: AddressCode
    city_id_new: AddressCode | None
    ward_id_new: AddressCode
    old_ward_name: str | None
    new_ward_name: str | None
    old_province_name: str | None
    new_province_name: str | None
    old_district_name: str | None


WardMappingsByCode: TypeAlias = dict[AddressCode, list[WardMappingRow]]


class NewAddressMappingResult(TypedDict):
    province_id_new: AddressCode | None
    province_name_new: str | None
    ward_id_new: AddressCode
    ward_name_new: str | None
    raw: WardMappingRow


class OldAddressMappingResult(TypedDict):
    province_id_old: AddressCode | None
    province_name_old: str | None
    district_id_old: AddressCode | None
    district_name_old: str | None
    ward_id_old: AddressCode
    ward_name_old: str | None
    raw: WardMappingRow


class _AddressMappingSummaryBase(TypedDict):
    source: AddressComponentsResult
    target: AddressComponentsResult


class OldToNewAddressMappingSummary(_AddressMappingSummaryBase):
    direction: Literal["old_to_new"]
    source_format_is_new: Literal[False]
    mapping: NewAddressMappingResult


class NewToOldAddressMappingSummary(_AddressMappingSummaryBase):
    direction: Literal["new_to_old"]
    source_format_is_new: Literal[True]
    mapping: OldAddressMappingResult


AddressMappingSummary: TypeAlias = OldToNewAddressMappingSummary | NewToOldAddressMappingSummary


class _SearchDocumentOptionalFields(TypedDict, total=False):
    code: AddressCode


class SearchDocument(_SearchDocumentOptionalFields):
    """Validated metadata stored alongside one BM25 document."""

    level: AdministrativeLevel
    source: RegistrySource
    record: AdministrativeRecord
    province_code: AddressCode | None
    district_code: AddressCode | None
    unit_token: str | None


SearchDocumentInput: TypeAlias = SearchDocument


class SearchResult(SearchDocument):
    score: float


class SearchEngineState(TypedDict):
    token_corpus: list[list[str]]
    field_tokens: list[list[list[str]]]
    metadata: list[SearchDocument]
    token_sets: list[list[str]]


class AddressNodeState(TypedDict):
    province_name: str
    district_name: str
    ward_name: str
    province_id: AddressCode | None
    district_id: AddressCode | None
    ward_id: AddressCode | None
    is_new_format: bool | None
    standardized_full_name: str
    ngram_list: list[str]


class ComponentSignature(TypedDict):
    sequences: list[list[str]]
    tokens: set[str]
    abbreviation_sequences: set[tuple[str, ...]]


class StreetToken(TypedDict):
    """Raw and normalized token span used during street extraction."""

    start: int
    end: int
    raw: str
    norm: str


class ContextualWardCandidate(TypedDict):
    """Ranked old-registry ward candidate detected from an address suffix."""

    score: tuple[int, ...]
    segment_idx: int
    suffix_len: int
    ward_info: AdministrativeRecord
    district_info: AdministrativeRecord | None


class ContextualOldWardResult(TypedDict):
    """Typed context recovered from an explicit old-format ward segment."""

    province_info: AdministrativeRecord
    district_info: AdministrativeRecord
    ward_info: AdministrativeRecord
    street_address: str
    has_dedicated_district_segment: bool
    raw_ward_fragment: str


class PromotedContextualWardResult(TypedDict):
    """New-format projection of a contextual old ward match."""

    province_info: AdministrativeRecord
    ward_info: AdministrativeRecord


class ImmediateOldWardResult(TypedDict):
    """Fully resolved old-format result returned before fuzzy candidate ranking."""

    province_info: AdministrativeRecord
    district_info: AdministrativeRecord
    ward_info: AdministrativeRecord
    street_address: str


class OldDatasetSection(TypedDict):
    provinces: AdministrativeRecordsByCode
    districts: AdministrativeRecordsByCode
    wards: AdministrativeRecordsByCode


class NewDatasetSection(TypedDict):
    provinces: AdministrativeRecordsByCode
    wards: AdministrativeRecordsByCode


class MappingDatasetSection(TypedDict):
    ward_old_to_new: WardMappingsByCode
    ward_new_to_old: WardMappingsByCode


class RawAddressDataset(TypedDict):
    old: OldDatasetSection
    new: NewDatasetSection
    mapping: MappingDatasetSection


class ExternalNewDataset(TypedDict):
    provinces: AdministrativeRecordsByCode
    wards: AdministrativeRecordsByCode
    ward_mappings: list[ExternalWardMappingRow]


class _LegacyRecordOptionalFields(TypedDict, total=False):
    parent_code: AddressCode | None
    administrative_unit_id: int | None
    is_new_format: bool
    legacy_names: list[str]


class LegacyWardRecord(_LegacyRecordOptionalFields):
    id: AddressCode | None
    code: AddressCode | None
    full_name: str | None


class LegacyDistrictRecord(_LegacyRecordOptionalFields):
    id: AddressCode | None
    code: AddressCode | None
    full_name: str
    wards: dict[str, LegacyWardRecord]


class LegacyProvinceRecord(_LegacyRecordOptionalFields):
    id: AddressCode | None
    code: AddressCode | None
    full_name: str
    districts: dict[str, LegacyDistrictRecord]


LegacyAddressDataset: TypeAlias = dict[str, LegacyProvinceRecord]


class PreprocessedState(TypedDict):
    address_node_list: list[AddressNodeState]
    invert_ngrams_idx: dict[str, set[int]]
    invert_province_to_indices: defaultdict[str, set[int]]
    invert_district_to_indices: defaultdict[str, set[int]]
    invert_ward_to_indices: defaultdict[str, set[int]]
    province_names_std: set[str]
    district_names_std: set[str]
    ward_names_std: set[str]
    province_lookup: dict[str, AdministrativeRecord]
    district_lookup: dict[tuple[str, str], AdministrativeRecord]
    district_lookup_by_name: defaultdict[str, list[AdministrativeRecord]]
    ward_lookup: dict[tuple[str, str, str], AdministrativeRecord]
    ward_lookup_by_name: defaultdict[str, list[AdministrativeRecord]]
    ward_lookup_by_province_name: defaultdict[tuple[str, str], list[AdministrativeRecord]]
    ward_lookup_by_district_key: defaultdict[str, list[AdministrativeRecord]]
    ward_mapping_by_old_code: WardMappingsByCode
    ward_mapping_by_new_code: WardMappingsByCode
    old_province_records: AdministrativeRecordsByCode
    old_district_records: AdministrativeRecordsByCode
    old_ward_records: AdministrativeRecordsByCode
    new_province_records: AdministrativeRecordsByCode
    new_ward_records: AdministrativeRecordsByCode
    external_new_province_records: AdministrativeRecordsByCode
    external_new_ward_records: AdministrativeRecordsByCode
    search_engine: SearchEngineState | None


class PreprocessedCachePayload(TypedDict):
    version: int
    signature: DatasetSignature
    state: PreprocessedState


class ComponentSnapshot(TypedDict):
    id: AddressCode
    code: AddressCode
    full_name: str


class RegressionExpectedResult(TypedDict):
    format: RegistryFormat
    is_new: bool
    street_address: str
    province: ComponentSnapshot | None
    district: ComponentSnapshot | None
    ward: ComponentSnapshot | None


class RegressionCase(TypedDict):
    case_id: str
    address: str
    expected: RegressionExpectedResult


class RegressionMetadata(TypedDict):
    old_case_count: int
    old_compact_numeric_ward_case_count: int
    new_case_count: int
    total_case_count: int


class RegressionCorpus(TypedDict):
    metadata: RegressionMetadata
    old_cases: list[RegressionCase]
    new_cases: list[RegressionCase]


class _ReviewedRegressionCaseOptionalFields(TypedDict, total=False):
    submission_id: str
    submission_id_retest: str


class ReviewedRegressionCase(_ReviewedRegressionCaseOptionalFields):
    sub_id: str
    doc_id: str
    address: str
    format: RegistryFormat
    is_new: bool
    street_address: str
    province: str | None
    province_id: AddressCode | None
    district: str | None
    district_id: AddressCode | None
    ward: str | None
    ward_id: AddressCode | None


__all__ = [
    "AddressCode",
    "AddressCodeInput",
    "AddressComponentsResult",
    "AddressFormat",
    "AddressMappingSummary",
    "AddressNodeState",
    "ADMINISTRATIVE_RECORD_KEYS",
    "ADMINISTRATIVE_RECORD_OPTIONAL_INTEGER_KEYS",
    "ADMINISTRATIVE_RECORD_OPTIONAL_STRING_KEYS",
    "ADMINISTRATIVE_RECORD_OPTIONAL_STRING_LIST_KEYS",
    "ADMINISTRATIVE_RECORD_REQUIRED_KEYS",
    "ADMINISTRATIVE_RECORD_STRING_KEYS",
    "AdministrativeLevel",
    "AdministrativeRecord",
    "AdministrativeRecordsByCode",
    "CandidateHit",
    "ComponentSignature",
    "ComponentSnapshot",
    "ContextualOldWardResult",
    "ContextualWardCandidate",
    "DatasetFileSignature",
    "DatasetSignature",
    "DetectedComponents",
    "ExternalNewDataset",
    "ExternalWardMappingRow",
    "FuzzyChoiceProfile",
    "ImmediateOldWardResult",
    "LegacyAddressDataset",
    "LegacyDistrictRecord",
    "LegacyProvinceRecord",
    "LegacyWardRecord",
    "MappingDirection",
    "NewAddressMappingResult",
    "NewAddressParseResult",
    "NewDatasetSection",
    "NewToOldAddressMappingSummary",
    "NgramHit",
    "NormalizationMode",
    "OldAddressMappingResult",
    "OldAddressParseResult",
    "OldDatasetSection",
    "OldToNewAddressMappingSummary",
    "ParseResult",
    "ParsedAddressComponent",
    "PreprocessedCachePayload",
    "PreprocessedState",
    "PromotedContextualWardResult",
    "ProjectedAddressComponent",
    "RawAddressDataset",
    "RegistryFormat",
    "RegistrySource",
    "RegressionCase",
    "RegressionCorpus",
    "RegressionExpectedResult",
    "RegressionMetadata",
    "ReviewedRegressionCase",
    "SEARCH_DOCUMENT_KEYS",
    "SEARCH_DOCUMENT_REQUIRED_KEYS",
    "SEARCH_ENGINE_STATE_KEYS",
    "SearchDocument",
    "SearchDocumentInput",
    "SearchEngineState",
    "SearchResult",
    "StreetToken",
    "UnknownAddressParseResult",
    "WardMappingRow",
    "WardMappingsByCode",
]
