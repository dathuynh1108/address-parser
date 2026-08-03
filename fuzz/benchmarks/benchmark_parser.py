"""Deterministic constructor and steady-state parser benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from typing import TypedDict

from address_parser import (
    AddressParser,
    native_acceleration_available,
    require_native_acceleration,
)
from address_parser.contracts import RegressionCase
from full_dataset_regression_cases import build_regression_cases


class _Arguments(argparse.Namespace):
    constructor_rounds: int
    new_cases: int
    old_cases: int
    require_native: bool
    rounds: int
    warmup_rounds: int


class BenchmarkResult(TypedDict):
    backend: str
    case_count: int
    constructor_median_ms: float
    constructor_round_ms: list[float]
    new_case_count: int
    observations: int
    old_case_count: int
    parse_mean_ms: float
    parse_p50_ms: float
    parse_p95_ms: float
    platform: str
    python: str
    round_ms_per_parse: list[float]
    throughput_per_second: float


def _parse_arguments() -> _Arguments:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--constructor-rounds", type=int, default=3)
    argument_parser.add_argument("--old-cases", type=int, default=150)
    argument_parser.add_argument("--new-cases", type=int, default=150)
    argument_parser.add_argument("--rounds", type=int, default=5)
    argument_parser.add_argument("--warmup-rounds", type=int, default=1)
    argument_parser.add_argument("--require-native", action="store_true")
    arguments = argument_parser.parse_args(namespace=_Arguments())
    for option, value in (
        ("--constructor-rounds", arguments.constructor_rounds),
        ("--old-cases", arguments.old_cases),
        ("--new-cases", arguments.new_cases),
        ("--rounds", arguments.rounds),
        ("--warmup-rounds", arguments.warmup_rounds),
    ):
        if value < 1:
            argument_parser.error(f"{option} must be positive")
    return arguments


def _even_sample(cases: list[RegressionCase], count: int) -> list[RegressionCase]:
    if count >= len(cases):
        return list(cases)
    if count == 1:
        return [cases[0]]
    last_index = len(cases) - 1
    return [cases[round(index * last_index / (count - 1))] for index in range(count)]


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def main() -> None:
    arguments = _parse_arguments()
    if arguments.require_native:
        require_native_acceleration()

    constructor_times: list[float] = []
    parser: AddressParser | None = None
    for _ in range(arguments.constructor_rounds):
        started = time.perf_counter()
        parser = AddressParser()
        constructor_times.append((time.perf_counter() - started) * 1_000)
    if parser is None:
        raise RuntimeError("parser benchmark did not construct an AddressParser")

    corpus = build_regression_cases(parser)
    old_cases = _even_sample(corpus["old_cases"], arguments.old_cases)
    new_cases = _even_sample(corpus["new_cases"], arguments.new_cases)
    cases = old_cases + new_cases

    for _ in range(arguments.warmup_rounds):
        for case in cases:
            parser.process(case["address"])

    observations: list[float] = []
    round_times: list[float] = []
    for _ in range(arguments.rounds):
        round_started = time.perf_counter()
        for case in cases:
            parse_started = time.perf_counter()
            parser.process(case["address"])
            observations.append((time.perf_counter() - parse_started) * 1_000)
        round_times.append((time.perf_counter() - round_started) * 1_000 / len(cases))

    mean_ms = statistics.fmean(observations)
    result: BenchmarkResult = {
        "backend": "native" if native_acceleration_available() else "python",
        "case_count": len(cases),
        "constructor_median_ms": statistics.median(constructor_times),
        "constructor_round_ms": constructor_times,
        "new_case_count": len(new_cases),
        "observations": len(observations),
        "old_case_count": len(old_cases),
        "parse_mean_ms": mean_ms,
        "parse_p50_ms": _percentile(observations, 0.50),
        "parse_p95_ms": _percentile(observations, 0.95),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "round_ms_per_parse": round_times,
        "throughput_per_second": 1_000 / mean_ms,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
