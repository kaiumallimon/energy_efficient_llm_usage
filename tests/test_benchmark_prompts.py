from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark import DEFAULT_BENCHMARK_PATH, load_benchmark_cases, run_benchmark


@pytest.fixture
def benchmark_path() -> Path:
    return Path(DEFAULT_BENCHMARK_PATH)


def test_benchmark_file_loads_cases(benchmark_path: Path) -> None:
    cases = load_benchmark_cases(benchmark_path)

    assert len(cases) >= 20
    assert all(case.id for case in cases)
    assert all(case.query for case in cases)


def test_benchmark_cases_match_expectations(benchmark_path: Path) -> None:
    results = run_benchmark(benchmark_path)
    failures = [
        (result.case.id, result.failures)
        for result in results
        if not result.passed
    ]

    if failures:
        lines = [f"{case_id}: {'; '.join(case_failures)}" for case_id, case_failures in failures]
        pytest.fail("Benchmark mismatches:\n" + "\n".join(lines))
