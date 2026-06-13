from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark import QUERY_TYPE_BENCHMARK_PATH, load_benchmark_cases, run_benchmark


@pytest.fixture
def query_type_benchmark_path() -> Path:
    return Path(QUERY_TYPE_BENCHMARK_PATH)


def test_query_type_benchmark_loads_cases(query_type_benchmark_path: Path) -> None:
    cases = load_benchmark_cases(query_type_benchmark_path)
    assert len(cases) == 5
    assert {case.id for case in cases} == {
        "definition_cfg",
        "summarize_lecture_notes",
        "reasoning_cyclomatic",
        "code_cfg_builder",
        "rag_notes_qa",
    }


def test_query_type_benchmark_cases_match_expectations(query_type_benchmark_path: Path) -> None:
    results = run_benchmark(query_type_benchmark_path)
    failures = [
        (result.case.id, result.failures)
        for result in results
        if not result.passed
    ]

    if failures:
        lines = [f"{case_id}: {'; '.join(case_failures)}" for case_id, case_failures in failures]
        pytest.fail("Query-type benchmark mismatches:\n" + "\n".join(lines))
