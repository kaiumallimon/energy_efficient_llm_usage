from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.pipeline import PipelineResult, PromptPipeline

DEFAULT_BENCHMARK_PATH = Path("data/benchmark_prompts.json")


@dataclass
class BenchmarkCase:
    id: str
    description: str
    query: str
    context: str | None = None
    context_file: str | None = None
    expected: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], base_dir: Path | None = None) -> BenchmarkCase:
        context = payload.get("context")
        context_file = payload.get("context_file")
        if context_file:
            path = Path(context_file)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            context = path.read_text(encoding="utf-8")

        return cls(
            id=payload["id"],
            description=payload.get("description", ""),
            query=payload["query"],
            context=context,
            context_file=context_file,
            expected=payload.get("expected", {}),
        )


@dataclass
class BenchmarkCaseResult:
    case: BenchmarkCase
    result: PipelineResult
    passed: bool
    failures: list[str] = field(default_factory=list)


def load_benchmark_cases(path: Path | str = DEFAULT_BENCHMARK_PATH) -> list[BenchmarkCase]:
    benchmark_path = Path(path)
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    base_dir = benchmark_path.parent.parent
    return [BenchmarkCase.from_dict(case, base_dir=base_dir) for case in payload["cases"]]


def evaluate_case(case: BenchmarkCase, result: PipelineResult) -> list[str]:
    failures: list[str] = []
    expected = case.expected
    analysis = result.analysis
    optimization = result.optimization
    generation = result.generation

    exact_fields = {
        "level": analysis.level.value,
        "task_type": analysis.task_type.value,
        "policy": analysis.policy.value,
        "model_tier": generation.model_tier,
        "template_id": generation.template_id,
    }

    for field_name, actual in exact_fields.items():
        if field_name in expected and actual != expected[field_name]:
            failures.append(
                f"{field_name}: expected {expected[field_name]!r}, got {actual!r}"
            )

    if "should_modify" in expected and optimization.was_modified != expected["should_modify"]:
        failures.append(
            f"should_modify: expected {expected['should_modify']}, "
            f"got {optimization.was_modified}"
        )

    if "min_word_reduction" in expected:
        actual = optimization.word_reduction_percent
        if actual < expected["min_word_reduction"]:
            failures.append(
                f"min_word_reduction: expected >= {expected['min_word_reduction']}, got {actual}"
            )

    optimized_query = optimization.optimized_query.lower()
    for text in expected.get("optimized_query_contains", []):
        if text.lower() not in optimized_query:
            failures.append(f"optimized_query missing expected text: {text!r}")

    for text in expected.get("optimized_query_excludes", []):
        if text.lower() in optimized_query:
            failures.append(f"optimized_query still contains: {text!r}")

    optimized_context = (optimization.optimized_context or "").lower()
    for text in expected.get("optimized_context_excludes", []):
        if text.lower() in optimized_context:
            failures.append(f"optimized_context still contains: {text!r}")

    system_prompt = generation.system_prompt.lower()
    for text in expected.get("system_prompt_contains", []):
        if text.lower() not in system_prompt:
            failures.append(f"system_prompt missing expected text: {text!r}")

    user_prompt = generation.user_prompt.lower()
    for text in expected.get("user_prompt_contains", []):
        if text.lower() not in user_prompt:
            failures.append(f"user_prompt missing expected text: {text!r}")

    return failures


def run_benchmark(
    path: Path | str = DEFAULT_BENCHMARK_PATH,
    case_ids: list[str] | None = None,
    pipeline: PromptPipeline | None = None,
) -> list[BenchmarkCaseResult]:
    cases = load_benchmark_cases(path)
    if case_ids:
        selected = {case_id for case_id in case_ids}
        cases = [case for case in cases if case.id in selected]
        missing = selected - {case.id for case in cases}
        if missing:
            raise ValueError(f"Unknown benchmark case id(s): {', '.join(sorted(missing))}")

    runner = pipeline or PromptPipeline()
    results: list[BenchmarkCaseResult] = []

    for case in cases:
        pipeline_result = runner.process(case.query, case.context)
        failures = evaluate_case(case, pipeline_result)
        results.append(
            BenchmarkCaseResult(
                case=case,
                result=pipeline_result,
                passed=not failures,
                failures=failures,
            )
        )

    return results


def summarize_results(results: list[BenchmarkCaseResult]) -> dict[str, Any]:
    passed = sum(1 for result in results if result.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round((passed / len(results)) * 100, 1) if results else 0.0,
    }
