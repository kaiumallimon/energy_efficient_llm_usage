"""CLI for running benchmark prompts against the full pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark import DEFAULT_BENCHMARK_PATH, run_benchmark, summarize_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run labeled benchmark prompts through the pipeline.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--id",
        action="append",
        dest="case_ids",
        help="Run only specific benchmark case id(s); can be repeated",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show passing cases too",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results = run_benchmark(args.file, case_ids=args.case_ids)
    summary = summarize_results(results)

    if args.json:
        payload = {
            "summary": summary,
            "results": [
                {
                    "id": result.case.id,
                    "description": result.case.description,
                    "passed": result.passed,
                    "failures": result.failures,
                    "analysis": result.result.analysis.to_dict(),
                    "optimization": result.result.optimization.to_dict(),
                    "generation": result.result.generation.to_dict(),
                }
                for result in results
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0 if summary["failed"] == 0 else 1

    print(f"Benchmark: {args.file}")
    print(f"Passed: {summary['passed']}/{summary['total']} ({summary['pass_rate']}%)")
    print()

    for result in results:
        if result.passed and not args.verbose:
            continue

        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case.id}")
        print(f"  {result.case.description}")
        if result.failures:
            for failure in result.failures:
                print(f"  - {failure}")
        elif args.verbose:
            analysis = result.result.analysis
            optimization = result.result.optimization
            generation = result.result.generation
            print(
                f"  level={analysis.level.value} "
                f"task={analysis.task_type.value} "
                f"policy={analysis.policy.value} "
                f"tier={generation.model_tier} "
                f"modified={optimization.was_modified}"
            )
        print()

    if summary["failed"]:
        print("Some benchmark cases failed.")
        return 1

    print("All benchmark cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
