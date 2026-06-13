"""CLI for running analyzer + optimizer + generator pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import PromptPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze, optimize, and assemble a prompt for LLM usage.",
    )
    parser.add_argument("query", nargs="?", help="User prompt to process")
    parser.add_argument(
        "--context",
        "-c",
        help="Optional external context (file path or inline text)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Read query from a text file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full pipeline result as JSON",
    )
    return parser


def load_context(value: str | None) -> str | None:
    if not value:
        return None

    path = Path(value)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")

    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    query = args.query
    if args.file:
        query = args.file.read_text(encoding="utf-8")

    if not query:
        parser.error("Provide a query argument or --file")

    context = load_context(args.context)
    result = PromptPipeline().process(query, context)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    analysis = result.analysis
    optimization = result.optimization
    generation = result.generation

    print("Analysis")
    print(f"  Level:      {analysis.level.value}")
    print(f"  Score:      {analysis.score}")
    print(f"  Task type:  {analysis.task_type.value}")
    print(f"  Policy:     {analysis.policy.value}")
    print(f"  Confidence: {analysis.confidence}")

    print("\nOptimization")
    print(f"  Modified:   {optimization.was_modified}")
    print(
        "  Words:      "
        f"{optimization.original_word_count} -> "
        f"{optimization.optimized_word_count} "
        f"({optimization.word_reduction_percent:.1f}% reduction)"
    )
    print(f"  Query:      {optimization.optimized_query}")
    if optimization.optimized_context is not None:
        print(f"  Context:    {optimization.optimized_context}")

    print("\nChanges:")
    for change in optimization.changes:
        print(f"  - {change}")

    print("\nGeneration")
    print(f"  Template:   {generation.template_id}")
    print(f"  Model tier: {generation.model_tier}")
    print(f"  System:     {generation.system_prompt}")
    print(f"  User:       {generation.user_prompt}")

    if generation.notes:
        print("\nGenerator notes:")
        for note in generation.notes:
            print(f"  - {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
