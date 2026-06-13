"""CLI for analyzing prompt complexity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analyzer import ComplexityAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify user prompt complexity for energy-aware optimization.",
    )
    parser.add_argument("query", nargs="?", help="User prompt to analyze")
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
        help="Print full result as JSON",
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
    result = ComplexityAnalyzer().analyze(query, context)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"Level:      {result.level.value}")
    print(f"Score:      {result.score}")
    print(f"Task type:  {result.task_type.value}")
    print(f"Policy:     {result.policy.value}")
    print(f"Confidence: {result.confidence}")
    print("\nRationale:")
    for line in result.rationale:
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
