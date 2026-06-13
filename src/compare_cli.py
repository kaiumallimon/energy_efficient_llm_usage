"""CLI for comparing baseline vs optimized output quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.comparison import OutputComparator
from src.evaluation.quality_scorer import QualityScorer
from src.llm import OllamaClient, OllamaConfig, OllamaError
from src.pipeline import PromptPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and optimized LLM outputs using semantic similarity "
            "and an LLM-as-a-judge rubric."
        ),
    )
    parser.add_argument("query", nargs="?", help="User query used to generate both outputs")
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
        "--baseline",
        help="Baseline completion text (skip live baseline call)",
    )
    parser.add_argument(
        "--optimized",
        help="Optimized completion text (skip live optimized call)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full comparison as JSON",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-a-judge rubric scoring",
    )
    parser.add_argument(
        "--model",
        help="Override Ollama model name",
    )
    parser.add_argument(
        "--ollama-url",
        help="Override Ollama base URL",
    )
    parser.add_argument(
        "--think",
        choices=["true", "false", "auto"],
        help="Control Qwen thinking mode",
    )
    return parser


def load_context(value: str | None) -> str | None:
    if not value:
        return None

    path = Path(value)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")

    return value


def resolve_think_override(args: argparse.Namespace) -> bool | None:
    if args.think == "true":
        return True
    if args.think == "false":
        return False
    return None


def build_client(args: argparse.Namespace) -> OllamaClient:
    config = OllamaConfig.load()
    if args.ollama_url:
        config = OllamaConfig(
            base_url=args.ollama_url.rstrip("/"),
            model=args.model or config.model,
            timeout_seconds=config.timeout_seconds,
            think=config.think,
        )
    elif args.model:
        config = OllamaConfig(
            base_url=config.base_url,
            model=args.model,
            timeout_seconds=config.timeout_seconds,
            think=config.think,
        )
    return OllamaClient(config)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    query = args.query
    if args.file:
        query = args.file.read_text(encoding="utf-8")

    if not query:
        parser.error("Provide a query argument or --file")

    context = load_context(args.context)
    think_override = resolve_think_override(args)
    client = build_client(args)
    scorer = QualityScorer(
        embedding_client=client,
        judge_client=client if not args.no_judge else None,
        use_llm_judge=not args.no_judge,
    )
    comparator = OutputComparator(scorer=scorer)

    try:
        if args.baseline and args.optimized:
            comparison = comparator.compare(
                query,
                args.baseline,
                args.optimized,
            )
        else:
            pipeline = PromptPipeline(llm_client=client, use_ollama=True)
            result = pipeline.process(
                query,
                context,
                call_llm=True,
                evaluate=True,
                think=think_override,
                fallback=True,
            )
            if result.llm is None or result.baseline_llm is None:
                parser.error("Live comparison requires successful baseline and optimized LLM calls.")

            comparison = comparator.compare(
                query,
                result.baseline_llm.response,
                result.llm.response,
                baseline_llm=result.baseline_llm,
                optimized_llm=result.llm,
            )
    except OllamaError as exc:
        print(f"Ollama error: {exc}")
        return 1

    if args.json:
        print(json.dumps(comparison.to_dict(), indent=2))
        return 0

    quality = comparison.quality
    semantic = quality["semantic_similarity"]
    print("Output Comparison")
    print(f"  Query: {comparison.query}")
    print(f"  Semantic ({semantic['method']}): {semantic['score']:.3f}")
    print(f"  Overall quality score: {quality['overall_score']:.3f}")
    print(f"  Acceptable quality: {'yes' if comparison.acceptable_quality else 'no'}")

    rubric = quality.get("rubric")
    if rubric:
        print("\nLLM Judge Rubric (1-5)")
        print(f"  Accuracy:      {rubric['accuracy']:.1f}")
        print(f"  Completeness:  {rubric['completeness']:.1f}")
        print(f"  Brevity:       {rubric['brevity']:.1f}")
        print(f"  Usefulness:    {rubric['usefulness']:.1f}")
        print(f"  Overall:       {rubric['overall']:.1f}")
        if rubric.get("rationale"):
            print(f"  Rationale:     {rubric['rationale']}")

    if comparison.efficiency:
        efficiency = comparison.efficiency
        print("\nEfficiency")
        if efficiency.get("total_token_reduction_percent") is not None:
            print(
                "  Total tokens saved: "
                f"{efficiency['total_token_reduction_percent']:.1f}%"
            )
        if efficiency.get("energy_proxy_savings_percent") is not None:
            print(
                "  Energy proxy saved: "
                f"{efficiency['energy_proxy_savings_percent']:.1f}%"
            )
        if efficiency.get("energy_measured_savings_percent") is not None:
            print(
                "  Measured energy saved: "
                f"{efficiency['energy_measured_savings_percent']:.1f}%"
            )

    print("\nBaseline output:")
    print(comparison.baseline_completion)
    print("\nOptimized output:")
    print(comparison.optimized_completion)

    for note in comparison.notes:
        print(f"\nNote: {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
