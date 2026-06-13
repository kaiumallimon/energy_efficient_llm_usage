"""CLI for running analyzer + optimizer + generator + optional Ollama LLM call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.llm import OllamaClient, OllamaConfig, OllamaError
from src.pipeline import PromptPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze, optimize, assemble, and optionally call a local Ollama model.",
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
    parser.add_argument(
        "--call",
        action="store_true",
        help="Call the configured Ollama model and include token usage",
    )
    parser.add_argument(
        "--model",
        help="Override Ollama model name (default: config/ollama.json or OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--ollama-url",
        help="Override Ollama base URL (default: config/ollama.json or OLLAMA_BASE_URL)",
    )
    parser.add_argument(
        "--think",
        choices=["true", "false", "auto"],
        help="Control Qwen thinking mode: true, false, or auto by complexity",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compare optimized output against baseline; with --call, runs both LLM paths",
    )
    return parser


def load_context(value: str | None) -> str | None:
    if not value:
        return None

    path = Path(value)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")

    return value


def build_pipeline(args: argparse.Namespace) -> PromptPipeline:
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

    return PromptPipeline(llm_client=OllamaClient(config))


def resolve_think_override(args: argparse.Namespace) -> bool | None:
    if args.think == "true":
        return True
    if args.think == "false":
        return False
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    query = args.query
    if args.file:
        query = args.file.read_text(encoding="utf-8")

    if not query:
        parser.error("Provide a query argument or --file")

    context = load_context(args.context)
    pipeline = build_pipeline(args)
    think_override = resolve_think_override(args)

    try:
        result = pipeline.process(
            query,
            context,
            call_llm=args.call,
            evaluate=args.evaluate,
            think=think_override,
        )
    except OllamaError as exc:
        print(f"Ollama error: {exc}")
        return 1

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

    if result.llm is not None:
        llm = result.llm
        usage = llm.usage
        print("\nLLM Call (optimized)")
        print(f"  Provider:   {llm.provider}")
        print(f"  Model:      {llm.model}")
        print(f"  Latency:    {llm.latency_ms:.0f} ms")
        print(
            "  Tokens:     "
            f"prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
        )
        print(f"  Thinking:   {usage.thinking_enabled}")
        print(f"  Energy est: {llm.energy_proxy:.4f} (proxy units)")
        print(f"  Completion: {llm.response}")

    if result.baseline_llm is not None:
        baseline = result.baseline_llm
        usage = baseline.usage
        print("\nLLM Call (baseline)")
        print(f"  Model:      {baseline.model}")
        print(f"  Latency:    {baseline.latency_ms:.0f} ms")
        print(
            "  Tokens:     "
            f"prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
        )
        print(f"  Completion: {baseline.response}")

    if result.monitoring is not None:
        monitoring = result.monitoring
        print("\nMonitoring")
        optimized = monitoring.optimized
        print(f"  Optimized prompt words: {optimized.prompt_words}")
        if optimized.total_tokens is not None:
            print(f"  Optimized total tokens: {optimized.total_tokens}")
        if monitoring.baseline is not None and monitoring.baseline.total_tokens is not None:
            print(f"  Baseline total tokens:  {monitoring.baseline.total_tokens}")

    if result.evaluation is not None:
        evaluation = result.evaluation
        efficiency = evaluation.efficiency
        print("\nEvaluation")
        print(f"  Word reduction:   {efficiency.word_reduction_percent:.1f}%")
        if efficiency.prompt_token_reduction_percent is not None:
            print(
                "  Prompt tokens:    "
                f"{efficiency.prompt_token_reduction_percent:.1f}% saved"
            )
        if efficiency.total_token_reduction_percent is not None:
            print(
                "  Total tokens:     "
                f"{efficiency.total_token_reduction_percent:.1f}% saved"
            )
        if efficiency.energy_savings_percent is not None:
            print(f"  Energy proxy:     {efficiency.energy_savings_percent:.1f}% saved")
        if evaluation.completions_match is not None:
            print(f"  Completions match: {evaluation.completions_match}")
        print(f"  Quality gate:     {'PASS' if evaluation.passed_quality_gate else 'REVIEW'}")
        if evaluation.optimized_completion is not None:
            print(f"  Optimized output: {evaluation.optimized_completion}")
        if evaluation.baseline_completion is not None:
            print(f"  Baseline output:  {evaluation.baseline_completion}")
        for note in evaluation.notes:
            print(f"  - {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
