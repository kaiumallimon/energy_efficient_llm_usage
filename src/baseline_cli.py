"""CLI for sending an unoptimized query directly to Ollama (baseline measurement)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline import BaselineRunner
from src.llm import OllamaClient, OllamaConfig, OllamaError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send a raw, unoptimized prompt directly to Ollama for baseline "
            "token and performance measurement."
        ),
    )
    parser.add_argument("query", nargs="?", help="User prompt to send as-is")
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
    parser.add_argument(
        "--call",
        action="store_true",
        help="Accepted for parity with pipeline_cli (baseline always calls Ollama)",
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
        help="Control Qwen thinking mode: true, false, or config default",
    )
    return parser


def load_context(value: str | None) -> str | None:
    if not value:
        return None

    path = Path(value)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")

    return value


def build_runner(args: argparse.Namespace) -> BaselineRunner:
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

    return BaselineRunner(OllamaClient(config))


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
    runner = build_runner(args)
    think_override = resolve_think_override(args)

    try:
        result = runner.run(query, context, think=think_override)
    except OllamaError as exc:
        print(f"Ollama error: {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    llm = result.llm
    usage = llm.usage
    metrics = result.metrics

    print("Baseline LLM Call")
    print(f"  Path:       baseline (unoptimized)")
    print(f"  Query:      {result.query}")
    if result.context is not None:
        print(f"  Context:    {result.context}")
    print(f"  Prompt words: {result.prompt_words}")
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

    print("\nMonitoring")
    print(f"  Path:         {metrics.path}")
    print(f"  Prompt words: {metrics.prompt_words}")
    print(f"  Total tokens: {metrics.total_tokens}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
