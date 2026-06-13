from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm import LLMCallResult, OllamaClient
from src.monitoring import PathMetrics
from src.optimizer.rules import count_words


@dataclass(frozen=True)
class BaselineResult:
    """Result of sending an unoptimized query directly to the LLM."""

    query: str
    context: str | None
    llm: LLMCallResult
    metrics: PathMetrics

    @property
    def completion(self) -> str:
        return self.llm.response

    @property
    def prompt_words(self) -> int:
        return self.metrics.prompt_words

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": "baseline",
            "query": self.query,
            "context": self.context,
            "prompt_words": self.prompt_words,
            "llm": self.llm.to_dict(),
            "completion": self.llm.response,
            "monitoring": {"baseline": self.metrics.to_dict()},
        }


class BaselineRunner:
    """Sends raw user input to the LLM without optimization."""

    def __init__(self, llm_client: OllamaClient | None = None) -> None:
        self.llm_client = llm_client or OllamaClient()

    def run(
        self,
        query: str,
        context: str | None = None,
        *,
        think: bool | None = None,
    ) -> BaselineResult:
        llm_result = self.llm_client.call_baseline(query, context, think=think)

        prompt_words = count_words(query)
        if context:
            prompt_words += count_words(context)

        baseline_metrics = PathMetrics(
            path="baseline",
            prompt_words=prompt_words,
            model=llm_result.model,
            prompt_tokens=llm_result.usage.prompt_tokens,
            completion_tokens=llm_result.usage.completion_tokens,
            total_tokens=llm_result.usage.total_tokens,
            latency_ms=llm_result.latency_ms,
            energy_proxy=llm_result.energy_proxy,
            completion=llm_result.response,
        )

        return BaselineResult(
            query=query,
            context=context,
            llm=llm_result,
            metrics=baseline_metrics,
        )
