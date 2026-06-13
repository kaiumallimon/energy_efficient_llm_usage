from __future__ import annotations

from src.analyzer.models import ComplexityResult
from src.generator.models import GeneratedPrompt
from src.llm.models import LLMCallResult
from src.monitoring.models import MonitoringSnapshot, PathMetrics
from src.optimizer.models import OptimizationResult
from src.optimizer.rules import count_words


class PerformanceMonitor:
    """Collects per-request usage metrics for monitoring and evaluation."""

    @staticmethod
    def collect(
        query: str,
        context: str | None,
        analysis: ComplexityResult,
        optimization: OptimizationResult,
        generation: GeneratedPrompt,
        *,
        llm_result: LLMCallResult | None = None,
        baseline_llm: LLMCallResult | None = None,
    ) -> MonitoringSnapshot:
        optimized = PerformanceMonitor._path_metrics(
            path="optimized",
            prompt_words=optimization.optimized_word_count,
            model_tier=generation.model_tier,
            llm_result=llm_result,
        )

        baseline = None
        if baseline_llm is not None:
            baseline_words = count_words(query)
            if context:
                baseline_words += count_words(context)
            baseline = PerformanceMonitor._path_metrics(
                path="baseline",
                prompt_words=baseline_words,
                model_tier=None,
                llm_result=baseline_llm,
            )

        return MonitoringSnapshot(optimized=optimized, baseline=baseline)

    @staticmethod
    def _path_metrics(
        path: str,
        prompt_words: int,
        model_tier: str | None,
        llm_result: LLMCallResult | None,
    ) -> PathMetrics:
        if llm_result is None:
            return PathMetrics(path=path, prompt_words=prompt_words, model_tier=model_tier)

        usage = llm_result.usage
        return PathMetrics(
            path=path,
            prompt_words=prompt_words,
            model_tier=model_tier,
            model=llm_result.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=llm_result.latency_ms,
            energy_proxy=llm_result.energy_proxy,
            energy_measured_j=llm_result.energy_measured_j,
            completion=llm_result.response,
        )
