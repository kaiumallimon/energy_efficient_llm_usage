from __future__ import annotations

from src.evaluation.models import EfficiencyComparison, EvaluationResult
from src.llm.models import LLMCallResult
from src.monitoring.models import MonitoringSnapshot
from src.optimizer.models import OptimizationResult


def _percent_reduction(baseline: float, optimized: float) -> float | None:
    if baseline <= 0:
        return None
    return ((baseline - optimized) / baseline) * 100.0


def _normalize_completion(text: str) -> str:
    return " ".join(text.strip().lower().split())


class PipelineEvaluator:
    """Compares optimized pipeline results against an unoptimized baseline."""

    @staticmethod
    def evaluate(
        optimization: OptimizationResult,
        monitoring: MonitoringSnapshot,
        *,
        llm_result: LLMCallResult | None = None,
        baseline_llm: LLMCallResult | None = None,
    ) -> EvaluationResult:
        notes: list[str] = []
        word_reduction = optimization.word_reduction_percent

        prompt_token_reduction = None
        total_token_reduction = None
        energy_savings = None
        latency_delta = None
        baseline_completion = None
        optimized_completion = None
        completions_match = None
        passed_quality_gate = True

        optimized_metrics = monitoring.optimized
        baseline_metrics = monitoring.baseline

        if llm_result is not None:
            optimized_completion = llm_result.response
            notes.append("Optimized path includes a live LLM completion.")

        if baseline_llm is not None:
            baseline_completion = baseline_llm.response
            notes.append("Baseline path includes a live LLM completion for comparison.")

        if baseline_metrics is not None and optimized_metrics.prompt_tokens is not None:
            if baseline_metrics.prompt_tokens is not None:
                prompt_token_reduction = _percent_reduction(
                    float(baseline_metrics.prompt_tokens),
                    float(optimized_metrics.prompt_tokens),
                )
            if (
                baseline_metrics.total_tokens is not None
                and optimized_metrics.total_tokens is not None
            ):
                total_token_reduction = _percent_reduction(
                    float(baseline_metrics.total_tokens),
                    float(optimized_metrics.total_tokens),
                )
            if (
                baseline_metrics.energy_proxy is not None
                and optimized_metrics.energy_proxy is not None
            ):
                energy_savings = _percent_reduction(
                    baseline_metrics.energy_proxy,
                    optimized_metrics.energy_proxy,
                )
            if (
                baseline_metrics.latency_ms is not None
                and optimized_metrics.latency_ms is not None
            ):
                latency_delta = optimized_metrics.latency_ms - baseline_metrics.latency_ms

        if word_reduction > 0:
            notes.append(f"Prompt compression saved {word_reduction:.1f}% of input words.")
        elif not optimization.was_modified:
            notes.append("Prompt was already concise; no word reduction applied.")

        if prompt_token_reduction is not None:
            if prompt_token_reduction > 0:
                notes.append(
                    f"Optimized prompt used {prompt_token_reduction:.1f}% fewer prompt tokens."
                )
            elif prompt_token_reduction < 0:
                notes.append(
                    "Optimized prompt used more prompt tokens due to system instructions."
                )

        if baseline_completion is not None and optimized_completion is not None:
            completions_match = _normalize_completion(baseline_completion) == _normalize_completion(
                optimized_completion
            )
            if completions_match:
                notes.append("Baseline and optimized completions match (normalized).")
            else:
                passed_quality_gate = False
                notes.append(
                    "Baseline and optimized completions differ; review quality manually."
                )

        efficiency = EfficiencyComparison(
            word_reduction_percent=word_reduction,
            prompt_token_reduction_percent=prompt_token_reduction,
            total_token_reduction_percent=total_token_reduction,
            energy_savings_percent=energy_savings,
            latency_delta_ms=latency_delta,
        )

        return EvaluationResult(
            efficiency=efficiency,
            baseline_completion=baseline_completion,
            optimized_completion=optimized_completion,
            completions_match=completions_match,
            passed_quality_gate=passed_quality_gate,
            notes=notes,
        )
