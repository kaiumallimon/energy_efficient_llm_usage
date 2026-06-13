"""Compare baseline and optimized LLM outputs with efficiency and quality metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.evaluation.quality_scorer import QualityScorer
from src.llm.models import LLMCallResult


@dataclass(frozen=True)
class OutputComparison:
    query: str
    baseline_completion: str
    optimized_completion: str
    quality: dict[str, Any]
    baseline_metrics: dict[str, Any] | None = None
    optimized_metrics: dict[str, Any] | None = None
    efficiency: dict[str, Any] | None = None
    acceptable_quality: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "baseline_completion": self.baseline_completion,
            "optimized_completion": self.optimized_completion,
            "quality": self.quality,
            "baseline_metrics": self.baseline_metrics,
            "optimized_metrics": self.optimized_metrics,
            "efficiency": self.efficiency,
            "acceptable_quality": self.acceptable_quality,
            "notes": self.notes,
        }


def _percent_reduction(baseline: float, optimized: float) -> float | None:
    if baseline <= 0:
        return None
    return ((baseline - optimized) / baseline) * 100.0


def _llm_metrics(result: LLMCallResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "model": result.model,
        "latency_ms": round(result.latency_ms, 2),
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
        "energy_proxy": round(result.energy_proxy, 4),
        "energy_measured_j": (
            round(result.energy_measured_j, 4)
            if result.energy_measured_j is not None
            else None
        ),
    }


class OutputComparator:
    """Compares optimized and baseline completions."""

    def __init__(self, scorer: QualityScorer | None = None) -> None:
        self.scorer = scorer or QualityScorer(use_llm_judge=False)

    def compare(
        self,
        query: str,
        baseline_completion: str,
        optimized_completion: str,
        *,
        baseline_llm: LLMCallResult | None = None,
        optimized_llm: LLMCallResult | None = None,
        min_quality_score: float = 0.55,
        min_rubric_overall: float = 3.0,
    ) -> OutputComparison:
        notes: list[str] = []
        quality = self.scorer.score(query, baseline_completion, optimized_completion)
        acceptable = quality.overall_score >= min_quality_score

        if quality.rubric is not None and quality.rubric.overall < min_rubric_overall:
            acceptable = False
            notes.append(
                f"Rubric overall {quality.rubric.overall:.1f} below threshold {min_rubric_overall:.1f}."
            )
        elif acceptable:
            notes.append("Optimized output meets automatic quality threshold.")

        efficiency = None
        if baseline_llm is not None and optimized_llm is not None:
            efficiency = {
                "total_token_reduction_percent": _percent_reduction(
                    float(baseline_llm.usage.total_tokens),
                    float(optimized_llm.usage.total_tokens),
                ),
                "completion_token_reduction_percent": _percent_reduction(
                    float(baseline_llm.usage.completion_tokens),
                    float(optimized_llm.usage.completion_tokens),
                ),
                "latency_delta_ms": round(
                    optimized_llm.latency_ms - baseline_llm.latency_ms,
                    2,
                ),
                "energy_proxy_savings_percent": _percent_reduction(
                    baseline_llm.energy_proxy,
                    optimized_llm.energy_proxy,
                ),
                "energy_measured_savings_percent": (
                    _percent_reduction(
                        baseline_llm.energy_measured_j,
                        optimized_llm.energy_measured_j,
                    )
                    if baseline_llm.energy_measured_j is not None
                    and optimized_llm.energy_measured_j is not None
                    else None
                ),
            }

        return OutputComparison(
            query=query,
            baseline_completion=baseline_completion,
            optimized_completion=optimized_completion,
            quality=quality.to_dict(),
            baseline_metrics=_llm_metrics(baseline_llm),
            optimized_metrics=_llm_metrics(optimized_llm),
            efficiency=efficiency,
            acceptable_quality=acceptable,
            notes=notes,
        )
