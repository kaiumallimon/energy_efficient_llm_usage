from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EfficiencyComparison:
    word_reduction_percent: float
    prompt_token_reduction_percent: float | None = None
    total_token_reduction_percent: float | None = None
    energy_savings_percent: float | None = None
    latency_delta_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word_reduction_percent": round(self.word_reduction_percent, 2),
            "prompt_token_reduction_percent": (
                round(self.prompt_token_reduction_percent, 2)
                if self.prompt_token_reduction_percent is not None
                else None
            ),
            "total_token_reduction_percent": (
                round(self.total_token_reduction_percent, 2)
                if self.total_token_reduction_percent is not None
                else None
            ),
            "energy_savings_percent": (
                round(self.energy_savings_percent, 2)
                if self.energy_savings_percent is not None
                else None
            ),
            "latency_delta_ms": (
                round(self.latency_delta_ms, 2) if self.latency_delta_ms is not None else None
            ),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Comparison of optimized pipeline output against an unoptimized baseline."""

    efficiency: EfficiencyComparison
    baseline_completion: str | None = None
    optimized_completion: str | None = None
    completions_match: bool | None = None
    passed_quality_gate: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "efficiency": self.efficiency.to_dict(),
            "baseline_completion": self.baseline_completion,
            "optimized_completion": self.optimized_completion,
            "completions_match": self.completions_match,
            "passed_quality_gate": self.passed_quality_gate,
            "notes": self.notes,
        }
