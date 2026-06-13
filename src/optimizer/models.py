from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.analyzer.models import ComplexityLevel, OptimizationPolicy


@dataclass(frozen=True)
class OptimizationResult:
    original_query: str
    optimized_query: str
    original_context: str | None
    optimized_context: str | None
    policy: OptimizationPolicy
    complexity_level: ComplexityLevel
    changes: list[str] = field(default_factory=list)
    original_word_count: int = 0
    optimized_word_count: int = 0
    word_reduction_percent: float = 0.0
    was_modified: bool = False
    validation: Any | None = None
    optimizer_source: str = "hybrid"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "original_query": self.original_query,
            "optimized_query": self.optimized_query,
            "original_context": self.original_context,
            "optimized_context": self.optimized_context,
            "policy": self.policy.value,
            "complexity_level": self.complexity_level.value,
            "changes": self.changes,
            "original_word_count": self.original_word_count,
            "optimized_word_count": self.optimized_word_count,
            "word_reduction_percent": round(self.word_reduction_percent, 2),
            "was_modified": self.was_modified,
            "optimizer_source": self.optimizer_source,
        }
        if self.validation is not None and hasattr(self.validation, "to_dict"):
            payload["validation"] = self.validation.to_dict()
        return payload
