from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComplexityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    FACTUAL = "factual"
    CREATIVE = "creative"
    REASONING = "reasoning"
    CODING = "coding"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


class OptimizationPolicy(str, Enum):
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    MINIMAL = "minimal"


@dataclass(frozen=True)
class ComplexityResult:
    level: ComplexityLevel
    score: float
    task_type: TaskType
    policy: OptimizationPolicy
    confidence: float
    signals: dict[str, Any] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "score": round(self.score, 2),
            "task_type": self.task_type.value,
            "policy": self.policy.value,
            "confidence": round(self.confidence, 2),
            "signals": self.signals,
            "rationale": self.rationale,
        }
