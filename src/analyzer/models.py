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
    DEFINITION = "definition"
    CONCEPT_EXPLANATION = "concept_explanation"
    EDUCATIONAL = "educational"
    EXAM_HELP = "exam_help"
    CREATIVE = "creative"
    REASONING = "reasoning"
    CODING = "coding"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


INTENT_LABELS: tuple[str, ...] = (
    TaskType.EDUCATIONAL.value,
    TaskType.FACTUAL.value,
    TaskType.DEFINITION.value,
    TaskType.CONCEPT_EXPLANATION.value,
    TaskType.EXAM_HELP.value,
    TaskType.CODING.value,
    TaskType.REASONING.value,
    TaskType.SUMMARIZATION.value,
    TaskType.EXTRACTION.value,
    TaskType.CREATIVE.value,
    TaskType.CONVERSATIONAL.value,
)


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
    intent_source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "score": round(self.score, 2),
            "task_type": self.task_type.value,
            "policy": self.policy.value,
            "confidence": round(self.confidence, 2),
            "signals": self.signals,
            "rationale": self.rationale,
            "intent_source": self.intent_source,
        }
