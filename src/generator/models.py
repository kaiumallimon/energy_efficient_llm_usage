from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.analyzer.models import ComplexityLevel, OptimizationPolicy, TaskType


class ModelTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class GeneratedPrompt:
    system_prompt: str
    user_prompt: str
    context: str | None
    messages: list[dict[str, str]]
    full_prompt: str
    model_tier: str
    template_id: str
    task_type: TaskType
    complexity_level: ComplexityLevel
    policy: OptimizationPolicy
    sections: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "context": self.context,
            "messages": self.messages,
            "full_prompt": self.full_prompt,
            "model_tier": self.model_tier,
            "template_id": self.template_id,
            "task_type": self.task_type.value,
            "complexity_level": self.complexity_level.value,
            "policy": self.policy.value,
            "sections": self.sections,
            "notes": self.notes,
        }
