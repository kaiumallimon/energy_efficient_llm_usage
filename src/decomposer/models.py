from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecomposedPrompt:
    intent: str
    core_request: str
    entities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    redundant_segments: list[str] = field(default_factory=list)
    context_summary: str | None = None
    requires_context: bool = False
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "core_request": self.core_request,
            "entities": self.entities,
            "constraints": self.constraints,
            "redundant_segments": self.redundant_segments,
            "context_summary": self.context_summary,
            "requires_context": self.requires_context,
            "source": self.source,
        }
