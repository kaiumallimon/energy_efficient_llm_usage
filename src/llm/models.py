from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "thinking_enabled": self.thinking_enabled,
        }


@dataclass(frozen=True)
class LLMCallResult:
    model: str
    provider: str
    response: str
    thinking: str | None
    usage: TokenUsage
    latency_ms: float
    load_duration_ms: float | None = None
    eval_duration_ms: float | None = None
    energy_proxy: float = 0.0
    energy_measured_j: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def completion(self) -> str:
        return self.response

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "response": self.response,
            "completion": self.response,
            "thinking": self.thinking,
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "load_duration_ms": (
                round(self.load_duration_ms, 2) if self.load_duration_ms is not None else None
            ),
            "eval_duration_ms": (
                round(self.eval_duration_ms, 2) if self.eval_duration_ms is not None else None
            ),
            "energy_proxy": round(self.energy_proxy, 4),
            "energy_measured_j": (
                round(self.energy_measured_j, 4)
                if self.energy_measured_j is not None
                else None
            ),
        }
