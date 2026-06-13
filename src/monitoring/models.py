from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PathMetrics:
    """Runtime metrics for one execution path (baseline or optimized)."""

    path: str
    prompt_words: int
    model_tier: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    energy_proxy: float | None = None
    completion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "prompt_words": self.prompt_words,
            "model_tier": self.model_tier,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms is not None else None,
            "energy_proxy": round(self.energy_proxy, 4) if self.energy_proxy is not None else None,
            "completion": self.completion,
        }


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Collected telemetry for a single pipeline run."""

    optimized: PathMetrics
    baseline: PathMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"optimized": self.optimized.to_dict()}
        if self.baseline is not None:
            payload["baseline"] = self.baseline.to_dict()
        return payload
