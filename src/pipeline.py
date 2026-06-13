from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.analyzer import ComplexityAnalyzer
from src.analyzer.models import ComplexityResult
from src.optimizer import OptimizationResult, PromptOptimizer


@dataclass(frozen=True)
class PipelineResult:
    analysis: ComplexityResult
    optimization: OptimizationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis.to_dict(),
            "optimization": self.optimization.to_dict(),
        }


class PromptPipeline:
    """Runs complexity analysis and policy-driven prompt optimization."""

    def __init__(
        self,
        analyzer: ComplexityAnalyzer | None = None,
        optimizer: PromptOptimizer | None = None,
    ) -> None:
        self.analyzer = analyzer or ComplexityAnalyzer()
        self.optimizer = optimizer or PromptOptimizer()

    def process(self, query: str, context: str | None = None) -> PipelineResult:
        analysis = self.analyzer.analyze(query, context)
        optimization = self.optimizer.optimize(query, context, analysis)
        return PipelineResult(analysis=analysis, optimization=optimization)
