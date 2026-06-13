from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.analyzer import ComplexityAnalyzer
from src.analyzer.models import ComplexityResult
from src.generator import AdaptivePromptGenerator, GeneratedPrompt
from src.llm import LLMCallResult, OllamaClient
from src.optimizer import OptimizationResult, PromptOptimizer


@dataclass(frozen=True)
class PipelineResult:
    analysis: ComplexityResult
    optimization: OptimizationResult
    generation: GeneratedPrompt
    llm: LLMCallResult | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "analysis": self.analysis.to_dict(),
            "optimization": self.optimization.to_dict(),
            "generation": self.generation.to_dict(),
        }
        if self.llm is not None:
            payload["llm"] = self.llm.to_dict()
        return payload


class PromptPipeline:
    """Runs complexity analysis, optimization, prompt generation, and optional LLM call."""

    def __init__(
        self,
        analyzer: ComplexityAnalyzer | None = None,
        optimizer: PromptOptimizer | None = None,
        generator: AdaptivePromptGenerator | None = None,
        llm_client: OllamaClient | None = None,
    ) -> None:
        self.analyzer = analyzer or ComplexityAnalyzer()
        self.optimizer = optimizer or PromptOptimizer()
        self.generator = generator or AdaptivePromptGenerator()
        self.llm_client = llm_client

    def process(
        self,
        query: str,
        context: str | None = None,
        *,
        call_llm: bool = False,
        think: bool | None = None,
    ) -> PipelineResult:
        analysis = self.analyzer.analyze(query, context)
        optimization = self.optimizer.optimize(query, context, analysis)
        generation = self.generator.generate(analysis, optimization)

        llm_result = None
        if call_llm:
            client = self.llm_client or OllamaClient()
            llm_result = client.call(generation, think=think)

        return PipelineResult(
            analysis=analysis,
            optimization=optimization,
            generation=generation,
            llm=llm_result,
        )
