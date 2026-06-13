from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.analyzer import ComplexityAnalyzer
from src.analyzer.models import ComplexityResult
from src.evaluation import EvaluationResult, PipelineEvaluator
from src.generator import AdaptivePromptGenerator, GeneratedPrompt
from src.llm import LLMCallResult, OllamaClient
from src.monitoring import MonitoringSnapshot, PerformanceMonitor
from src.optimizer import OptimizationResult, PromptOptimizer


@dataclass(frozen=True)
class PipelineResult:
    analysis: ComplexityResult
    optimization: OptimizationResult
    generation: GeneratedPrompt
    llm: LLMCallResult | None = None
    baseline_llm: LLMCallResult | None = None
    monitoring: MonitoringSnapshot | None = None
    evaluation: EvaluationResult | None = None

    @property
    def completion(self) -> str | None:
        return self.llm.response if self.llm is not None else None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "analysis": self.analysis.to_dict(),
            "optimization": self.optimization.to_dict(),
            "generation": self.generation.to_dict(),
        }
        if self.llm is not None:
            payload["llm"] = self.llm.to_dict()
            payload["completion"] = self.llm.response
        if self.baseline_llm is not None:
            payload["baseline_llm"] = self.baseline_llm.to_dict()
        if self.monitoring is not None:
            payload["monitoring"] = self.monitoring.to_dict()
        if self.evaluation is not None:
            payload["evaluation"] = self.evaluation.to_dict()
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
        evaluate: bool = False,
        think: bool | None = None,
    ) -> PipelineResult:
        analysis = self.analyzer.analyze(query, context)
        optimization = self.optimizer.optimize(query, context, analysis)
        generation = self.generator.generate(analysis, optimization)

        llm_result = None
        baseline_llm = None
        client = None

        if call_llm or evaluate:
            client = self.llm_client or OllamaClient()

        if call_llm:
            llm_result = client.call(generation, think=think)

        if evaluate and call_llm:
            baseline_llm = client.call_baseline(query, context, think=think)

        monitoring = PerformanceMonitor.collect(
            query,
            context,
            analysis,
            optimization,
            generation,
            llm_result=llm_result,
            baseline_llm=baseline_llm,
        )

        evaluation = None
        if evaluate:
            evaluation = PipelineEvaluator.evaluate(
                optimization,
                monitoring,
                llm_result=llm_result,
                baseline_llm=baseline_llm,
            )

        return PipelineResult(
            analysis=analysis,
            optimization=optimization,
            generation=generation,
            llm=llm_result,
            baseline_llm=baseline_llm,
            monitoring=monitoring,
            evaluation=evaluation,
        )
