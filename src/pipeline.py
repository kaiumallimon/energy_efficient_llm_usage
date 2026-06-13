from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.analyzer import ComplexityAnalyzer
from src.analyzer.models import ComplexityResult
from src.decomposer import DecomposedPrompt, PromptDecomposer
from src.evaluation import EvaluationResult, PipelineEvaluator
from src.generator import AdaptivePromptGenerator, GeneratedPrompt
from src.llm import LLMCallResult, OllamaClient
from src.monitoring import MonitoringSnapshot, PerformanceMonitor
from src.optimizer import OptimizationResult, PromptOptimizer
from src.pipeline_fallback import FallbackResult, QualityFallbackHandler


@dataclass(frozen=True)
class PipelineResult:
    analysis: ComplexityResult
    decomposition: DecomposedPrompt
    optimization: OptimizationResult
    generation: GeneratedPrompt
    llm: LLMCallResult | None = None
    baseline_llm: LLMCallResult | None = None
    monitoring: MonitoringSnapshot | None = None
    evaluation: EvaluationResult | None = None
    fallback: FallbackResult | None = None

    @property
    def completion(self) -> str | None:
        return self.llm.response if self.llm is not None else None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "analysis": self.analysis.to_dict(),
            "decomposition": self.decomposition.to_dict(),
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
        if self.fallback is not None:
            payload["fallback"] = self.fallback.to_dict()
        return payload


class PromptPipeline:
    """Runs intent analysis, decomposition, semantic optimization, validation, and generation."""

    def __init__(
        self,
        analyzer: ComplexityAnalyzer | None = None,
        decomposer: PromptDecomposer | None = None,
        optimizer: PromptOptimizer | None = None,
        generator: AdaptivePromptGenerator | None = None,
        llm_client: OllamaClient | None = None,
        *,
        use_ollama: bool = False,
    ) -> None:
        self.llm_client = llm_client
        client = llm_client

        if analyzer is None:
            from src.analyzer.intent_classifier import IntentClassifier

            self.analyzer = ComplexityAnalyzer(
                intent_classifier=IntentClassifier(client=client, use_ollama=use_ollama),
                use_ollama=use_ollama,
            )
        else:
            self.analyzer = analyzer

        self.decomposer = decomposer or PromptDecomposer(client=client, use_ollama=use_ollama)

        if optimizer is None:
            from src.optimizer.llm_optimizer import LLMPromptOptimizer
            from src.validator import QualityValidator

            self.optimizer = PromptOptimizer(
                llm_optimizer=LLMPromptOptimizer(client=client, use_ollama=use_ollama),
                validator=QualityValidator(client=client, use_ollama=use_ollama),
                use_ollama=use_ollama,
            )
        else:
            self.optimizer = optimizer

        self.generator = generator or AdaptivePromptGenerator()
        self.use_ollama = use_ollama

    def process(
        self,
        query: str,
        context: str | None = None,
        *,
        call_llm: bool = False,
        evaluate: bool = False,
        think: bool | None = None,
        fallback: bool = False,
    ) -> PipelineResult:
        analysis = self.analyzer.analyze(query, context)
        decomposition = self.decomposer.decompose(query, context, analysis)
        optimization = self.optimizer.optimize(query, context, analysis, decomposition)
        generation = self.generator.generate(analysis, optimization, decomposition)

        llm_result = None
        baseline_llm = None
        fallback_result = None

        if call_llm:
            client = self.llm_client or OllamaClient()
            llm_result = client.call(generation, think=think)
            if evaluate:
                baseline_llm = client.call_baseline(query, context, think=think)

            if fallback:
                from src.evaluation.quality_scorer import QualityScorer

                handler = QualityFallbackHandler(
                    generator=self.generator,
                    scorer=QualityScorer(
                        embedding_client=client,
                        judge_client=None,
                        use_llm_judge=False,
                    ),
                )
                fallback_result = handler.maybe_rerun(
                    query,
                    analysis,
                    optimization,
                    decomposition,
                    llm_result,
                    baseline_completion=baseline_llm.response if baseline_llm else None,
                    llm_client=client,
                    think=think,
                )
                if fallback_result.used_fallback and fallback_result.fallback_llm is not None:
                    llm_result = fallback_result.fallback_llm
                    generation = fallback_result.fallback_generation or generation

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
            decomposition=decomposition,
            optimization=optimization,
            generation=generation,
            llm=llm_result,
            baseline_llm=baseline_llm,
            monitoring=monitoring,
            evaluation=evaluation,
            fallback=fallback_result,
        )
