from __future__ import annotations

import pytest

from src.analyzer.models import ComplexityLevel, OptimizationPolicy, TaskType
from src.generator import AdaptivePromptGenerator, ModelTier
from src.optimizer.models import OptimizationResult
from src.pipeline import PromptPipeline


@pytest.fixture
def generator() -> AdaptivePromptGenerator:
    return AdaptivePromptGenerator()


def make_optimization(
    query: str,
    context: str | None = None,
    policy: OptimizationPolicy = OptimizationPolicy.MODERATE,
    level: ComplexityLevel = ComplexityLevel.MEDIUM,
) -> OptimizationResult:
    return OptimizationResult(
        original_query=query,
        optimized_query=query,
        original_context=context,
        optimized_context=context,
        policy=policy,
        complexity_level=level,
    )


def make_analysis(
    task_type: TaskType,
    level: ComplexityLevel,
    policy: OptimizationPolicy,
    signals: dict | None = None,
):
    from src.analyzer.models import ComplexityResult

    return ComplexityResult(
        level=level,
        score=40.0,
        task_type=task_type,
        policy=policy,
        confidence=0.7,
        signals=signals or {},
    )


class TestAdaptivePromptGenerator:
    def test_factual_low_uses_small_model_tier(self, generator: AdaptivePromptGenerator) -> None:
        analysis = make_analysis(
            TaskType.FACTUAL,
            ComplexityLevel.LOW,
            OptimizationPolicy.AGGRESSIVE,
        )
        optimization = make_optimization(
            "What is the capital of France?",
            policy=OptimizationPolicy.AGGRESSIVE,
            level=ComplexityLevel.LOW,
        )

        result = generator.generate(analysis, optimization)

        assert result.model_tier == ModelTier.SMALL.value
        assert result.template_id == "factual_aggressive"
        assert result.user_prompt == optimization.optimized_query
        assert result.context is None
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "system"
        assert result.messages[1]["role"] == "user"

    def test_reasoning_medium_uses_medium_model_tier(
        self, generator: AdaptivePromptGenerator
    ) -> None:
        query = "Compare REST and GraphQL trade-offs."
        analysis = make_analysis(
            TaskType.REASONING,
            ComplexityLevel.MEDIUM,
            OptimizationPolicy.MODERATE,
        )
        optimization = make_optimization(query)

        result = generator.generate(analysis, optimization)

        assert result.model_tier == ModelTier.MEDIUM.value
        assert "structure for comparisons" in result.system_prompt.lower()
        assert result.sections["query"] == query

    def test_coding_high_uses_large_model_tier(self, generator: AdaptivePromptGenerator) -> None:
        analysis = make_analysis(
            TaskType.CODING,
            ComplexityLevel.HIGH,
            OptimizationPolicy.CONSERVATIVE,
        )
        optimization = make_optimization(
            "Debug this Python function.",
            policy=OptimizationPolicy.CONSERVATIVE,
            level=ComplexityLevel.HIGH,
        )

        result = generator.generate(analysis, optimization)

        assert result.model_tier == ModelTier.LARGE.value
        assert "runnable code" in result.system_prompt.lower()

    def test_safety_critical_adds_guardrails(self, generator: AdaptivePromptGenerator) -> None:
        analysis = make_analysis(
            TaskType.FACTUAL,
            ComplexityLevel.CRITICAL,
            OptimizationPolicy.MINIMAL,
            signals={"safety_score": 0.7},
        )
        optimization = make_optimization(
            "What dosage should I take?",
            policy=OptimizationPolicy.MINIMAL,
            level=ComplexityLevel.CRITICAL,
        )

        result = generator.generate(analysis, optimization)

        assert result.model_tier == ModelTier.LARGE.value
        assert "medical" in result.system_prompt.lower()
        assert any("safety" in note.lower() for note in result.notes)

    def test_context_is_structured_in_user_prompt(
        self, generator: AdaptivePromptGenerator
    ) -> None:
        analysis = make_analysis(
            TaskType.SUMMARIZATION,
            ComplexityLevel.MEDIUM,
            OptimizationPolicy.MODERATE,
        )
        optimization = make_optimization(
            "Summarize in 3 bullets.",
            context="Revenue grew 12%. Hiring paused in Q2.",
        )

        result = generator.generate(analysis, optimization)

        assert result.context == optimization.optimized_context
        assert "Context:" in result.user_prompt
        assert "Request:" in result.user_prompt
        assert result.sections["context"] == optimization.optimized_context

    def test_constraint_signals_add_preservation_note(
        self, generator: AdaptivePromptGenerator
    ) -> None:
        analysis = make_analysis(
            TaskType.EXTRACTION,
            ComplexityLevel.MEDIUM,
            OptimizationPolicy.MODERATE,
            signals={"constraint_score": 0.5},
        )
        optimization = make_optimization("Return JSON only.")

        result = generator.generate(analysis, optimization)

        assert "output-format constraints" in result.system_prompt.lower()
        assert any("constraint" in note.lower() for note in result.notes)

    def test_to_dict_contains_messages_and_full_prompt(
        self, generator: AdaptivePromptGenerator
    ) -> None:
        analysis = make_analysis(
            TaskType.CONVERSATIONAL,
            ComplexityLevel.LOW,
            OptimizationPolicy.AGGRESSIVE,
        )
        optimization = make_optimization("Hello there")

        payload = generator.generate(analysis, optimization).to_dict()

        assert "messages" in payload
        assert "full_prompt" in payload
        assert payload["task_type"] == TaskType.CONVERSATIONAL.value


class TestPromptPipelineWithGenerator:
    def test_pipeline_returns_generation_stage(self) -> None:
        result = PromptPipeline().process("What is 2 + 2?")

        assert result.generation is not None
        assert result.generation.user_prompt == result.optimization.optimized_query
        assert result.generation.policy == result.analysis.policy
        assert result.generation.task_type == result.analysis.task_type

    def test_pipeline_to_dict_includes_generation(self) -> None:
        payload = PromptPipeline().process("Hello there").to_dict()

        assert "generation" in payload
        assert "model_tier" in payload["generation"]
        assert "template_id" in payload["generation"]

    def test_verbose_query_flows_through_all_stages(self) -> None:
        query = "Could you please tell me what the capital of France is?"
        result = PromptPipeline().process(query)

        assert result.optimization.was_modified
        assert result.generation.user_prompt == result.optimization.optimized_query
        assert result.generation.model_tier == ModelTier.SMALL.value
