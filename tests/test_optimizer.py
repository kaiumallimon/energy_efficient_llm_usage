from __future__ import annotations

import pytest

from src.analyzer.models import ComplexityLevel, OptimizationPolicy
from src.optimizer import PromptOptimizer
from src.pipeline import PromptPipeline


@pytest.fixture
def optimizer() -> PromptOptimizer:
    return PromptOptimizer()


def make_analysis(policy: OptimizationPolicy, level: ComplexityLevel = ComplexityLevel.LOW):
    from src.analyzer.models import ComplexityResult, TaskType

    return ComplexityResult(
        level=level,
        score=10.0,
        task_type=TaskType.FACTUAL,
        policy=policy,
        confidence=0.8,
    )


class TestPromptOptimizer:
    def test_aggressive_removes_polite_fillers(self, optimizer: PromptOptimizer) -> None:
        query = "Could you please tell me what the capital of France is?"
        result = optimizer.optimize(query, None, make_analysis(OptimizationPolicy.AGGRESSIVE))

        assert result.was_modified
        assert "please" not in result.optimized_query.lower()
        assert result.optimized_word_count < result.original_word_count

    def test_aggressive_compresses_redundant_phrasing(self, optimizer: PromptOptimizer) -> None:
        query = "Explain in order to understand why this happens."
        result = optimizer.optimize(query, None, make_analysis(OptimizationPolicy.AGGRESSIVE))

        assert "in order to" not in result.optimized_query.lower()
        assert "to understand" in result.optimized_query.lower()

    def test_minimal_policy_leaves_prompt_unchanged(self, optimizer: PromptOptimizer) -> None:
        query = "Could you please explain medical dosage options?"
        result = optimizer.optimize(
            query,
            None,
            make_analysis(OptimizationPolicy.MINIMAL, ComplexityLevel.CRITICAL),
        )

        assert result.optimized_query == query
        assert result.word_reduction_percent == 0.0

    def test_aggressive_trims_long_context(self, optimizer: PromptOptimizer) -> None:
        query = "Summarize this."
        context = "word " * 300
        result = optimizer.optimize(query, context, make_analysis(OptimizationPolicy.AGGRESSIVE))

        assert result.optimized_context is not None
        assert result.optimized_word_count < result.original_word_count

    def test_conservative_keeps_query_but_dedupes_context(
        self, optimizer: PromptOptimizer
    ) -> None:
        query = "Analyze the attached notes."
        context = "Line one.\nLine one.\nLine two."
        result = optimizer.optimize(
            query,
            context,
            make_analysis(OptimizationPolicy.CONSERVATIVE, ComplexityLevel.HIGH),
        )

        assert result.optimized_query == query
        assert result.optimized_context is not None
        assert result.optimized_context.count("Line one.") == 1

    def test_real_world_verbose_factual_question(self, optimizer: PromptOptimizer) -> None:
        query = "could you please tell me what is the capital of Bangladesh??"
        result = optimizer.optimize(query, None, make_analysis(OptimizationPolicy.AGGRESSIVE))

        assert result.was_modified
        assert "please" not in result.optimized_query.lower()
        assert "could you" not in result.optimized_query.lower()
        assert "bangladesh" in result.optimized_query.lower()
        assert result.optimized_query.endswith("?")
        assert result.optimized_query.startswith("What")

    def test_real_world_awkward_math_phrasing(self, optimizer: PromptOptimizer) -> None:
        query = "could you please calculate what is the answer of 2+2 is?"
        result = optimizer.optimize(query, None, make_analysis(OptimizationPolicy.AGGRESSIVE))

        assert result.was_modified
        assert "please" not in result.optimized_query.lower()
        assert "2+2" in result.optimized_query.replace(" ", "")

    def test_real_world_repeated_polite_openers(self, optimizer: PromptOptimizer) -> None:
        query = (
            "Hey, so I was wondering if you could please tell me, if you don't mind, "
            "what is the capital of France?"
        )
        result = optimizer.optimize(query, None, make_analysis(OptimizationPolicy.AGGRESSIVE))

        assert result.was_modified
        assert "wondering" not in result.optimized_query.lower()
        assert "france" in result.optimized_query.lower()


class TestPromptPipeline:
    def test_pipeline_links_analyzer_and_optimizer(self) -> None:
        query = "Could you please tell me what 2 + 2 equals?"
        result = PromptPipeline().process(query)

        assert result.analysis.policy == OptimizationPolicy.AGGRESSIVE
        assert result.optimization.policy == result.analysis.policy
        assert result.optimization.complexity_level == result.analysis.level
        assert result.optimization.was_modified

    def test_pipeline_to_dict_has_all_stages(self) -> None:
        payload = PromptPipeline().process("Hello there").to_dict()

        assert "analysis" in payload
        assert "optimization" in payload
        assert "generation" in payload
        assert "monitoring" in payload
        assert "policy" in payload["analysis"]
        assert "optimized_query" in payload["optimization"]
        assert "messages" in payload["generation"]

    def test_critical_prompt_is_not_aggressively_modified(self) -> None:
        query = "What dosage should I take for these medical symptoms?"
        result = PromptPipeline().process(query)

        assert result.analysis.policy == OptimizationPolicy.MINIMAL
        assert result.optimization.optimized_query == query
