from __future__ import annotations

import pytest

from src.analyzer.models import ComplexityLevel, OptimizationPolicy, TaskType
from src.decomposer import PromptDecomposer
from src.pipeline import PromptPipeline
from src.validator import QualityValidator, cosine_similarity


def make_analysis(
    task_type: TaskType = TaskType.FACTUAL,
    policy: OptimizationPolicy = OptimizationPolicy.AGGRESSIVE,
    level: ComplexityLevel = ComplexityLevel.LOW,
):
    from src.analyzer.models import ComplexityResult

    return ComplexityResult(
        level=level,
        score=10.0,
        task_type=task_type,
        policy=policy,
        confidence=0.8,
    )


class TestPromptDecomposer:
    def test_decompose_short_definition(self) -> None:
        analysis = make_analysis(TaskType.DEFINITION)
        result = PromptDecomposer(use_ollama=False).decompose(
            "what is CFG shortly?",
            None,
            analysis,
        )

        assert result.intent == TaskType.DEFINITION.value
        assert "CFG" in result.core_request
        assert result.source == "heuristic"


class TestQualityValidator:
    def test_identical_queries_pass(self) -> None:
        result = QualityValidator(use_ollama=False).validate(
            "what is CFG shortly?",
            "what is CFG shortly?",
        )

        assert result.passed is True
        assert result.similarity == 1.0

    def test_different_queries_fail_without_embeddings(self) -> None:
        result = QualityValidator(use_ollama=False).validate(
            "what is CFG shortly?",
            "tell me a joke",
        )

        assert result.passed is False
        assert result.accepted_query == "what is CFG shortly?"

    def test_cosine_similarity(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


class TestSemanticPipeline:
    def test_pipeline_includes_decomposition_stage(self) -> None:
        result = PromptPipeline(use_ollama=False).process("what is CFG shortly?")

        assert result.decomposition.core_request
        assert result.analysis.task_type != TaskType.CODING
        assert result.generation.inference_params.max_tokens > 0
        assert "decomposition" in result.to_dict()

    def test_definition_template_not_coding(self) -> None:
        result = PromptPipeline(use_ollama=False).process("what is CFG shortly?")

        assert "runnable code" not in result.generation.system_prompt.lower()
        assert result.generation.template_id.startswith(
            (
                TaskType.DEFINITION.value,
                TaskType.CONCEPT_EXPLANATION.value,
                TaskType.FACTUAL.value,
            )
        )
