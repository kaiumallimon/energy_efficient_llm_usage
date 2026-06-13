from __future__ import annotations

import pytest

from src.analyzer import ComplexityAnalyzer
from src.analyzer.models import ComplexityLevel, OptimizationPolicy, TaskType


@pytest.fixture
def analyzer() -> ComplexityAnalyzer:
    return ComplexityAnalyzer()


class TestSimplePrompts:
    def test_short_factual_question_is_low(self, analyzer: ComplexityAnalyzer) -> None:
        result = analyzer.analyze("What is the capital of France?")

        assert result.level == ComplexityLevel.LOW
        assert result.task_type == TaskType.FACTUAL
        assert result.policy == OptimizationPolicy.AGGRESSIVE
        assert result.score < 35

    def test_greeting_is_low(self, analyzer: ComplexityAnalyzer) -> None:
        result = analyzer.analyze("Hello there")

        assert result.level == ComplexityLevel.LOW
        assert result.task_type == TaskType.CONVERSATIONAL


class TestReasoningPrompts:
    def test_compare_and_explain_is_high(self, analyzer: ComplexityAnalyzer) -> None:
        query = (
            "Compare microservices and monolith architectures for a fintech startup. "
            "Explain trade-offs step by step and evaluate operational complexity."
        )
        result = analyzer.analyze(query)

        assert result.task_type == TaskType.REASONING
        assert result.level in {ComplexityLevel.MEDIUM, ComplexityLevel.HIGH}
        assert result.policy in {
            OptimizationPolicy.MODERATE,
            OptimizationPolicy.CONSERVATIVE,
        }


class TestCodingPrompts:
    def test_debug_request_is_at_least_medium(self, analyzer: ComplexityAnalyzer) -> None:
        query = (
            "Debug this Python function that parses JSON from an API response. "
            "Return only the fixed code and explain the bug."
        )
        result = analyzer.analyze(query)

        assert result.task_type == TaskType.CODING
        assert result.level in {
            ComplexityLevel.MEDIUM,
            ComplexityLevel.HIGH,
            ComplexityLevel.CRITICAL,
        }


class TestSafetySensitivePrompts:
    def test_medical_prompt_is_critical(self, analyzer: ComplexityAnalyzer) -> None:
        query = "What dosage should I take for these medical symptoms?"
        result = analyzer.analyze(query)

        assert result.level == ComplexityLevel.CRITICAL
        assert result.policy == OptimizationPolicy.MINIMAL
        assert result.signals["safety_score"] >= 0.35


class TestConstraintPrompts:
    def test_json_format_constraints_raise_complexity(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        query = (
            "Extract names and emails from the text. "
            "Respond with JSON only and do not include extra commentary."
        )
        result = analyzer.analyze(query)

        assert result.task_type in {TaskType.EXTRACTION, TaskType.FACTUAL}
        assert result.signals["constraint_score"] >= 0.3
        assert result.level != ComplexityLevel.LOW


class TestContextHandling:
    def test_large_context_increases_score(self, analyzer: ComplexityAnalyzer) -> None:
        query = "Summarize the key points."
        short_context = "Short meeting notes about a team sync."
        long_context = "word " * 250

        short_result = analyzer.analyze(query, short_context)
        long_result = analyzer.analyze(query, long_context)

        assert long_result.score > short_result.score
        assert long_result.signals["context_word_count"] > short_result.signals["context_word_count"]


class TestResultShape:
    def test_to_dict_contains_expected_fields(self, analyzer: ComplexityAnalyzer) -> None:
        result = analyzer.analyze("Define photosynthesis.")
        payload = result.to_dict()

        assert payload["level"] == ComplexityLevel.LOW.value
        assert "score" in payload
        assert "signals" in payload
        assert "rationale" in payload
        assert isinstance(payload["rationale"], list)
