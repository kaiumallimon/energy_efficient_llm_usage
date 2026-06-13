from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.evaluation import PipelineEvaluator
from src.llm.models import LLMCallResult, TokenUsage
from src.monitoring import MonitoringSnapshot, PathMetrics, PerformanceMonitor
from src.optimizer.models import OptimizationResult
from src.pipeline import PromptPipeline
from src.analyzer.models import ComplexityLevel, OptimizationPolicy, TaskType


def make_optimization(
    *,
    word_reduction: float = 25.0,
    was_modified: bool = True,
) -> OptimizationResult:
    return OptimizationResult(
        original_query="Could you please tell me what 2 + 2 is?",
        optimized_query="tell me what 2 + 2 is?",
        original_context=None,
        optimized_context=None,
        policy=OptimizationPolicy.AGGRESSIVE,
        complexity_level=ComplexityLevel.LOW,
        original_word_count=12,
        optimized_word_count=9,
        word_reduction_percent=word_reduction,
        was_modified=was_modified,
    )


def make_llm_result(
    response: str,
    *,
    prompt_tokens: int = 30,
    completion_tokens: int = 2,
) -> LLMCallResult:
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return LLMCallResult(
        model="qwen3.5:4b",
        provider="ollama",
        response=response,
        thinking=None,
        usage=usage,
        latency_ms=100.0,
        energy_proxy=usage.total_tokens * 0.0008,
    )


class TestPerformanceMonitor:
    def test_collects_offline_optimized_metrics(self) -> None:
        result = PromptPipeline().process("What is 2 + 2?")

        assert result.monitoring is not None
        assert result.monitoring.optimized.path == "optimized"
        assert result.monitoring.optimized.prompt_words == result.optimization.optimized_word_count
        assert result.monitoring.optimized.total_tokens is None
        assert result.monitoring.baseline is None

    def test_collects_live_metrics_for_both_paths(self) -> None:
        optimization = make_optimization()
        monitoring = MonitoringSnapshot(
            optimized=PathMetrics(
                path="optimized",
                prompt_words=9,
                model_tier="small",
                model="qwen3.5:4b",
                prompt_tokens=20,
                completion_tokens=2,
                total_tokens=22,
                latency_ms=90.0,
                energy_proxy=0.0176,
                completion="4",
            ),
            baseline=PathMetrics(
                path="baseline",
                prompt_words=12,
                model="qwen3.5:4b",
                prompt_tokens=30,
                completion_tokens=2,
                total_tokens=32,
                latency_ms=110.0,
                energy_proxy=0.0256,
                completion="4",
            ),
        )

        evaluation = PipelineEvaluator.evaluate(
            optimization,
            monitoring,
            llm_result=make_llm_result("4", prompt_tokens=20),
            baseline_llm=make_llm_result("4", prompt_tokens=30),
        )

        assert evaluation.efficiency.word_reduction_percent == 25.0
        assert evaluation.efficiency.prompt_token_reduction_percent == pytest.approx(33.33, abs=0.1)
        assert evaluation.completions_match is True
        assert evaluation.passed_quality_gate is True


class TestPipelineEvaluator:
    def test_offline_evaluation_reports_word_reduction(self) -> None:
        result = PromptPipeline().process(
            "Could you please tell me what 2 + 2 equals?",
            evaluate=True,
        )

        assert result.evaluation is not None
        assert result.evaluation.efficiency.word_reduction_percent > 0
        assert result.evaluation.optimized_completion is None
        assert result.evaluation.baseline_completion is None
        assert result.evaluation.completions_match is None

    def test_live_evaluation_flags_different_completions(self) -> None:
        optimization = make_optimization()
        monitoring = MonitoringSnapshot(
            optimized=PathMetrics(
                path="optimized",
                prompt_words=9,
                prompt_tokens=20,
                completion_tokens=2,
                total_tokens=22,
                completion="4",
            ),
            baseline=PathMetrics(
                path="baseline",
                prompt_words=12,
                prompt_tokens=30,
                completion_tokens=4,
                total_tokens=34,
                completion="The answer is 4.",
            ),
        )

        evaluation = PipelineEvaluator.evaluate(
            optimization,
            monitoring,
            llm_result=make_llm_result("4"),
            baseline_llm=make_llm_result("The answer is 4."),
        )

        assert evaluation.completions_match is False
        assert evaluation.passed_quality_gate is False


class TestPipelineIntegration:
    def test_completion_exposed_at_top_level_when_llm_called(self) -> None:
        mock_client = MagicMock()
        mock_client.call.return_value = make_llm_result("4")

        result = PromptPipeline(llm_client=mock_client).process(
            "What is 2 + 2?",
            call_llm=True,
            think=False,
        )

        assert result.completion == "4"
        payload = result.to_dict()
        assert payload["completion"] == "4"
        assert payload["llm"]["completion"] == "4"
        assert "monitoring" in payload

    def test_evaluate_with_call_runs_baseline_and_optimized(self) -> None:
        mock_client = MagicMock()
        mock_client.call.return_value = make_llm_result("4", prompt_tokens=18)
        mock_client.call_baseline.return_value = make_llm_result("4", prompt_tokens=28)

        result = PromptPipeline(llm_client=mock_client).process(
            "Could you please tell me what 2 + 2 is?",
            call_llm=True,
            evaluate=True,
            think=False,
        )

        mock_client.call.assert_called_once()
        mock_client.call_baseline.assert_called_once()
        assert result.baseline_llm is not None
        assert result.evaluation is not None
        assert result.evaluation.baseline_completion == "4"
        assert result.evaluation.optimized_completion == "4"
        assert "evaluation" in result.to_dict()

    def test_monitoring_always_present(self) -> None:
        payload = PromptPipeline().process("Hello there").to_dict()

        assert "monitoring" in payload
        assert payload["monitoring"]["optimized"]["path"] == "optimized"
