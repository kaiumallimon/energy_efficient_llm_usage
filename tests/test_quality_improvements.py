from __future__ import annotations

import pytest

from src.evaluation.comparison import OutputComparator
from src.evaluation.quality_scorer import QualityScorer
from src.generator.policy_matrix import (
    COMPLEXITY_ACTIONS,
    relax_policy,
    select_policy_action,
)
from src.analyzer.models import ComplexityLevel, ComplexityResult, OptimizationPolicy, TaskType
from src.llm.models import LLMCallResult, TokenUsage
from src.monitoring.energy import EnergyCalibration, EnergyMonitor, MeasuredEnergy
from src.pipeline_fallback import QualityFallbackHandler


def make_analysis(
    *,
    level: ComplexityLevel = ComplexityLevel.LOW,
    policy: OptimizationPolicy = OptimizationPolicy.AGGRESSIVE,
    task_type: TaskType = TaskType.DEFINITION,
) -> ComplexityResult:
    return ComplexityResult(
        level=level,
        score=20.0,
        task_type=task_type,
        policy=policy,
        confidence=0.8,
        signals={},
    )


def make_llm_result(response: str) -> LLMCallResult:
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return LLMCallResult(
        model="qwen3.5:4b",
        provider="ollama",
        response=response,
        thinking=None,
        usage=usage,
        latency_ms=50.0,
        energy_proxy=0.01,
        energy_measured_j=1.2,
    )


class TestPolicyMatrix:
    def test_low_complexity_uses_aggressive_budget(self) -> None:
        action = select_policy_action(make_analysis())
        assert action.model_tier == "small"
        assert action.inference_params.max_tokens == 256
        assert action.inference_params.temperature == 0.1

    def test_high_complexity_uses_large_budget(self) -> None:
        analysis = make_analysis(
            level=ComplexityLevel.HIGH,
            policy=OptimizationPolicy.CONSERVATIVE,
            task_type=TaskType.REASONING,
        )
        action = select_policy_action(analysis)
        assert action.model_tier == "large"
        assert action.inference_params.max_tokens == 1024
        assert action.enable_thinking is True

    def test_relax_policy_moves_to_more_conservative(self) -> None:
        assert relax_policy(OptimizationPolicy.AGGRESSIVE) == OptimizationPolicy.MODERATE
        assert relax_policy(OptimizationPolicy.MODERATE) == OptimizationPolicy.CONSERVATIVE

    def test_complexity_actions_cover_all_levels(self) -> None:
        assert set(COMPLEXITY_ACTIONS) == set(ComplexityLevel)


class TestQualityScorer:
    def test_token_overlap_similarity(self) -> None:
        score = QualityScorer(use_llm_judge=False).score(
            "What is CFG?",
            "A control flow graph represents execution paths.",
            "A CFG is a graph of program execution paths.",
        )
        assert score.semantic_similarity.score > 0.2
        assert score.semantic_similarity.method == "token_overlap"


class TestOutputComparator:
    def test_compare_reports_efficiency(self) -> None:
        baseline = make_llm_result("Long baseline answer about CFG nodes and edges.")
        optimized = make_llm_result("CFG is a graph of basic blocks.")
        optimized = LLMCallResult(
            model=optimized.model,
            provider=optimized.provider,
            response=optimized.response,
            thinking=optimized.thinking,
            usage=TokenUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
            latency_ms=30.0,
            energy_proxy=0.005,
            energy_measured_j=0.8,
        )

        comparison = OutputComparator(
            scorer=QualityScorer(use_llm_judge=False)
        ).compare(
            "What is CFG?",
            baseline.response,
            optimized.response,
            baseline_llm=baseline,
            optimized_llm=optimized,
        )

        assert comparison.efficiency is not None
        assert comparison.efficiency["total_token_reduction_percent"] == pytest.approx(26.67, abs=0.2)
        assert comparison.efficiency["energy_measured_savings_percent"] == pytest.approx(33.33, abs=0.2)


class TestQualityFallback:
    def test_short_answer_triggers_fallback_decision(self) -> None:
        handler = QualityFallbackHandler()
        decision = handler.assess_completion(
            "What is CFG?",
            "CFG.",
            make_analysis(),
        )
        assert decision.should_fallback is True

    def test_adequate_answer_skips_fallback(self) -> None:
        handler = QualityFallbackHandler()
        decision = handler.assess_completion(
            "What is CFG?",
            "A control flow graph models execution paths using basic blocks and edges.",
            make_analysis(),
        )
        assert decision.should_fallback is False


class TestEnergyMonitor:
    def test_integrates_power_over_duration(self) -> None:
        calibration = EnergyCalibration(
            platform_name="test",
            base_power_watts=8.0,
            cpu_watts_per_percent=0.35,
        )
        measured = MeasuredEnergy(
            joules=(8.0 + 25.0 * 0.35) * 1.0,
            duration_s=1.0,
            average_power_watts=8.0 + 25.0 * 0.35,
            sample_count=3,
            calibration=calibration,
            method="test",
            notes=[],
        )

        assert measured.joules > 0
        assert measured.sample_count == 3
