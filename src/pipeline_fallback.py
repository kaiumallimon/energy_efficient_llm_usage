"""Quality fallback when optimized output is too short or low confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.analyzer.models import ComplexityLevel, ComplexityResult, OptimizationPolicy, TaskType
from src.evaluation.quality_scorer import QualityScorer
from src.generator.generator import AdaptivePromptGenerator
from src.generator.models import GeneratedPrompt
from src.generator.policy_matrix import relax_policy, relaxed_policy_action
from src.llm.models import LLMCallResult


MIN_COMPLETION_CHARS: dict[TaskType, int] = {
    TaskType.DEFINITION: 40,
    TaskType.FACTUAL: 20,
    TaskType.CONCEPT_EXPLANATION: 60,
    TaskType.REASONING: 80,
    TaskType.CODING: 100,
    TaskType.SUMMARIZATION: 80,
    TaskType.EXTRACTION: 30,
}

DEFAULT_MIN_COMPLETION_CHARS = 25
MIN_SEMANTIC_SIMILARITY = 0.35
MIN_RUBRIC_OVERALL = 2.8


@dataclass(frozen=True)
class FallbackDecision:
    should_fallback: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_fallback": self.should_fallback,
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class FallbackResult:
    used_fallback: bool
    original_completion: str | None
    final_completion: str | None
    original_policy: str
    fallback_policy: str | None = None
    fallback_generation: GeneratedPrompt | None = None
    fallback_llm: LLMCallResult | None = None
    decision: FallbackDecision | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_fallback": self.used_fallback,
            "original_completion": self.original_completion,
            "final_completion": self.final_completion,
            "original_policy": self.original_policy,
            "fallback_policy": self.fallback_policy,
            "decision": self.decision.to_dict() if self.decision is not None else None,
            "notes": self.notes,
        }


class QualityFallbackHandler:
    """Reruns generation with a less aggressive policy when quality is too low."""

    def __init__(
        self,
        generator: AdaptivePromptGenerator | None = None,
        scorer: QualityScorer | None = None,
    ) -> None:
        self.generator = generator or AdaptivePromptGenerator()
        self.scorer = scorer

    def assess_completion(
        self,
        query: str,
        completion: str,
        analysis: ComplexityResult,
        *,
        baseline_completion: str | None = None,
    ) -> FallbackDecision:
        reasons: list[str] = []
        stripped = completion.strip()

        if not stripped:
            reasons.append("Completion is empty.")
            return FallbackDecision(should_fallback=True, reasons=reasons)

        min_chars = MIN_COMPLETION_CHARS.get(analysis.task_type, DEFAULT_MIN_COMPLETION_CHARS)
        if len(stripped) < min_chars:
            reasons.append(
                f"Completion length {len(stripped)} below minimum {min_chars} for "
                f"{analysis.task_type.value}."
            )

        if analysis.policy == OptimizationPolicy.AGGRESSIVE and len(stripped.split()) < 8:
            reasons.append("Aggressive policy produced an unusually short answer.")

        if baseline_completion and self.scorer is not None:
            quality = self.scorer.score(query, baseline_completion, stripped)
            if quality.semantic_similarity.score < MIN_SEMANTIC_SIMILARITY:
                reasons.append(
                    "Semantic similarity to baseline below "
                    f"{MIN_SEMANTIC_SIMILARITY:.2f}."
                )
            if quality.rubric is not None and quality.rubric.overall < MIN_RUBRIC_OVERALL:
                reasons.append(
                    f"Rubric overall {quality.rubric.overall:.1f} below {MIN_RUBRIC_OVERALL:.1f}."
                )

        if analysis.policy == OptimizationPolicy.MINIMAL:
            return FallbackDecision(should_fallback=False, reasons=["Minimal policy already active."])

        return FallbackDecision(should_fallback=bool(reasons), reasons=reasons)

    def maybe_rerun(
        self,
        query: str,
        analysis: ComplexityResult,
        optimization,
        decomposition,
        llm_result: LLMCallResult,
        *,
        baseline_completion: str | None = None,
        llm_client: Any | None = None,
        think: bool | None = None,
    ) -> FallbackResult:
        decision = self.assess_completion(
            query,
            llm_result.response,
            analysis,
            baseline_completion=baseline_completion,
        )

        if not decision.should_fallback or llm_client is None:
            return FallbackResult(
                used_fallback=False,
                original_completion=llm_result.response,
                final_completion=llm_result.response,
                original_policy=analysis.policy.value,
                decision=decision,
                notes=["No fallback rerun required."],
            )

        if analysis.policy == OptimizationPolicy.MINIMAL:
            return FallbackResult(
                used_fallback=False,
                original_completion=llm_result.response,
                final_completion=llm_result.response,
                original_policy=analysis.policy.value,
                decision=decision,
                notes=["Fallback skipped because policy is already minimal."],
            )

        relaxed_policy = relax_policy(analysis.policy)
        relaxed_analysis = ComplexityResult(
            level=analysis.level,
            score=analysis.score,
            task_type=analysis.task_type,
            policy=relaxed_policy,
            confidence=analysis.confidence,
            signals=analysis.signals,
            rationale=[*analysis.rationale, "Quality fallback relaxed optimization policy."],
            intent_source=analysis.intent_source,
        )
        action = relaxed_policy_action(analysis)
        generation = self.generator.generate(relaxed_analysis, optimization, decomposition)
        generation = GeneratedPrompt(
            system_prompt=generation.system_prompt,
            user_prompt=generation.user_prompt,
            context=generation.context,
            messages=generation.messages,
            full_prompt=generation.full_prompt,
            model_tier=action.model_tier,
            template_id=generation.template_id,
            task_type=generation.task_type,
            complexity_level=generation.complexity_level,
            policy=relaxed_policy,
            inference_params=action.inference_params,
            inference_options=action.inference_params.to_options(),
            sections=generation.sections,
            notes=[*generation.notes, action.description],
        )

        fallback_think = think
        if think is None and action.enable_thinking:
            fallback_think = True

        fallback_llm = llm_client.call(generation, think=fallback_think)
        notes = [
            f"Fallback rerun triggered: {', '.join(decision.reasons)}",
            f"Relaxed policy from {analysis.policy.value} to {relaxed_policy.value}.",
        ]

        return FallbackResult(
            used_fallback=True,
            original_completion=llm_result.response,
            final_completion=fallback_llm.response,
            original_policy=analysis.policy.value,
            fallback_policy=relaxed_policy.value,
            fallback_generation=generation,
            fallback_llm=fallback_llm,
            decision=decision,
            notes=notes,
        )
