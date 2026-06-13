"""Explicit complexity-to-action policy matrix for adaptive generation."""

from __future__ import annotations

from dataclasses import dataclass

from src.analyzer.models import ComplexityLevel, ComplexityResult, OptimizationPolicy, TaskType
from src.generator.models import InferenceParams, ModelTier


@dataclass(frozen=True)
class PolicyAction:
    """Generation actions selected from query complexity."""

    template_policy: OptimizationPolicy
    model_tier: str
    inference_params: InferenceParams
    enable_thinking: bool
    compression_level: str
    description: str

    def to_dict(self) -> dict[str, object]:
        return {
            "template_policy": self.template_policy.value,
            "model_tier": self.model_tier,
            "inference_params": {
                "max_tokens": self.inference_params.max_tokens,
                "temperature": self.inference_params.temperature,
                "top_p": self.inference_params.top_p,
            },
            "enable_thinking": self.enable_thinking,
            "compression_level": self.compression_level,
            "description": self.description,
        }


# Base actions keyed by complexity level. Task-specific overrides are applied below.
COMPLEXITY_ACTIONS: dict[ComplexityLevel, PolicyAction] = {
    ComplexityLevel.LOW: PolicyAction(
        template_policy=OptimizationPolicy.AGGRESSIVE,
        model_tier=ModelTier.SMALL.value,
        inference_params=InferenceParams(max_tokens=256, temperature=0.1, top_p=0.85),
        enable_thinking=False,
        compression_level="aggressive",
        description="Aggressive template, low max_tokens, low temperature.",
    ),
    ComplexityLevel.MEDIUM: PolicyAction(
        template_policy=OptimizationPolicy.MODERATE,
        model_tier=ModelTier.MEDIUM.value,
        inference_params=InferenceParams(max_tokens=512, temperature=0.15, top_p=0.9),
        enable_thinking=False,
        compression_level="moderate",
        description="Moderate template and token budget.",
    ),
    ComplexityLevel.HIGH: PolicyAction(
        template_policy=OptimizationPolicy.CONSERVATIVE,
        model_tier=ModelTier.LARGE.value,
        inference_params=InferenceParams(max_tokens=1024, temperature=0.15, top_p=0.9),
        enable_thinking=True,
        compression_level="minimal",
        description="Minimal compression, higher token budget, optional reasoning.",
    ),
    ComplexityLevel.CRITICAL: PolicyAction(
        template_policy=OptimizationPolicy.MINIMAL,
        model_tier=ModelTier.LARGE.value,
        inference_params=InferenceParams(max_tokens=768, temperature=0.2, top_p=0.9),
        enable_thinking=False,
        compression_level="none",
        description="Safety-first: preserve fidelity over brevity.",
    ),
}

POLICY_RELAXATION_ORDER: tuple[OptimizationPolicy, ...] = (
    OptimizationPolicy.AGGRESSIVE,
    OptimizationPolicy.MODERATE,
    OptimizationPolicy.CONSERVATIVE,
    OptimizationPolicy.MINIMAL,
)


def relax_policy(policy: OptimizationPolicy) -> OptimizationPolicy:
    """Move to a less aggressive (more conservative) optimization policy."""
    try:
        index = POLICY_RELAXATION_ORDER.index(policy)
    except ValueError:
        return OptimizationPolicy.MINIMAL
    if index >= len(POLICY_RELAXATION_ORDER) - 1:
        return OptimizationPolicy.MINIMAL
    return POLICY_RELAXATION_ORDER[index + 1]


def select_policy_action(analysis: ComplexityResult) -> PolicyAction:
    """Resolve the effective policy action for a classified query."""
    base = COMPLEXITY_ACTIONS[analysis.level]
    effective_policy = analysis.policy

    if analysis.level == ComplexityLevel.CRITICAL or effective_policy == OptimizationPolicy.MINIMAL:
        return COMPLEXITY_ACTIONS[ComplexityLevel.CRITICAL]

    if analysis.task_type == TaskType.CODING:
        return PolicyAction(
            template_policy=effective_policy,
            model_tier=ModelTier.LARGE.value,
            inference_params=InferenceParams(max_tokens=1024, temperature=0.15, top_p=0.9),
            enable_thinking=analysis.level == ComplexityLevel.HIGH,
            compression_level="minimal" if analysis.level == ComplexityLevel.HIGH else "moderate",
            description="Coding tasks use a large model tier with a higher token budget.",
        )

    if (
        analysis.level == ComplexityLevel.LOW
        and effective_policy == OptimizationPolicy.AGGRESSIVE
        and analysis.task_type
        in {TaskType.DEFINITION, TaskType.FACTUAL, TaskType.CONCEPT_EXPLANATION}
    ):
        return PolicyAction(
            template_policy=effective_policy,
            model_tier=ModelTier.SMALL.value,
            inference_params=InferenceParams(max_tokens=256, temperature=0.1, top_p=0.85),
            enable_thinking=False,
            compression_level="aggressive",
            description="Simple definition/factual queries: concise template and tight budget.",
        )

    if effective_policy != base.template_policy:
        return PolicyAction(
            template_policy=effective_policy,
            model_tier=base.model_tier,
            inference_params=base.inference_params,
            enable_thinking=base.enable_thinking,
            compression_level=base.compression_level,
            description=base.description,
        )

    return base


def relaxed_policy_action(analysis: ComplexityResult) -> PolicyAction:
    """Return a less aggressive action for quality fallback reruns."""
    relaxed_policy = relax_policy(analysis.policy)
    relaxed_analysis = ComplexityResult(
        level=analysis.level,
        score=analysis.score,
        task_type=analysis.task_type,
        policy=relaxed_policy,
        confidence=analysis.confidence,
        signals=analysis.signals,
        rationale=[*analysis.rationale, f"Fallback relaxed policy to {relaxed_policy.value}."],
        intent_source=analysis.intent_source,
    )
    action = select_policy_action(relaxed_analysis)

    bumped_tokens = min(2048, int(action.inference_params.max_tokens * 1.75))
    bumped_params = InferenceParams(
        max_tokens=bumped_tokens,
        temperature=min(0.25, action.inference_params.temperature + 0.05),
        top_p=action.inference_params.top_p,
    )
    return PolicyAction(
        template_policy=relaxed_policy,
        model_tier=ModelTier.LARGE.value if relaxed_policy != OptimizationPolicy.AGGRESSIVE else action.model_tier,
        inference_params=bumped_params,
        enable_thinking=action.enable_thinking or analysis.level == ComplexityLevel.HIGH,
        compression_level=action.compression_level,
        description=f"Fallback policy: {relaxed_policy.value} with expanded token budget.",
    )
