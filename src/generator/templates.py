from __future__ import annotations

from src.analyzer.models import (
    ComplexityLevel,
    ComplexityResult,
    OptimizationPolicy,
    TaskType,
)
from src.generator.models import ModelTier
from src.optimizer.models import OptimizationResult

TASK_INSTRUCTIONS: dict[TaskType, str] = {
    TaskType.FACTUAL: "Answer factual questions directly and accurately.",
    TaskType.CONVERSATIONAL: "Respond naturally and helpfully.",
    TaskType.CREATIVE: "Follow the user's creative brief while staying concise.",
    TaskType.REASONING: "Explain reasoning clearly; use structure for comparisons.",
    TaskType.CODING: "Provide correct, runnable code and brief explanations when asked.",
    TaskType.SUMMARIZATION: "Summarize faithfully without adding unsupported details.",
    TaskType.EXTRACTION: "Extract only what is requested and honor any output format.",
    TaskType.UNKNOWN: "Follow the user's instructions precisely.",
}

POLICY_STYLE: dict[OptimizationPolicy, str] = {
    OptimizationPolicy.AGGRESSIVE: "Be concise.",
    OptimizationPolicy.MODERATE: "Be clear and efficient.",
    OptimizationPolicy.CONSERVATIVE: "Preserve nuance and constraints.",
    OptimizationPolicy.MINIMAL: "Prioritize safety and fidelity over brevity.",
}

LEVEL_GUIDANCE: dict[ComplexityLevel, str] = {
    ComplexityLevel.LOW: "",
    ComplexityLevel.MEDIUM: "Address each part of the request.",
    ComplexityLevel.HIGH: "Work step by step when the task is multi-part.",
    ComplexityLevel.CRITICAL: "Flag uncertainty and avoid overconfident claims.",
}

SAFETY_APPENDIX = (
    "Sensitive topic detected: do not provide definitive medical, legal, or "
    "financial advice; recommend qualified professionals when appropriate."
)

CONSTRAINT_APPENDIX = (
    "Preserve all output-format constraints from the user prompt exactly."
)

RETRIEVAL_APPENDIX = (
    "If external or up-to-date information is required, state limitations clearly."
)


def select_model_tier(analysis: ComplexityResult) -> str:
    if analysis.level == ComplexityLevel.CRITICAL or analysis.policy == OptimizationPolicy.MINIMAL:
        return ModelTier.LARGE.value

    if analysis.task_type == TaskType.CODING or analysis.level == ComplexityLevel.HIGH:
        return ModelTier.LARGE.value

    if analysis.level == ComplexityLevel.LOW and analysis.policy == OptimizationPolicy.AGGRESSIVE:
        return ModelTier.SMALL.value

    return ModelTier.MEDIUM.value


def build_template_id(analysis: ComplexityResult) -> str:
    return f"{analysis.task_type.value}_{analysis.policy.value}"


def build_system_prompt(analysis: ComplexityResult) -> tuple[str, list[str]]:
    notes: list[str] = []
    parts = [
        TASK_INSTRUCTIONS[analysis.task_type],
        POLICY_STYLE[analysis.policy],
    ]

    level_hint = LEVEL_GUIDANCE[analysis.level]
    if level_hint:
        parts.append(level_hint)

    signals = analysis.signals
    safety_score = signals.get("safety_score", 0.0)
    constraint_score = signals.get("constraint_score", 0.0)
    retrieval_score = signals.get("retrieval_score", 0.0)

    if safety_score >= 0.35:
        parts.append(SAFETY_APPENDIX)
        notes.append("Added safety guidance to system prompt.")

    if constraint_score >= 0.3:
        parts.append(CONSTRAINT_APPENDIX)
        notes.append("Added format-constraint preservation guidance.")

    if retrieval_score >= 0.3:
        parts.append(RETRIEVAL_APPENDIX)
        notes.append("Added retrieval-limitation guidance.")

    return " ".join(part for part in parts if part), notes


def build_user_prompt(optimization: OptimizationResult) -> tuple[str, dict[str, str]]:
    query = optimization.optimized_query
    context = optimization.optimized_context
    sections: dict[str, str] = {"query": query}

    if not context:
        return query, sections

    sections["context"] = context
    user_prompt = f"Context:\n{context}\n\nRequest:\n{query}"
    return user_prompt, sections


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_full_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
