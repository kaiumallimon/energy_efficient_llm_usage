from __future__ import annotations

import re

from src.analyzer.models import (
    ComplexityLevel,
    ComplexityResult,
    OptimizationPolicy,
    TaskType,
)
from src.generator.models import ModelTier
from src.optimizer.models import OptimizationResult
from src.optimizer.phrases import CONSTRAINT_MARKERS

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

VERBOSITY_APPENDIX = (
    "The user request has been condensed; preserve the original intent and constraints."
)

QUESTION_SPLIT_RE = re.compile(r"\?(?:\s+|$)")


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
    verbosity_score = signals.get("verbosity_score", 0.0)

    if safety_score >= 0.35:
        parts.append(SAFETY_APPENDIX)
        notes.append("Added safety guidance to system prompt.")

    if constraint_score >= 0.3:
        parts.append(CONSTRAINT_APPENDIX)
        notes.append("Added format-constraint preservation guidance.")

    if retrieval_score >= 0.3:
        parts.append(RETRIEVAL_APPENDIX)
        notes.append("Added retrieval-limitation guidance.")

    if verbosity_score >= 0.35 and analysis.policy != OptimizationPolicy.MINIMAL:
        parts.append(VERBOSITY_APPENDIX)
        notes.append("Added intent-preservation note for compressed raw prompt.")

    return " ".join(part for part in parts if part), notes


def extract_constraints(query: str) -> list[str]:
    constraints: list[str] = []
    for pattern in CONSTRAINT_MARKERS:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            start = max(0, match.start() - 40)
            end = min(len(query), match.end() + 80)
            snippet = query[start:end].strip()
            if snippet and snippet not in constraints:
                constraints.append(snippet)
    return constraints


def split_questions(query: str) -> list[str]:
    if "?" not in query:
        return []

    parts = QUESTION_SPLIT_RE.split(query.strip())
    questions: list[str] = []
    for part in parts:
        cleaned = part.strip(" ,;")
        if not cleaned:
            continue
        if not cleaned.endswith("?"):
            cleaned = f"{cleaned}?"
        questions.append(cleaned)
    return questions


def structure_request(query: str, analysis: ComplexityResult) -> tuple[str, list[str]]:
    notes: list[str] = []
    question_count = analysis.signals.get("question_count", 0)
    multi_part_score = analysis.signals.get("multi_part_score", 0.0)

    if question_count > 1 or multi_part_score >= 0.3:
        questions = split_questions(query)
        if len(questions) > 1:
            lines = [f"{index}. {item}" for index, item in enumerate(questions, start=1)]
            notes.append("Structured multi-part request as a numbered list.")
            return "\n".join(lines), notes

    constraints = extract_constraints(query)
    if constraints:
        constraint_block = "\n".join(f"- {item}" for item in constraints)
        notes.append("Preserved explicit constraints in a dedicated section.")
        return f"Request:\n{query}\n\nConstraints:\n{constraint_block}", notes

    return query, notes


def build_user_prompt(
    optimization: OptimizationResult,
    analysis: ComplexityResult,
) -> tuple[str, dict[str, str]]:
    query = optimization.optimized_query
    context = optimization.optimized_context
    sections: dict[str, str] = {"query": query}

    structured_query, structure_notes = structure_request(query, analysis)
    sections["request"] = structured_query

    if context:
        sections["context"] = context
        if structured_query != query:
            sections["query"] = query
            user_prompt = f"Context:\n{context}\n\n{structured_query}"
        else:
            user_prompt = f"Context:\n{context}\n\nRequest:\n{structured_query}"
    elif structured_query != query or structure_notes:
        user_prompt = structured_query
    else:
        user_prompt = query

    return user_prompt, sections


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_full_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
