from __future__ import annotations

import re
from dataclasses import dataclass

from src.analyzer.models import TaskType

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+")

TASK_PATTERNS: dict[TaskType, tuple[str, ...]] = {
    TaskType.CODING: (
        r"\b(code|function|class|debug|implement|refactor|api|sql|python|javascript|typescript|regex)\b",
        r"\b(bug|error|stack trace|compile|syntax)\b",
    ),
    TaskType.REASONING: (
        r"\b(why|how|explain|analyze|compare|evaluate|prove|derive|step by step)\b",
        r"\b(trade-?offs?|pros and cons|reasoning|logic)\b",
    ),
    TaskType.SUMMARIZATION: (
        r"\b(summarize|summary|tl;dr|condense|brief overview)\b",
        r"\b(key points|main ideas|in short)\b",
    ),
    TaskType.EXTRACTION: (
        r"\b(extract|parse|find all|list all|identify|pull out)\b",
        r"\b(json|csv|table|schema|fields?)\b",
    ),
    TaskType.CREATIVE: (
        r"\b(write|story|poem|creative|brainstorm|ideas for|marketing copy)\b",
        r"\b(tagline|slogan|script|dialogue)\b",
    ),
    TaskType.FACTUAL: (
        r"\b(what is|who is|when did|where is|define|capital of)\b",
        r"\b(how many|how much)\b",
    ),
}

REASONING_PATTERNS = (
    r"\b(why|how|explain|analyze|compare|evaluate|prove|derive)\b",
    r"\b(step by step|chain of thought|think through|reasoning)\b",
    r"\b(if .+ then|what would happen if)\b",
)

CONSTRAINT_PATTERNS = (
    r"\b(json|yaml|xml|markdown|bullet points?|numbered list)\b",
    r"\b(exactly \d+ words|word limit|must include|do not|don't|never)\b",
    r"\b(format:|output format|respond with|return only)\b",
    r"\b(cite sources|with citations|verbatim)\b",
)

SAFETY_PATTERNS = (
    r"\b(medical|diagnos|prescri|symptom|dosage)\b",
    r"\b(legal|lawsuit|contract|liability|compliance)\b",
    r"\b(financial advice|investment|tax advice|credit score)\b",
    r"\b(pii|personal data|ssn|password|api key|secret)\b",
)

MULTI_PART_PATTERNS = (
    r"\b(and also|as well as|in addition|follow(?:ing)? up)\b",
    r"\b(first,.+second|1[\).]|2[\).]|3[\).])\b",
)

AMBIGUITY_PATTERNS = (
    r"\b(something|stuff|things|maybe|probably|somehow)\b",
    r"\b(it|this|that|they)\b",
)

RETRIEVAL_PATTERNS = (
    r"\b(search|look up|find sources|latest|current|today|recent news)\b",
    r"\b(according to|based on the document|from the file|attached)\b",
)


@dataclass(frozen=True)
class PromptSignals:
    word_count: int
    sentence_count: int
    estimated_tokens: int
    question_count: int
    task_type: TaskType
    task_type_scores: dict[str, float]
    reasoning_score: float
    constraint_score: float
    safety_score: float
    multi_part_score: float
    ambiguity_score: float
    retrieval_score: float
    context_word_count: int
    context_ratio: float


def estimate_tokens(text: str) -> int:
    words = WORD_RE.findall(text)
    return max(1, int(len(words) * 1.3))


def count_pattern_hits(text: str, patterns: tuple[str, ...]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def score_patterns(text: str, patterns: tuple[str, ...], scale: float = 1.0) -> float:
    hits = count_pattern_hits(text, patterns)
    if hits == 0:
        return 0.0
    return min(1.0, hits * scale)


def detect_task_type(text: str) -> tuple[TaskType, dict[str, float]]:
    scores: dict[str, float] = {}

    for task_type, patterns in TASK_PATTERNS.items():
        hits = count_pattern_hits(text, patterns)
        if hits:
            scores[task_type.value] = min(1.0, hits * 0.35)

    if not scores:
        if len(text.split()) <= 12 and "?" in text:
            return TaskType.FACTUAL, {TaskType.FACTUAL.value: 0.4}
        return TaskType.CONVERSATIONAL, {TaskType.CONVERSATIONAL.value: 0.3}

    best = max(scores, key=scores.get)
    return TaskType(best), scores


def extract_signals(query: str, context: str | None = None) -> PromptSignals:
    query = query.strip()
    context = (context or "").strip()
    combined = f"{query}\n{context}".strip() if context else query

    words = WORD_RE.findall(combined)
    word_count = len(words)
    sentence_count = max(1, len(SENTENCE_RE.findall(combined)))
    question_count = combined.count("?")

    context_words = len(WORD_RE.findall(context)) if context else 0
    context_ratio = context_words / word_count if word_count else 0.0

    task_type, task_type_scores = detect_task_type(combined)

    reasoning_score = score_patterns(combined, REASONING_PATTERNS, scale=0.25)
    constraint_score = score_patterns(combined, CONSTRAINT_PATTERNS, scale=0.2)
    safety_score = score_patterns(combined, SAFETY_PATTERNS, scale=0.35)
    multi_part_score = score_patterns(combined, MULTI_PART_PATTERNS, scale=0.3)
    ambiguity_score = score_patterns(combined, AMBIGUITY_PATTERNS, scale=0.15)
    retrieval_score = score_patterns(combined, RETRIEVAL_PATTERNS, scale=0.25)

    if question_count > 1:
        multi_part_score = min(1.0, multi_part_score + 0.2 * (question_count - 1))

    if word_count > 120:
        reasoning_score = min(1.0, reasoning_score + 0.15)
    if word_count > 300:
        reasoning_score = min(1.0, reasoning_score + 0.2)

    return PromptSignals(
        word_count=word_count,
        sentence_count=sentence_count,
        estimated_tokens=estimate_tokens(combined),
        question_count=question_count,
        task_type=task_type,
        task_type_scores=task_type_scores,
        reasoning_score=reasoning_score,
        constraint_score=constraint_score,
        safety_score=safety_score,
        multi_part_score=multi_part_score,
        ambiguity_score=ambiguity_score,
        retrieval_score=retrieval_score,
        context_word_count=context_words,
        context_ratio=round(context_ratio, 3),
    )
