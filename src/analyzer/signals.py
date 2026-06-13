from __future__ import annotations

import re
from dataclasses import dataclass

from src.analyzer.models import TaskType
from src.optimizer.phrases import ANALYZER_FILLER_PATTERNS
from src.text_utils import WORD_RE
SENTENCE_RE = re.compile(r"[.!?]+")

TASK_PATTERNS: dict[TaskType, tuple[str, ...]] = {
    TaskType.DEFINITION: (
        r"\b(what is|what's|define|definition of|meaning of|stand for|stands for)\b",
        r"\b(in short|briefly|shortly|quick definition)\b",
    ),
    TaskType.CONCEPT_EXPLANATION: (
        r"\b(explain|describe|how does .+ work|what does .+ mean)\b",
        r"\b(concept|theory|principle|overview of)\b",
    ),
    TaskType.EDUCATIONAL: (
        r"\b(teach me|help me learn|learn about|tutorial|walk me through)\b",
        r"\b(for beginners|intro to|introduction to)\b",
    ),
    TaskType.EXAM_HELP: (
        r"\b(exam|quiz|homework|assignment|test prep|study guide)\b",
        r"\b(practice question|sample question|mock test)\b",
    ),
    TaskType.CODING: (
        r"\b(write code|implement|debug|refactor|fix this code|stack trace|compile error)\b",
        r"\b(generate|build|create)\b.*\b(code|script|function|program|builder)\b",
        r"\b(python script|javascript function|typescript class|sql query)\b",
        r"\b(in python|in javascript|in typescript)\b",
        r"\b(code snippet|run this program|syntax error)\b",
    ),
    TaskType.REASONING: (
        r"\b(why|how|explain|analyze|compare|evaluate|prove|derive|step by step)\b",
        r"\b(calculate|compute|show your steps|work through)\b",
        r"\b(trade-?offs?|pros and cons|reasoning|logic|cyclomatic)\b",
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
    r"\b(step by step|chain of thought|think through|reasoning|show your steps)\b",
    r"\b(calculate|compute|cyclomatic)\b",
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
    r"\b(according to|based on the document|from the file|attached|uploaded notes)\b",
    r"\banswer from\b.*\b(notes|document|file|context|upload)\b",
)


@dataclass(frozen=True)
class PromptSignals:
    word_count: int
    sentence_count: int
    question_count: int
    task_type: TaskType
    task_type_scores: dict[str, float]
    reasoning_score: float
    constraint_score: float
    safety_score: float
    multi_part_score: float
    ambiguity_score: float
    retrieval_score: float
    filler_score: float
    repetition_score: float
    verbosity_score: float
    context_word_count: int
    context_ratio: float


def count_pattern_hits(text: str, patterns: tuple[str, ...]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def score_patterns(text: str, patterns: tuple[str, ...], scale: float = 1.0) -> float:
    hits = count_pattern_hits(text, patterns)
    if hits == 0:
        return 0.0
    return min(1.0, hits * scale)


def detect_task_type(text: str) -> tuple[TaskType, dict[str, float]]:
    normalized = _normalize_for_task_detection(text)
    scores: dict[str, float] = {}

    for task_type, patterns in TASK_PATTERNS.items():
        hits = count_pattern_hits(normalized, patterns)
        if hits:
            weight = 0.45 if task_type in {
                TaskType.DEFINITION,
                TaskType.CONCEPT_EXPLANATION,
            } else 0.35
            scores[task_type.value] = min(1.0, hits * weight)

    word_count = len(normalized.split())
    if re.search(r"\d", normalized) and re.search(r"[\+\-\*/=]", normalized):
        scores[TaskType.FACTUAL.value] = max(scores.get(TaskType.FACTUAL.value, 0.0), 0.85)
    elif word_count <= 15 and re.search(
        r"\b(what is|what's|define|meaning of|stand for)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        factual_indicator = re.search(
            r"\b(capital of|who is|when did|where is|how many|how much)\b",
            normalized,
            flags=re.IGNORECASE,
        )
        if not factual_indicator:
            definition_score = 0.55
            if re.search(
                r"\b(shortly|briefly|in short|quick definition)\b",
                normalized,
                flags=re.IGNORECASE,
            ):
                definition_score = 0.9
            elif re.search(r"\b[A-Z]{2,}\b", text):
                definition_score = 0.88
            scores[TaskType.DEFINITION.value] = max(
                scores.get(TaskType.DEFINITION.value, 0.0),
                definition_score,
            )
            if definition_score >= 0.85:
                scores.pop(TaskType.CODING.value, None)

    if word_count <= 20 and re.search(r"\b(explain|describe)\b", normalized, flags=re.IGNORECASE):
        if not re.search(
            r"\b(debug|implement|refactor|fix this code|python|javascript|typescript)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            scores[TaskType.CONCEPT_EXPLANATION.value] = max(
                scores.get(TaskType.CONCEPT_EXPLANATION.value, 0.0),
                0.7,
            )

    if re.search(
        r"\b(debug|implement|refactor|fix|generate|build|create)\b.*\b(code|function|python|javascript|typescript|program|builder)\b",
        normalized,
        flags=re.IGNORECASE,
    ) or re.search(
        r"\b(python|javascript|typescript)\b.*\b(function|script|code|builder)\b",
        normalized,
        flags=re.IGNORECASE,
    ) or re.search(r"\bin python\b", normalized, flags=re.IGNORECASE):
        scores[TaskType.CODING.value] = max(scores.get(TaskType.CODING.value, 0.0), 0.9)
        scores.pop(TaskType.CONCEPT_EXPLANATION.value, None)

    if re.search(
        r"\b(calculate|compute|show your steps|cyclomatic)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        scores[TaskType.REASONING.value] = max(scores.get(TaskType.REASONING.value, 0.0), 0.85)
        scores.pop(TaskType.CONVERSATIONAL.value, None)

    if re.search(
        r"\banswer from\b.*\b(notes|document|file|context|upload)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        scores[TaskType.EXTRACTION.value] = max(scores.get(TaskType.EXTRACTION.value, 0.0), 0.75)
        scores[TaskType.FACTUAL.value] = max(scores.get(TaskType.FACTUAL.value, 0.0), 0.65)

    if not scores:
        if len(normalized.split()) <= 12 and "?" in normalized:
            return TaskType.FACTUAL, {TaskType.FACTUAL.value: 0.4}
        return TaskType.CONVERSATIONAL, {TaskType.CONVERSATIONAL.value: 0.3}

    best = max(scores, key=scores.get)
    return TaskType(best), scores


def _normalize_for_task_detection(text: str) -> str:
    """Strip common openers so task intent is easier to detect in raw prompts."""
    updated = text.strip().lower()
    openers = (
        r"^(could you please|can you please|could you|can you|please|hey|hi|hello|so)\s+",
        r"^(i want to know|i would like to know|tell me|help me)\s+",
    )
    for pattern in openers:
        updated = re.sub(pattern, "", updated, flags=re.IGNORECASE)
    return updated


def compute_repetition_score(text: str) -> float:
    sentences = [part.strip().lower() for part in re.split(r"[.!?]+", text) if part.strip()]
    if len(sentences) <= 1:
        words = [w.lower() for w in WORD_RE.findall(text)]
        if len(words) < 6:
            return 0.0
        half = len(words) // 2
        left = " ".join(words[:half])
        right = " ".join(words[half:])
        if left and left == right:
            return 0.8
        return 0.0

    unique = set(sentences)
    duplicate_ratio = 1.0 - (len(unique) / len(sentences))
    return min(1.0, duplicate_ratio)


def compute_filler_score(text: str) -> float:
    return score_patterns(text, ANALYZER_FILLER_PATTERNS, scale=0.12)


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

    filler_score = compute_filler_score(query)
    repetition_score = compute_repetition_score(query)
    verbosity_score = min(1.0, filler_score * 0.7 + repetition_score * 0.8)

    return PromptSignals(
        word_count=word_count,
        sentence_count=sentence_count,
        question_count=question_count,
        task_type=task_type,
        task_type_scores=task_type_scores,
        reasoning_score=reasoning_score,
        constraint_score=constraint_score,
        safety_score=safety_score,
        multi_part_score=multi_part_score,
        ambiguity_score=ambiguity_score,
        retrieval_score=retrieval_score,
        filler_score=filler_score,
        repetition_score=repetition_score,
        verbosity_score=verbosity_score,
        context_word_count=context_words,
        context_ratio=round(context_ratio, 3),
    )
