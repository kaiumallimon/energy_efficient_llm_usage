from __future__ import annotations

from dataclasses import dataclass

from src.analyzer.models import (
    ComplexityLevel,
    ComplexityResult,
    OptimizationPolicy,
    TaskType,
)
from src.analyzer.signals import PromptSignals, extract_signals


@dataclass
class ComplexityThresholds:
    low_max: float = 35.0
    medium_max: float = 60.0
    high_max: float = 80.0


TASK_COMPLEXITY_WEIGHT: dict[TaskType, float] = {
    TaskType.FACTUAL: 8.0,
    TaskType.CONVERSATIONAL: 10.0,
    TaskType.CREATIVE: 18.0,
    TaskType.SUMMARIZATION: 22.0,
    TaskType.EXTRACTION: 24.0,
    TaskType.REASONING: 30.0,
    TaskType.CODING: 32.0,
    TaskType.UNKNOWN: 15.0,
}


class ComplexityAnalyzer:
    """Heuristic complexity classifier for user prompts."""

    def __init__(self, thresholds: ComplexityThresholds | None = None) -> None:
        self.thresholds = thresholds or ComplexityThresholds()

    def analyze(self, query: str, context: str | None = None) -> ComplexityResult:
        signals = extract_signals(query, context)
        score = self._score(signals)
        level = self._level(score, signals)
        policy = self._policy(level, signals)
        confidence = self._confidence(signals, score)
        rationale = self._rationale(signals, score, level)

        return ComplexityResult(
            level=level,
            score=round(score, 2),
            task_type=signals.task_type,
            policy=policy,
            confidence=round(confidence, 2),
            signals=self._public_signals(signals),
            rationale=rationale,
        )

    def _score(self, signals: PromptSignals) -> float:
        score = TASK_COMPLEXITY_WEIGHT[signals.task_type]

        score += min(20.0, signals.word_count / 19.0)
        score += signals.reasoning_score * 18.0
        score += signals.constraint_score * 12.0
        score += signals.multi_part_score * 14.0
        score += signals.retrieval_score * 10.0
        score += signals.ambiguity_score * 8.0
        score += signals.safety_score * 22.0

        if signals.context_word_count > 200:
            score += 12.0
        elif signals.context_word_count > 80:
            score += 6.0

        if signals.context_ratio > 0.7 and signals.context_word_count > 50:
            score += 5.0

        if signals.task_type == TaskType.FACTUAL and signals.word_count <= 12:
            score -= 8.0

        if signals.constraint_score >= 0.3:
            score += 10.0

        if (
            signals.task_type == TaskType.CONVERSATIONAL
            and signals.word_count <= 8
            and signals.reasoning_score < 0.2
        ):
            score -= 6.0

        if signals.verbosity_score >= 0.35 and signals.reasoning_score < 0.35:
            score -= 4.0

        return max(0.0, min(100.0, score))

    def _level(self, score: float, signals: PromptSignals) -> ComplexityLevel:
        if signals.safety_score >= 0.35:
            return ComplexityLevel.CRITICAL

        if score <= self.thresholds.low_max:
            level = ComplexityLevel.LOW
        elif score <= self.thresholds.medium_max:
            level = ComplexityLevel.MEDIUM
        elif score <= self.thresholds.high_max:
            level = ComplexityLevel.HIGH
        else:
            level = ComplexityLevel.CRITICAL

        if signals.constraint_score >= 0.3 and level == ComplexityLevel.LOW:
            return ComplexityLevel.MEDIUM

        return level

    def _policy(self, level: ComplexityLevel, signals: PromptSignals) -> OptimizationPolicy:
        if level == ComplexityLevel.CRITICAL or signals.safety_score >= 0.35:
            return OptimizationPolicy.MINIMAL
        if level == ComplexityLevel.LOW:
            policy = OptimizationPolicy.AGGRESSIVE
        elif level == ComplexityLevel.MEDIUM:
            policy = OptimizationPolicy.MODERATE
        else:
            policy = OptimizationPolicy.CONSERVATIVE

        if signals.constraint_score >= 0.3 and policy == OptimizationPolicy.AGGRESSIVE:
            return OptimizationPolicy.MODERATE

        return policy

    def _confidence(self, signals: PromptSignals, score: float) -> float:
        confidence = 0.55

        if signals.task_type_scores:
            top = max(signals.task_type_scores.values())
            confidence += min(0.2, top * 0.2)

        if signals.word_count >= 20:
            confidence += 0.1
        if signals.word_count >= 60:
            confidence += 0.05

        if score <= self.thresholds.low_max or score >= self.thresholds.high_max:
            confidence += 0.1

        return min(0.95, confidence)

    def _rationale(
        self,
        signals: PromptSignals,
        score: float,
        level: ComplexityLevel,
    ) -> list[str]:
        rationale = [f"Detected task type: {signals.task_type.value}."]

        if signals.word_count <= 12:
            rationale.append("Prompt is short and likely direct.")
        elif signals.word_count >= 120:
            rationale.append("Prompt is long, which often needs richer context handling.")

        if signals.reasoning_score >= 0.4:
            rationale.append("Reasoning-oriented language detected.")
        if signals.constraint_score >= 0.3:
            rationale.append("Output constraints detected; preserve formatting rules.")
        if signals.safety_score >= 0.35:
            rationale.append("Sensitive domain detected; avoid aggressive compression.")
        if signals.multi_part_score >= 0.3:
            rationale.append("Multi-part request detected.")
        if signals.context_word_count >= 80:
            rationale.append("Large external context increases optimization risk.")
        if signals.retrieval_score >= 0.3:
            rationale.append("Retrieval or freshness requirements detected.")
        if signals.verbosity_score >= 0.35:
            rationale.append("High filler or repetition detected; safe to compress wording.")
        if signals.repetition_score >= 0.4:
            rationale.append("Repeated phrasing detected in the user prompt.")

        rationale.append(f"Composite complexity score: {score:.1f} -> {level.value}.")
        return rationale

    def _public_signals(self, signals: PromptSignals) -> dict:
        return {
            "word_count": signals.word_count,
            "sentence_count": signals.sentence_count,
            "question_count": signals.question_count,
            "task_type_scores": signals.task_type_scores,
            "reasoning_score": round(signals.reasoning_score, 3),
            "constraint_score": round(signals.constraint_score, 3),
            "safety_score": round(signals.safety_score, 3),
            "multi_part_score": round(signals.multi_part_score, 3),
            "ambiguity_score": round(signals.ambiguity_score, 3),
            "retrieval_score": round(signals.retrieval_score, 3),
            "filler_score": round(signals.filler_score, 3),
            "repetition_score": round(signals.repetition_score, 3),
            "verbosity_score": round(signals.verbosity_score, 3),
            "context_word_count": signals.context_word_count,
            "context_ratio": signals.context_ratio,
        }
