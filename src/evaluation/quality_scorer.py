"""Automatic output quality scoring for baseline vs optimized comparisons."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator for LLM answers. "
    "Return ONLY valid JSON with keys: accuracy, completeness, brevity, usefulness, "
    "overall, confidence, rationale. "
    "Each score is 1-5 (higher is better). "
    "Brevity rewards concise answers that still answer the question."
)

JUDGE_USER_TEMPLATE = """Evaluate the candidate answer against the reference answer for this query.

Query:
{query}

Reference answer (baseline):
{baseline}

Candidate answer (optimized):
{optimized}

Score each rubric dimension from 1 (poor) to 5 (excellent).
"""


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _token_overlap_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload

    raise ValueError("No JSON object found in judge response.")


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class RubricScores:
    accuracy: float
    completeness: float
    brevity: float
    usefulness: float
    overall: float
    confidence: float
    rationale: str
    source: str = "llm_judge"

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 2),
            "completeness": round(self.completeness, 2),
            "brevity": round(self.brevity, 2),
            "usefulness": round(self.usefulness, 2),
            "overall": round(self.overall, 2),
            "confidence": round(self.confidence, 2),
            "rationale": self.rationale,
            "source": self.source,
        }


@dataclass(frozen=True)
class SemanticSimilarity:
    method: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class QualityScore:
    semantic_similarity: SemanticSimilarity
    rubric: RubricScores | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if self.rubric is not None:
            return (self.semantic_similarity.score * 0.35) + (self.rubric.overall / 5.0 * 0.65)
        return self.semantic_similarity.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_similarity": self.semantic_similarity.to_dict(),
            "rubric": self.rubric.to_dict() if self.rubric is not None else None,
            "overall_score": round(self.overall_score, 4),
            "notes": self.notes,
        }


class QualityScorer:
    """Scores optimized output against a baseline using semantic and rubric metrics."""

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        judge_client: Any | None = None,
        *,
        use_llm_judge: bool = True,
    ) -> None:
        self.embedding_client = embedding_client
        self.judge_client = judge_client
        self.use_llm_judge = use_llm_judge

    def score(
        self,
        query: str,
        baseline: str,
        optimized: str,
    ) -> QualityScore:
        notes: list[str] = []
        semantic = self._semantic_similarity(baseline, optimized, notes)
        rubric = None

        if self.use_llm_judge and self.judge_client is not None:
            rubric = self._llm_judge(query, baseline, optimized, notes)
        elif self.use_llm_judge:
            notes.append("LLM judge skipped: no judge client configured.")

        return QualityScore(
            semantic_similarity=semantic,
            rubric=rubric,
            notes=notes,
        )

    def _semantic_similarity(
        self,
        baseline: str,
        optimized: str,
        notes: list[str],
    ) -> SemanticSimilarity:
        if self.embedding_client is not None:
            try:
                baseline_vec = self.embedding_client.embed(baseline)
                optimized_vec = self.embedding_client.embed(optimized)
                score = _cosine_similarity(baseline_vec, optimized_vec)
                notes.append("Semantic similarity computed with sentence embeddings.")
                return SemanticSimilarity(method="sentence_embedding", score=score)
            except Exception as exc:
                notes.append(f"Embedding similarity failed: {exc}")

        score = _token_overlap_similarity(baseline, optimized)
        notes.append("Semantic similarity fell back to token overlap.")
        return SemanticSimilarity(method="token_overlap", score=score)

    def _llm_judge(
        self,
        query: str,
        baseline: str,
        optimized: str,
        notes: list[str],
    ) -> RubricScores:
        prompt = JUDGE_USER_TEMPLATE.format(
            query=query.strip(),
            baseline=baseline.strip(),
            optimized=optimized.strip(),
        )

        try:
            raw = self.judge_client.generate_text(
                prompt,
                system=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=384,
            )
            payload = _extract_json_object(raw)
        except Exception as exc:
            notes.append(f"LLM judge failed: {exc}")
            return RubricScores(
                accuracy=3.0,
                completeness=3.0,
                brevity=3.0,
                usefulness=3.0,
                overall=3.0,
                confidence=0.2,
                rationale="Judge unavailable; neutral scores applied.",
                source="fallback",
            )

        def _score(name: str, default: float = 3.0) -> float:
            value = payload.get(name, default)
            try:
                return max(1.0, min(5.0, float(value)))
            except (TypeError, ValueError):
                return default

        rubric = RubricScores(
            accuracy=_score("accuracy"),
            completeness=_score("completeness"),
            brevity=_score("brevity"),
            usefulness=_score("usefulness"),
            overall=_score("overall"),
            confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.7)))),
            rationale=str(payload.get("rationale", "")).strip(),
        )
        notes.append("LLM-as-a-judge rubric applied.")
        return rubric
