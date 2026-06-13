from __future__ import annotations

from src.llm.ollama_client import OllamaClient, OllamaError
from src.optimizer.guardrails import intent_preserved
from src.validator.models import ValidationResult


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


class QualityValidator:
    """Rejects semantic compression when embedding similarity is too low."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        use_ollama: bool = True,
        similarity_threshold: float | None = None,
    ) -> None:
        self.client = client
        self.use_ollama = use_ollama
        self.similarity_threshold = similarity_threshold

    def validate(self, original_query: str, candidate_query: str) -> ValidationResult:
        original = original_query.strip()
        candidate = candidate_query.strip()
        threshold = self._threshold()

        if candidate == original:
            return ValidationResult(
                passed=True,
                similarity=1.0,
                threshold=threshold,
                original_query=original,
                candidate_query=candidate,
                accepted_query=original,
                source="unchanged",
                notes=["Candidate matches original query."],
            )

        if not intent_preserved(original, candidate):
            return ValidationResult(
                passed=False,
                similarity=0.0,
                threshold=threshold,
                original_query=original,
                candidate_query=candidate,
                accepted_query=original,
                source="guardrail",
                notes=["Rejected: substantive content words were lost."],
            )

        similarity = 1.0
        source = "heuristic"
        notes: list[str] = []

        if self.use_ollama:
            try:
                similarity = self._embedding_similarity(original, candidate)
                source = "embedding"
                notes.append(
                    f"Embedding similarity={similarity:.3f} (threshold={threshold:.2f})."
                )
            except OllamaError:
                notes.append("Embedding validation unavailable; used lexical guardrails only.")
                source = "guardrail"

        passed = similarity >= threshold
        accepted = candidate if passed else original
        if not passed:
            notes.append("Rejected optimization: similarity below threshold.")

        return ValidationResult(
            passed=passed,
            similarity=similarity,
            threshold=threshold,
            original_query=original,
            candidate_query=candidate,
            accepted_query=accepted,
            source=source,
            notes=notes,
        )

    def _threshold(self) -> float:
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        if self.client is not None:
            return self.client.config.validation_similarity_threshold
        return OllamaClient().config.validation_similarity_threshold

    def _embedding_similarity(self, original: str, candidate: str) -> float:
        client = self.client or OllamaClient()
        original_embedding = client.embed(original)
        candidate_embedding = client.embed(candidate)
        return cosine_similarity(original_embedding, candidate_embedding)
