from __future__ import annotations

import json
import re

from src.analyzer.models import INTENT_LABELS, TaskType

INTENT_SYSTEM_PROMPT = (
    "You classify user prompts for an energy-efficient LLM pipeline. "
    "Return ONLY valid JSON with keys: task_type, confidence, rationale. "
    f"Allowed task_type values: {', '.join(INTENT_LABELS)}."
)

INTENT_USER_TEMPLATE = """Classify this user prompt.

Rules:
- Short "what is X" / "define X" / acronym questions are definition or concept_explanation, NOT coding.
- coding only when the user asks to write, debug, implement, or fix code.
- exam_help when the user asks for homework, exam, quiz, or study help.
- educational for teaching-style requests ("teach me", "help me learn").

Prompt:
{query}
"""


def _extract_json_object(text: str) -> dict[str, object]:
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

    raise ValueError("No JSON object found in classifier response.")


def _normalize_task_type(raw: object) -> TaskType:
    if raw is None:
        return TaskType.UNKNOWN

    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "concept": TaskType.CONCEPT_EXPLANATION,
        "concept_explain": TaskType.CONCEPT_EXPLANATION,
        "explain_concept": TaskType.CONCEPT_EXPLANATION,
        "define": TaskType.DEFINITION,
        "qa": TaskType.FACTUAL,
        "question": TaskType.FACTUAL,
        "study": TaskType.EXAM_HELP,
        "homework": TaskType.EXAM_HELP,
    }
    if normalized in aliases:
        return aliases[normalized]

    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.UNKNOWN


class IntentClassifier:
    """Lightweight Ollama-backed intent classifier with heuristic fallback."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        use_ollama: bool = True,
    ) -> None:
        self.client = client
        self.use_ollama = use_ollama

    def classify(self, query: str) -> tuple[TaskType, float, str, str]:
        if self.use_ollama:
            try:
                return self._classify_with_ollama(query)
            except Exception:
                pass

        from src.analyzer.signals import detect_task_type

        task_type, scores = detect_task_type(query)
        confidence = max(scores.values(), default=0.4)
        return task_type, confidence, "Heuristic intent classification.", "heuristic"

    def _classify_with_ollama(self, query: str) -> tuple[TaskType, float, str, str]:
        from src.llm.ollama_client import OllamaClient, OllamaError

        client = self.client or OllamaClient()
        response = client.generate_text(
            INTENT_USER_TEMPLATE.format(query=query.strip()),
            model=client.config.analyzer_model,
            system=INTENT_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=160,
        )
        payload = _extract_json_object(response)
        task_type = _normalize_task_type(payload.get("task_type"))
        confidence_raw = payload.get("confidence", 0.75)
        confidence = max(0.0, min(1.0, float(confidence_raw)))
        rationale = str(payload.get("rationale") or "Ollama intent classification.")
        return task_type, confidence, rationale, "ollama"
