from __future__ import annotations

import json
import re

from src.analyzer.models import ComplexityResult
from src.decomposer.models import DecomposedPrompt
from src.constraints import extract_constraints
from src.llm.ollama_client import OllamaClient, OllamaError
from src.optimizer.phrases import ANALYZER_FILLER_PATTERNS
from src.text_utils import WORD_RE

DECOMPOSER_SYSTEM_PROMPT = (
    "You decompose user prompts into structured JSON for semantic compression. "
    "Return ONLY valid JSON with keys: intent, core_request, entities, constraints, "
    "redundant_segments, context_summary, requires_context."
)

DECOMPOSER_USER_TEMPLATE = """Decompose this prompt.

Task type: {task_type}
Complexity: {complexity}

User prompt:
{query}

Context (optional):
{context}
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

    raise ValueError("No JSON object found in decomposer response.")


def _detect_redundant_segments(query: str) -> list[str]:
    redundant: list[str] = []
    lowered = query.lower()
    for pattern in ANALYZER_FILLER_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            segment = query[match.start() : match.end()].strip()
            if segment and segment not in redundant:
                redundant.append(segment)
    return redundant


def _extract_entities(query: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z]{2,}\b", query)
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
    entities = [item for group in quoted for item in group if item]
    entities.extend(candidates)
    unique: list[str] = []
    for entity in entities:
        if entity not in unique:
            unique.append(entity)
    return unique


class PromptDecomposer:
    """Builds a structured representation used by the semantic optimizer."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        use_ollama: bool = True,
    ) -> None:
        self.client = client
        self.use_ollama = use_ollama

    def decompose(
        self,
        query: str,
        context: str | None,
        analysis: ComplexityResult,
    ) -> DecomposedPrompt:
        if self.use_ollama:
            try:
                return self._decompose_with_ollama(query, context, analysis)
            except (OllamaError, ValueError, json.JSONDecodeError):
                pass
        return self._decompose_heuristic(query, context, analysis)

    def _decompose_heuristic(
        self,
        query: str,
        context: str | None,
        analysis: ComplexityResult,
    ) -> DecomposedPrompt:
        constraints = extract_constraints(query)
        entities = _extract_entities(query)
        redundant = _detect_redundant_segments(query)
        context_words = len(WORD_RE.findall(context or ""))
        context_summary = None
        if context and context_words > 40:
            words = WORD_RE.findall(context)
            context_summary = " ".join(words[:40]) + ("..." if len(words) > 40 else "")

        return DecomposedPrompt(
            intent=analysis.task_type.value,
            core_request=query.strip(),
            entities=entities,
            constraints=constraints,
            redundant_segments=redundant,
            context_summary=context_summary,
            requires_context=bool(context),
            source="heuristic",
        )

    def _decompose_with_ollama(
        self,
        query: str,
        context: str | None,
        analysis: ComplexityResult,
    ) -> DecomposedPrompt:
        client = self.client or OllamaClient()
        response = client.generate_text(
            DECOMPOSER_USER_TEMPLATE.format(
                task_type=analysis.task_type.value,
                complexity=analysis.level.value,
                query=query.strip(),
                context=(context or "").strip() or "(none)",
            ),
            model=client.config.analyzer_model,
            system=DECOMPOSER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=320,
        )
        payload = _extract_json_object(response)
        entities = payload.get("entities", [])
        constraints = payload.get("constraints", [])
        redundant = payload.get("redundant_segments", [])
        return DecomposedPrompt(
            intent=str(payload.get("intent") or analysis.task_type.value),
            core_request=str(payload.get("core_request") or query.strip()),
            entities=[str(item) for item in entities] if isinstance(entities, list) else [],
            constraints=[str(item) for item in constraints] if isinstance(constraints, list) else [],
            redundant_segments=[str(item) for item in redundant] if isinstance(redundant, list) else [],
            context_summary=str(payload["context_summary"])
            if payload.get("context_summary")
            else None,
            requires_context=bool(payload.get("requires_context", bool(context))),
            source="ollama",
        )
