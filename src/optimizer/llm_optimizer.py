from __future__ import annotations

from src.analyzer.models import ComplexityResult, OptimizationPolicy
from src.decomposer.models import DecomposedPrompt
from src.llm.ollama_client import OllamaClient, OllamaError

OPTIMIZER_SYSTEM_PROMPT = (
    "You perform energy-aware semantic prompt compression. "
    "Rewrite user prompts to be shorter while preserving intent and constraints. "
    "Return ONLY the rewritten prompt with no commentary."
)

OPTIMIZER_USER_TEMPLATE = """Rewrite the prompt.

Requirements:
- preserve intent.
- Reduce tokens.
- Remove redundancy.
- keep quality.

Structured analysis:
- intent: {intent}
- core_request: {core_request}
- entities: {entities}
- constraints: {constraints}
- redundant_segments: {redundant_segments}

Input:
{user_input}
"""


class LLMPromptOptimizer:
    """Semantic prompt compressor backed by a small Ollama model."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        use_ollama: bool = True,
    ) -> None:
        self.client = client
        self.use_ollama = use_ollama

    def optimize_query(
        self,
        query: str,
        analysis: ComplexityResult,
        decomposed: DecomposedPrompt,
    ) -> tuple[str, list[str]]:
        if analysis.policy == OptimizationPolicy.MINIMAL:
            return query.strip(), ["Minimal policy: skipped LLM optimization."]

        if not self.use_ollama:
            return query.strip(), ["LLM optimization disabled."]

        try:
            rewritten = self._rewrite_with_ollama(query, decomposed)
        except OllamaError as exc:
            return query.strip(), [f"LLM optimization unavailable: {exc}"]

        rewritten = rewritten.strip().strip('"').strip("'")
        if not rewritten:
            return query.strip(), ["LLM optimizer returned empty text; kept original."]

        if rewritten == query.strip():
            return rewritten, ["LLM optimizer produced no semantic changes."]

        return rewritten, ["Applied Ollama semantic prompt compression."]

    def _rewrite_with_ollama(self, query: str, decomposed: DecomposedPrompt) -> str:
        client = self.client or OllamaClient()
        prompt = OPTIMIZER_USER_TEMPLATE.format(
            intent=decomposed.intent,
            core_request=decomposed.core_request,
            entities=", ".join(decomposed.entities) or "(none)",
            constraints="; ".join(decomposed.constraints) or "(none)",
            redundant_segments="; ".join(decomposed.redundant_segments) or "(none)",
            user_input=query.strip(),
        )
        return client.generate_text(
            prompt,
            model=client.config.optimizer_model,
            system=OPTIMIZER_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=256,
        )
