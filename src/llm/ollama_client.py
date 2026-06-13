from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from src.generator.models import GeneratedPrompt
from src.llm.config import OllamaConfig
from src.llm.models import LLMCallResult, TokenUsage

NS_TO_MS = 1_000_000.0


class OllamaError(RuntimeError):
    """Raised when an Ollama API call fails."""


class OllamaClient:
    """Calls a local Ollama server and records token usage from the API response."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig.load()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        think: bool | None = None,
    ) -> LLMCallResult:
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": False,
        }
        resolved_think = self._resolve_think(think)
        if resolved_think is not None:
            payload["think"] = resolved_think

        started = time.perf_counter()
        try:
            response_payload = self._post("/api/chat", payload)
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.config.base_url}. "
                "Is the Ollama service running?"
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._build_result(response_payload, latency_ms, resolved_think is True)

    def call(self, generated: GeneratedPrompt, *, think: bool | None = None) -> LLMCallResult:
        resolved_think = think if think is not None else self._auto_think(generated)
        return self.chat(generated.messages, think=resolved_think)

    def call_baseline(
        self,
        query: str,
        context: str | None = None,
        *,
        think: bool | None = None,
    ) -> LLMCallResult:
        """Send the raw user prompt without optimization or system instructions."""
        user_content = query.strip()
        if context:
            user_content = f"Context:\n{context.strip()}\n\nRequest:\n{user_content}"

        messages = [{"role": "user", "content": user_content}]
        return self.chat(messages, think=think)

    def _resolve_think(self, think: bool | None) -> bool | None:
        if think is not None:
            return think
        return self.config.think

    def _auto_think(self, generated: GeneratedPrompt) -> bool | None:
        if self.config.think is not None:
            return self.config.think
        return generated.complexity_level.value in {"high", "critical"}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise OllamaError("Ollama returned an unexpected response payload.")
        return data

    def _build_result(
        self,
        payload: dict[str, Any],
        latency_ms: float,
        thinking_enabled: bool,
    ) -> LLMCallResult:
        message = payload.get("message", {})
        if not isinstance(message, dict):
            raise OllamaError("Ollama response is missing message content.")

        prompt_tokens = int(payload.get("prompt_eval_count") or 0)
        completion_tokens = int(payload.get("eval_count") or 0)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            thinking_enabled=thinking_enabled,
        )

        return LLMCallResult(
            model=str(payload.get("model") or self.config.model),
            provider="ollama",
            response=str(message.get("content") or ""),
            thinking=message.get("thinking"),
            usage=usage,
            latency_ms=latency_ms,
            load_duration_ms=_ns_to_ms(payload.get("load_duration")),
            eval_duration_ms=_ns_to_ms(payload.get("eval_duration")),
            energy_proxy=_estimate_energy_proxy(usage.total_tokens, self.config.model),
            raw={
                "done": payload.get("done"),
                "done_reason": payload.get("done_reason"),
                "total_duration_ms": _ns_to_ms(payload.get("total_duration")),
                "prompt_eval_duration_ms": _ns_to_ms(payload.get("prompt_eval_duration")),
            },
        )


def _ns_to_ms(value: object) -> float | None:
    if value is None:
        return None
    return float(value) / NS_TO_MS


def _estimate_energy_proxy(total_tokens: int, model: str) -> float:
    """Simple local energy proxy for prototype reporting (not measured joules)."""
    model_lower = model.lower()
    if "4b" in model_lower or "3b" in model_lower:
        factor = 0.0008
    elif "7b" in model_lower or "8b" in model_lower:
        factor = 0.0012
    else:
        factor = 0.0015
    return total_tokens * factor
