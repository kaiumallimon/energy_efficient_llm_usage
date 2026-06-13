from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.generator.models import GeneratedPrompt
from src.llm import OllamaClient, OllamaConfig, OllamaError
from src.llm.config import _parse_optional_bool
from src.pipeline import PromptPipeline


@pytest.fixture
def sample_generated() -> GeneratedPrompt:
    from src.analyzer.models import ComplexityLevel, OptimizationPolicy, TaskType

    return GeneratedPrompt(
        system_prompt="Answer concisely.",
        user_prompt="What is 2 + 2?",
        context=None,
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        full_prompt="System:\nAnswer concisely.\n\nUser:\nWhat is 2 + 2?",
        model_tier="small",
        template_id="factual_aggressive",
        task_type=TaskType.FACTUAL,
        complexity_level=ComplexityLevel.LOW,
        policy=OptimizationPolicy.AGGRESSIVE,
    )


class TestOllamaConfig:
    def test_loads_defaults_without_file(self, tmp_path, monkeypatch) -> None:
        missing = tmp_path / "missing.json"
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)

        config = OllamaConfig.load(missing)

        assert config.base_url == "http://localhost:11434"
        assert config.model == "qwen3.5:4b"
        assert config.think is False

    def test_env_overrides_file(self, tmp_path, monkeypatch) -> None:
        config_path = tmp_path / "ollama.json"
        config_path.write_text(
            json.dumps({"base_url": "http://file:11434", "model": "file-model"}),
            encoding="utf-8",
        )
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "env-model")

        config = OllamaConfig.load(config_path)

        assert config.base_url == "http://env:11434"
        assert config.model == "env-model"

    def test_parse_optional_bool(self) -> None:
        assert _parse_optional_bool("auto") is None
        assert _parse_optional_bool("false") is False
        assert _parse_optional_bool(True) is True


class TestOllamaClient:
    def test_chat_parses_token_usage(self, sample_generated: GeneratedPrompt) -> None:
        payload = {
            "model": "qwen3.5:4b",
            "message": {"role": "assistant", "content": "4"},
            "prompt_eval_count": 12,
            "eval_count": 3,
            "load_duration": 50_000_000,
            "eval_duration": 150_000_000,
            "total_duration": 200_000_000,
            "done": True,
        }
        client = OllamaClient(OllamaConfig(model="qwen3.5:4b"))

        with patch.object(client, "_post", return_value=payload):
            result = client.call(sample_generated)

        assert result.provider == "ollama"
        assert result.response == "4"
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 3
        assert result.usage.total_tokens == 15
        assert result.usage.thinking_enabled is False
        assert result.load_duration_ms == 50.0
        assert result.eval_duration_ms == 150.0

    def test_chat_sends_think_flag_when_configured(
        self, sample_generated: GeneratedPrompt
    ) -> None:
        client = OllamaClient(OllamaConfig(model="qwen3.5:4b", think=False))
        captured: dict = {}

        def fake_post(path: str, body: dict) -> dict:
            captured["path"] = path
            captured["body"] = body
            return {
                "model": "qwen3.5:4b",
                "message": {"role": "assistant", "content": "4"},
                "prompt_eval_count": 10,
                "eval_count": 2,
            }

        with patch.object(client, "_post", side_effect=fake_post):
            client.call(sample_generated)

        assert captured["path"] == "/api/chat"
        assert captured["body"]["think"] is False
        assert captured["body"]["stream"] is False

    def test_unreachable_server_raises_clear_error(
        self, sample_generated: GeneratedPrompt
    ) -> None:
        client = OllamaClient(OllamaConfig(base_url="http://127.0.0.1:1"))

        with pytest.raises(OllamaError, match="Could not reach Ollama"):
            client.call(sample_generated)


class TestPromptPipelineLLM:
    def test_process_without_call_skips_llm(self) -> None:
        result = PromptPipeline().process("What is 2 + 2?")

        assert result.llm is None
        assert "llm" not in result.to_dict()

    def test_process_with_call_uses_client(self, sample_generated: GeneratedPrompt) -> None:
        mock_client = MagicMock()
        mock_client.call.return_value = MagicMock(
            to_dict=lambda: {"model": "qwen3.5:4b"}
        )

        pipeline = PromptPipeline(llm_client=mock_client)
        result = pipeline.process("What is 2 + 2?", call_llm=True, think=False)

        assert result.llm is not None
        mock_client.call.assert_called_once()
        assert mock_client.call.call_args.kwargs["think"] is False
        assert "llm" in result.to_dict()


@pytest.mark.integration
def test_live_ollama_call_if_available() -> None:
    client = OllamaClient()
    generated = PromptPipeline().process("Say hello in 3 words.").generation

    try:
        result = client.call(generated, think=False)
    except OllamaError:
        pytest.skip("Ollama is not available")

    assert result.response
    assert result.usage.total_tokens > 0
