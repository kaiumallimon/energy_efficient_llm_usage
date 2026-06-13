from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.baseline import BaselineRunner
from src.llm.models import LLMCallResult, TokenUsage


def make_llm_result(response: str = "Dhaka", *, prompt_tokens: int = 28) -> LLMCallResult:
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=2,
        total_tokens=prompt_tokens + 2,
    )
    return LLMCallResult(
        model="qwen3.5:4b",
        provider="ollama",
        response=response,
        thinking=None,
        usage=usage,
        latency_ms=150.0,
        energy_proxy=usage.total_tokens * 0.0008,
    )


class TestBaselineRunner:
    def test_sends_raw_query_to_llm(self) -> None:
        mock_client = MagicMock()
        mock_client.call_baseline.return_value = make_llm_result()

        result = BaselineRunner(mock_client).run(
            "could you please tell me what is the capital of Bangladesh??",
            think=False,
        )

        mock_client.call_baseline.assert_called_once_with(
            "could you please tell me what is the capital of Bangladesh??",
            None,
            think=False,
        )
        assert result.completion == "Dhaka"
        assert result.prompt_words == 11

    def test_to_dict_matches_pipeline_llm_fields(self) -> None:
        mock_client = MagicMock()
        mock_client.call_baseline.return_value = make_llm_result()

        payload = BaselineRunner(mock_client).run("What is 2 + 2?").to_dict()

        assert payload["path"] == "baseline"
        assert payload["query"] == "What is 2 + 2?"
        assert payload["completion"] == "Dhaka"
        assert payload["llm"]["completion"] == "Dhaka"
        assert payload["monitoring"]["baseline"]["path"] == "baseline"
        assert payload["monitoring"]["baseline"]["total_tokens"] == 30

    def test_includes_context_word_count(self) -> None:
        mock_client = MagicMock()
        mock_client.call_baseline.return_value = make_llm_result()

        result = BaselineRunner(mock_client).run(
            "Summarize this.",
            context="Revenue grew 12 percent.",
        )

        assert result.prompt_words == 6
        mock_client.call_baseline.assert_called_once_with(
            "Summarize this.",
            "Revenue grew 12 percent.",
            think=None,
        )


class TestBaselineCli:
    def test_main_json_output(self, capsys, monkeypatch) -> None:
        from src.baseline_cli import main

        mock_client = MagicMock()
        mock_client.call_baseline.return_value = make_llm_result("4")

        monkeypatch.setattr(
            "src.baseline_cli.build_runner",
            lambda args: BaselineRunner(mock_client),
        )

        exit_code = main(["What is 2 + 2?", "--json"])

        assert exit_code == 0
        captured = capsys.readouterr().out
        assert '"path": "baseline"' in captured
        assert '"completion": "4"' in captured

    def test_main_human_output(self, capsys, monkeypatch) -> None:
        from src.baseline_cli import main

        mock_client = MagicMock()
        mock_client.call_baseline.return_value = make_llm_result("4")

        monkeypatch.setattr(
            "src.baseline_cli.build_runner",
            lambda args: BaselineRunner(mock_client),
        )

        exit_code = main(["What is 2 + 2?"])

        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Baseline LLM Call" in captured
        assert "Completion: 4" in captured
        assert "Monitoring" in captured

    def test_main_accepts_call_flag(self, capsys, monkeypatch) -> None:
        from src.baseline_cli import main

        mock_client = MagicMock()
        mock_client.call_baseline.return_value = make_llm_result("Dhaka")

        monkeypatch.setattr(
            "src.baseline_cli.build_runner",
            lambda args: BaselineRunner(mock_client),
        )

        exit_code = main(
            [
                "could you please tell me what is the capital of Bangladesh??",
                "--call",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Completion: Dhaka" in captured
        mock_client.call_baseline.assert_called_once()
