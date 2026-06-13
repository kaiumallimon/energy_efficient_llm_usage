from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config/ollama.json")


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_THINK: bool | None = False


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    think: bool | None = DEFAULT_THINK

    @classmethod
    def load(cls, path: Path | str | None = None) -> OllamaConfig:
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        file_values: dict[str, object] = {}

        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid Ollama config format in {config_path}")
            file_values = payload

        base_url = os.getenv("OLLAMA_BASE_URL", file_values.get("base_url", DEFAULT_BASE_URL))
        model = os.getenv("OLLAMA_MODEL", file_values.get("model", DEFAULT_MODEL))
        timeout_raw = os.getenv(
            "OLLAMA_TIMEOUT_SECONDS",
            file_values.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        )
        think_raw = os.getenv("OLLAMA_THINK", file_values.get("think", DEFAULT_THINK))

        return cls(
            base_url=str(base_url).rstrip("/"),
            model=str(model),
            timeout_seconds=float(timeout_raw),
            think=_parse_optional_bool(think_raw),
        )


def _parse_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"auto", "none"}:
        return None
    raise ValueError(f"Invalid boolean config value: {value!r}")
