# Ollama Setup Instructions

This project uses **Ollama only** for all model calls: intent analysis, prompt decomposition, semantic optimization, embedding validation, and final inference.

## 1. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com), then verify the service is running:

```bash
ollama --version
curl http://localhost:11434/api/tags
```

Default API base URL: `http://localhost:11434`

## 2. Pull Required Models

Run these commands once before using the pipeline:

```bash
# Main inference model (final answer generation)
ollama pull qwen3.5:4b

# Lightweight models for analyzer, decomposer, and semantic optimizer
ollama pull qwen2.5:0.5b

# Embedding model for quality validation (similarity >= 0.9)
ollama pull nomic-embed-text
```

### Model roles

| Role | Default model | Purpose |
|------|---------------|---------|
| `model` | `qwen3.5:4b` | Final LLM response |
| `analyzer_model` | `qwen2.5:0.5b` | Intent & complexity classification |
| `optimizer_model` | `qwen2.5:0.5b` | Energy-aware semantic prompt compression |
| `embedding_model` | `nomic-embed-text` | Quality validator embedding similarity |

You can override models in `config/ollama.json` or with environment variables:

```bash
export OLLAMA_MODEL=qwen3.5:4b
export OLLAMA_ANALYZER_MODEL=qwen2.5:0.5b
export OLLAMA_OPTIMIZER_MODEL=qwen2.5:0.5b
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_VALIDATION_SIMILARITY_THRESHOLD=0.9
```

## 3. Verify Models

```bash
ollama list
```

Expected tags include at least:

- `qwen3.5:4b`
- `qwen2.5:0.5b`
- `nomic-embed-text`

Quick smoke tests:

```bash
# Chat model
ollama run qwen2.5:0.5b "Classify: what is CFG shortly?"

# Embedding model
curl http://localhost:11434/api/embed -d '{
  "model": "nomic-embed-text",
  "input": "what is CFG shortly?"
}'
```

## 4. Run the Pipeline

```bash
pip install -r requirements-dev.txt

# Offline analysis + rule fallback (no Ollama required)
python -m src.pipeline_cli "what is CFG shortly?" --json

# Full Ollama-backed semantic pipeline
python -m src.pipeline_cli "what is CFG shortly?" --json

# End-to-end with inference + baseline evaluation
python -m src.pipeline_cli "what is CFG shortly?" --call --evaluate --json
```

Enable Ollama-backed stages via `PromptPipeline(use_ollama=True)` or the CLI (enabled by default in `pipeline_cli`).

## 5. Architecture (Ollama Stages)

```
Raw user query
  -> Intent & complexity analyzer (qwen2.5:0.5b)
  -> Prompt decomposer (qwen2.5:0.5b)
  -> Semantic prompt optimizer (qwen2.5:0.5b)
  -> Quality validator (nomic-embed-text, threshold 0.9)
  -> Adaptive prompt generator (inference params by complexity)
  -> Final LLM call (qwen3.5:4b)
```

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| `Could not reach Ollama` | Start Ollama app/service; confirm port `11434` |
| `model not found` | Run the matching `ollama pull ...` command |
| Optimizer rejected | Embedding similarity `< 0.9`; original prompt is kept |
| Slow first call | First run loads model weights; later calls are faster |
| Wrong task type offline | Run with `use_ollama=True` and ensure analyzer model is pulled |

## 7. Hardware Notes

- `qwen2.5:0.5b` and `nomic-embed-text` are intended for low-cost local hardware.
- `qwen3.5:4b` is the default balanced inference model; swap in `config/ollama.json` if you need smaller/larger models.
- Semantic optimization cost is logged implicitly through Ollama calls; include optimizer + embed tokens when reporting total energy savings.
