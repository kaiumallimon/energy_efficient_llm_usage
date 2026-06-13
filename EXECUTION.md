# Execution Guide

How to set up, run, and test the **energy-efficient LLM usage** prototype.

For architecture and design rationale, see [`README.md`](README.md). For analyzer internals, see [`src/analyzer/README.md`](src/analyzer/README.md).

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Tested with 3.14 on Windows |
| **pytest** | Listed in `requirements-dev.txt` |
| **Ollama** (optional) | Required only for live LLM calls (`--call` or integration tests) |

All pipeline stages except the LLM call are **local, rule-based, and offline**. No API keys are needed for analyzer, optimizer, generator, or benchmark runs.

---

## Setup

From the repository root:

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install test dependencies
pip install -r requirements-dev.txt

# Optional: install package entry points (analyze-complexity, optimize-prompt)
pip install -e .
```

Run commands from the **repository root** so relative paths like `config/ollama.json` and `data/benchmark_prompts.json` resolve correctly.

If `analyze-complexity` or `optimize-prompt` are not found after `pip install -e .`, use the `python -m` forms below or add your Python `Scripts` directory to `PATH`.

---

## Architecture at a Glance

The implemented pipeline chains four core stages plus optional LLM execution, monitoring, and evaluation.

```
User query (+ optional context)
        ↓
1. Complexity Analyzer          src/analyzer/
        ↓
2. Prompt Optimizer             src/optimizer/
        ↓
3. Adaptive Prompt Generator    src/generator/
        ↓
4. (optional) Ollama LLM call       src/llm/
        ↓
5. Performance Monitoring       src/monitoring/
        ↓
6. Evaluation vs baseline       src/evaluation/
```

**Orchestration:** `src/pipeline.py` (`PromptPipeline`) wires all stages together.

Monitoring runs on every request. Evaluation runs when `--evaluate` is passed; with `--call`, it also runs a baseline LLM call on the raw prompt for token and completion comparison.

---

## Running the Pipeline

### Full pipeline (recommended entry point)

Analyze, optimize, and assemble the final prompt:

```bash
python -m src.pipeline_cli "Could you please tell me what the capital of France is?"
```

With machine-readable output:

```bash
python -m src.pipeline_cli "What is 2 + 2?" --json
```

**Useful flags:**

| Flag | Purpose |
|------|---------|
| `--context`, `-c` | Inline context text or path to a context file |
| `--file`, `-f` | Read the query from a text file |
| `--json` | Print the full `PipelineResult` as JSON |
| `--call` | Send the generated messages to Ollama and include token usage |
| `--evaluate` | Compare optimized vs baseline; with `--call`, runs both LLM paths |
| `--model` | Override the Ollama model name |
| `--ollama-url` | Override the Ollama base URL |
| `--think true\|false\|auto` | Control Qwen thinking mode (`auto` = on for high/critical complexity) |

**Examples:**

```bash
# Query from file with external context
python -m src.pipeline_cli --file my_prompt.txt --context data/context/meeting_notes.txt --json

# Run through the pipeline and call the local model
python -m src.pipeline_cli "Summarize quarterly revenue in 3 bullets." --call

# Call model and compare against unoptimized baseline
python -m src.pipeline_cli "Could you please calculate what is the answer of 2+2 is?" --call --evaluate --json

# Offline efficiency check (word reduction only, no LLM)
python -m src.pipeline_cli "Could you please tell me what 2 + 2 is?" --evaluate --json

# Override model and disable thinking
python -m src.pipeline_cli "What is REST?" --call --model qwen3.5:4b --think false
```

Installed entry point (equivalent):

```bash
optimize-prompt "What is 2 + 2?" --json
```

### Analyzer only (stage 2)

Classify complexity without optimizing or generating:

```bash
python -m src.analyzer.cli "Compare REST and GraphQL step by step"
python -m src.analyzer.cli "Summarize the key points" --context notes.txt --json
```

Installed entry point:

```bash
analyze-complexity "What is the capital of France?" --json
```

### Benchmark suite (stages 2–4, no LLM)

Runs labeled cases from `data/benchmark_prompts.json` through the full offline pipeline and checks expectations (complexity level, policy, model tier, optimization behavior, prompt content):

```bash
# Summary (shows failures only)
python -m src.benchmark_cli

# Show all cases
python -m src.benchmark_cli --verbose

# JSON report (exit code 1 if any case fails)
python -m src.benchmark_cli --json

# Run specific case(s)
python -m src.benchmark_cli --id simple_fact_verbose --id medical_safety
```

Custom benchmark file:

```bash
python -m src.benchmark_cli --file path/to/benchmark_prompts.json
```

---

## Ollama Setup (optional, for `--call`)

1. Install and start [Ollama](https://ollama.com/).
2. Pull the configured model (default: `qwen3.5:4b`):

   ```bash
   ollama pull qwen3.5:4b
   ```

3. Confirm the server is reachable:

   ```bash
   curl http://localhost:11434/api/tags
   ```

### Configuration

Settings are loaded in this order (later wins):

1. Defaults in `src/llm/config.py`
2. `config/ollama.json`
3. Environment variables

| Setting | Config key | Environment variable | Default |
|---------|------------|----------------------|---------|
| Base URL | `base_url` | `OLLAMA_BASE_URL` | `http://localhost:11434` |
| Model | `model` | `OLLAMA_MODEL` | `qwen3.5:4b` |
| Timeout (seconds) | `timeout_seconds` | `OLLAMA_TIMEOUT_SECONDS` | `120` |
| Thinking mode | `think` | `OLLAMA_THINK` | `false` |

`think` accepts `true`, `false`, or `auto`/`none` (let the client decide based on complexity).

When `--call` is used, the CLI reports prompt/completion/total tokens, latency, and a simple **energy proxy** (not measured joules—see `src/llm/ollama_client.py`).

---

## Python API

### End-to-end pipeline (offline)

```python
from src.pipeline import PromptPipeline

result = PromptPipeline().process(
    "Could you please explain in order to understand photosynthesis.",
)

print(result.analysis.policy)               # aggressive / moderate / conservative / minimal
print(result.optimization.optimized_query)  # compressed user text
print(result.generation.model_tier)         # small / medium / large
print(result.generation.messages)           # [{role, content}, ...]
print(result.to_dict())
```

### With optional Ollama call and evaluation

```python
from src.llm import OllamaClient
from src.pipeline import PromptPipeline

pipeline = PromptPipeline(llm_client=OllamaClient())
result = pipeline.process(
    "Could you please calculate what is the answer of 2+2 is?",
    call_llm=True,
    evaluate=True,
    think=False,
)

print(result.completion)                    # top-level optimized LLM output
print(result.llm.usage.total_tokens)        # optimized path tokens
print(result.evaluation.efficiency.word_reduction_percent)
print(result.evaluation.baseline_completion)
print(result.evaluation.optimized_completion)
print(result.monitoring.to_dict())
print(result.to_dict())
```

### Individual stages

```python
from src.analyzer import ComplexityAnalyzer
from src.optimizer import PromptOptimizer
from src.generator import AdaptivePromptGenerator

analysis = ComplexityAnalyzer().analyze("What is the capital of France?")
optimization = PromptOptimizer().optimize("Could you please ...", None, analysis)
generation = AdaptivePromptGenerator().generate(analysis, optimization)
```

### Benchmark programmatically

```python
from src.benchmark import run_benchmark, summarize_results

results = run_benchmark()  # default: data/benchmark_prompts.json
print(summarize_results(results))
```

---

## Testing

### Run everything (fast, no Ollama required)

```bash
python -m pytest
```

As of the current tree: **37 tests**, all offline except one optional integration test.

### Run by area

```bash
# Complexity analyzer
python -m pytest tests/test_complexity_analyzer.py -q

# Prompt optimizer + pipeline wiring
python -m pytest tests/test_optimizer.py -q

# Adaptive prompt generator
python -m pytest tests/test_generator.py -q

# Labeled benchmark cases (23 cases in data/benchmark_prompts.json)
python -m pytest tests/test_benchmark_prompts.py -q

# Ollama config/client (mocked) + pipeline LLM wiring
python -m pytest tests/test_llm.py -q
```

### Live Ollama integration test

One test is marked `@pytest.mark.integration`. It calls a real Ollama server and **skips** if Ollama is unavailable:

```bash
# Run only integration tests (requires Ollama running + model pulled)
python -m pytest -m integration

# Run all tests except integration
python -m pytest -m "not integration"
```

### What each test layer covers

| Test file | What it validates |
|-----------|-------------------|
| `test_complexity_analyzer.py` | Level/policy routing, safety overrides, context scoring, result shape |
| `test_optimizer.py` | Filler removal, redundancy compression, context trim/dedupe, minimal policy, pipeline link |
| `test_generator.py` | Model tier selection, system/user prompt assembly, safety/constraint guardrails |
| `test_benchmark_prompts.py` | Full offline pipeline against all labeled benchmark cases |
| `test_llm.py` | Config loading, mocked Ollama responses, optional live call |
| `test_monitoring_evaluation.py` | Monitoring snapshots, baseline comparison, completion in response |

---

## Data Files

| Path | Purpose |
|------|---------|
| `data/benchmark_prompts.json` | 23 labeled end-to-end cases for `benchmark_cli` and `test_benchmark_prompts.py` |
| `data/sample_prompts.json` | Smaller set of example prompts with expected complexity levels |
| `data/context/meeting_notes.txt` | Sample context file referenced by benchmark case `summarize_with_context_file` |
| `config/ollama.json` | Default Ollama connection settings |

---

## Quick Reference

| Goal | Command |
|------|---------|
| Install dev deps | `pip install -r requirements-dev.txt` |
| All unit tests | `python -m pytest` |
| Offline benchmark | `python -m src.benchmark_cli --verbose` |
| Analyze one prompt | `python -m src.analyzer.cli "..." --json` |
| Full pipeline (no LLM) | `python -m src.pipeline_cli "..." --json` |
| Full pipeline + Ollama | `python -m src.pipeline_cli "..." --call --json` |
| Full pipeline + evaluation | `python -m src.pipeline_cli "..." --call --evaluate --json` |
| Integration test only | `python -m pytest -m integration` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Could not reach Ollama` | Ollama not running | Start Ollama; verify `http://localhost:11434` |
| Model not found | Model not pulled | `ollama pull qwen3.5:4b` (or your `--model` value) |
| `ModuleNotFoundError: src` | Wrong working directory | `cd` to repository root before running |
| Benchmark file not found | Missing `data/` | Ensure `data/benchmark_prompts.json` exists locally |
| Console scripts not found | Scripts dir not on `PATH` | Use `python -m src.pipeline_cli` instead |

---

## Implementation Status

| Component | Status | How to exercise |
|-----------|--------|-----------------|
| Complexity analyzer | Done | `src.analyzer.cli`, `test_complexity_analyzer.py` |
| Prompt optimizer | Done | `src.pipeline_cli`, `test_optimizer.py` |
| Adaptive prompt generator | Done | `src.pipeline_cli`, `test_generator.py` |
| Pipeline orchestration | Done | `src.pipeline.py`, `src.pipeline_cli` |
| Ollama LLM wrapper | Done | `--call`, `test_llm.py` |
| Performance monitoring | Done | Always in `monitoring` field; `test_monitoring_evaluation.py` |
| Evaluation module | Done | `--evaluate`; baseline LLM when combined with `--call` |
| Benchmark runner | Done | `src.benchmark_cli`, `test_benchmark_prompts.py` |
