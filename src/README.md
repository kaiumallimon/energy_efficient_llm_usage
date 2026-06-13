# CLIs in src/

This file documents the command-line entry points under the `src/` package and shows example invocations with commonly used flags and arguments. Run commands from the repository root so relative paths (e.g. `config/ollama.json`, `data/`) resolve correctly.

Prerequisites
- Create and activate a virtual environment and install dev deps:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# or on macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

General notes
- Use the `python -m src.module_cli` form if you did not install the package entry points.
- If you installed the project editable (`pip install -e .`), the console scripts are available: `optimize-prompt`, `analyze-complexity`, `call-baseline`.

1) Pipeline CLI
----------------
Entry module: `src/pipeline_cli.py`

Run the full pipeline (analyze → optimize → generate):

```bash
python -m src.pipeline_cli "Could you please tell me what the capital of France is?"
```

With JSON output:

```bash
python -m src.pipeline_cli "What is 2 + 2?" --json
```

Common flags:
- `--context` / `-c`: Inline context text or path to a context file
- `--file` / `-f`: Read the query from a file
- `--json`: Print full `PipelineResult` as JSON
- `--call`: Send the generated messages to Ollama and include token usage
- `--evaluate`: Compare optimized path vs baseline (works with and without `--call`)
- `--model`: Override Ollama model
- `--ollama-url`: Override Ollama base URL
- `--think true|false|auto`: Control thinking mode (Qwen)

Examples:

```bash
# Run pipeline and call local model
python -m src.pipeline_cli "Summarize quarterly revenue in 3 bullets." --call

# Call model and compare against unoptimized baseline
python -m src.pipeline_cli "Could you please calculate what is the answer of 2+2 is?" --call --evaluate --json

# Offline efficiency check (no LLM)
python -m src.pipeline_cli "Could you please tell me what 2 + 2 is?" --evaluate --json

# Use external context file
python -m src.pipeline_cli --file my_prompt.txt --context data/context/meeting_notes.txt --json
```

2) Baseline CLI
----------------
Entry module: `src/baseline_cli.py`

Send the raw query directly to the LLM (unoptimized baseline):

```bash
python -m src.baseline_cli "could you please tell me what is the capital of Bangladesh??"
python -m src.baseline_cli "could you please tell me what is the capital of Bangladesh??" --json

# Installed entry point (if package installed):
call-baseline "What is 2 + 2?" --json
```

3) Benchmark CLI
-----------------
Entry module: `src/benchmark_cli.py`

Run the offline benchmark suite (stages 2–4, no LLM by default):

```bash
# Summary (shows failures only)
python -m src.benchmark_cli

# Show all cases
python -m src.benchmark_cli --verbose

# JSON report (exit code 1 if any case fails)
python -m src.benchmark_cli --json

# Run specific case(s)
python -m src.benchmark_cli --id simple_fact_verbose --id medical_safety

# Use a custom benchmark file
python -m src.benchmark_cli --file path/to/benchmark_prompts.json
```

4) Analyzer CLI
---------------
Entry module: `src/analyzer/cli.py`

Classify complexity without optimizing or generating:

```bash
python -m src.analyzer.cli "Compare REST and GraphQL step by step"
python -m src.analyzer.cli "Summarize the key points" --context notes.txt --json

# Installed entry point (if package installed):
analyze-complexity "What is the capital of France?" --json
```

5) Compare CLI
--------------
Entry module: `src/compare_cli.py`

Used to compare outputs or results (see source for options). Example:

```bash
python -m src.compare_cli --help
python -m src.compare_cli "first input" "second input" --json
```

6) Other useful modules
----------------------
- `src/pipeline_fallback.py`: fallback behavior for pipeline runs. Use `--help` to inspect options.

Inspect help for any CLI
------------------------
For a quick overview of flags and usage for any module, run:

```bash
python -m src.pipeline_cli --help
python -m src.baseline_cli --help
python -m src.benchmark_cli --help
python -m src.analyzer.cli --help
```

Notes about Ollama (`--call`)
- Ollama is optional. If you use `--call` the CLI will attempt to contact an Ollama server.
- Defaults are controlled by `config/ollama.json` and `src/llm/config.py`.
- Common override examples:

```bash
# Override model and disable thinking
python -m src.pipeline_cli "What is REST?" --call --model qwen3.5:4b --think false

# Set a custom Ollama URL
python -m src.pipeline_cli "Tell me a joke" --call --ollama-url http://localhost:11434
```

Problems & troubleshooting
- If you see `ModuleNotFoundError: src`, ensure you run commands from the repository root.
- If console scripts are not available after `pip install -e .`, use the `python -m` form or add your virtualenv `Scripts` dir to `PATH`.

Want this added to the top-level README instead?
Open an issue or update the file as needed.
