# OpenBench: The Open-Source LLM Evaluation Powerhouse

**Tired of siloed LLM evaluations? OpenBench provides a unified, provider-agnostic platform for standardized and reproducible benchmarking of large language models.**  [Access the original repository here](https://github.com/groq/openbench).

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench is a robust, open-source evaluation infrastructure designed for comprehensive LLM benchmarking. It supports a vast array of models from leading providers and facilitates reproducible results across various domains. This platform offers a simple CLI, extensible architecture, and first-class support for local evaluations, ensuring both flexibility and privacy.

## Key Features:

*   **Extensive Benchmark Suite**: Access over 30+ benchmarks spanning:
    *   Knowledge
    *   Math
    *   Reasoning
    *   Reading Comprehension
    *   Health
    *   Long-Context Recall
*   **Provider-Agnostic**: Works seamlessly with 15+ model providers: Groq, OpenAI, Anthropic, Cohere, Google, AWS Bedrock, Azure, local models via Ollama, and more.
*   **Simple CLI**: Easily navigate and execute evaluations with intuitive commands: `bench list`, `bench describe`, and `bench eval`.
*   **Local Evaluation Support**: Run privatized benchmarks with confidence using `bench eval <path>`.
*   **Built on Inspect AI**: Leverages the industry-standard Inspect AI framework for reliable evaluations.
*   **Extensible**: Add new benchmarks and metrics with ease.

## Quick Start: Evaluate a Model in 60 Seconds

**Prerequisite**: [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Create a virtual environment and install OpenBench (30 seconds)
uv venv
source .venv/bin/activate
uv pip install openbench

# Set your API key (any provider!)
export GROQ_API_KEY=your_key  # or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# Run your first eval (30 seconds)
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10

# That's it! 🎉 Check results in ./logs/ or view them in an interactive UI:
bench view
```

## Supported Providers and Examples

```bash
# Groq (blazing fast!)
bench eval gpqa_diamond --model groq/meta-llama/llama-4-maverick-17b-128e-instruct

# OpenAI
bench eval humaneval --model openai/o3-2025-04-16

# Anthropic
bench eval simpleqa --model anthropic/claude-sonnet-4-20250514

# Google
bench eval mmlu --model google/gemini-2.5-pro

# Local models with Ollama
bench eval musr --model ollama/llama3.1:70b

# Hugging Face Inference Providers
bench eval mmlu --model huggingface/gpt-oss-120b:groq

# Any provider supported by Inspect AI!
```

## Benchmarks by Category

| Category       | Benchmarks                                                                                                                                                                                                                                                           |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Knowledge**  | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version)                                                                               |
| **Coding**     | HumanEval (164 problems)                                                                                                                                                                                                                                          |
| **Math**       | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages)                                |
| **Reasoning**  | SimpleQA (factuality), MuSR (multi-step reasoning)                                                                                                                                                                                                               |
| **Long Context** | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                                                                                                                         |
| **Healthcare** | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant)                                                                                                                                    |

## Configuration

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/llama-3.1-70b
```

## Command-Line Interface

For a complete list of all commands and options, run: `bench --help`

| Command                  | Description                                        |
|--------------------------|----------------------------------------------------|
| `bench`                  | Show main menu with available commands             |
| `bench list`             | List available evaluations, models, and flags      |
| `bench eval <benchmark>` | Run benchmark evaluation on a model                |
| `bench view`             | View logs from previous benchmark runs             |
| `bench eval <path>`      | Run your local/private evals built with Inspect AI |

### Key `eval` Command Options

| Option               | Environment Variable     | Default                                          | Description                                      |
|----------------------|--------------------------|--------------------------------------------------|--------------------------------------------------|
| `--model`            | `BENCH_MODEL`            | `groq/meta-llama/llama-4-scout-17b-16e-instruct` | Model(s) to evaluate                             |
| `--epochs`           | `BENCH_EPOCHS`           | `1`                                              | Number of epochs to run each evaluation          |
| `--max-connections`  | `BENCH_MAX_CONNECTIONS`  | `10`                                             | Maximum parallel requests to model               |
| `--temperature`      | `BENCH_TEMPERATURE`      | `0.6`                                            | Model temperature                                |
| `--top-p`            | `BENCH_TOP_P`            | `1.0`                                            | Model top-p                                      |
| `--max-tokens`       | `BENCH_MAX_TOKENS`       | `None`                                           | Maximum tokens for model response                |
| `--seed`             | `BENCH_SEED`             | `None`                                           | Seed for deterministic generation                |
| `--limit`            | `BENCH_LIMIT`            | `None`                                           | Limit evaluated samples (number or start,end)    |
| `--logfile`          | `BENCH_OUTPUT`           | `None`                                           | Output file for results                          |
| `--sandbox`          | `BENCH_SANDBOX`          | `None`                                           | Environment to run evaluation (local/docker)     |
| `--timeout`          | `BENCH_TIMEOUT`          | `10000`                                          | Timeout for each API request (seconds)           |
| `--display`          | `BENCH_DISPLAY`          | `None`                                           | Display type (full/conversation/rich/plain/none) |
| `--reasoning-effort` | `BENCH_REASONING_EFFORT` | `None`                                           | Reasoning effort level (low/medium/high)         |
| `--json`             | None                     | `False`                                          | Output results in JSON format                    |

## Building Custom Evaluations

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/). Leverage Inspect AI's documentation to create your custom evaluations. Integrate them into OpenBench with `bench eval <path>` to run your private evals.

## FAQ

### How does OpenBench differ from Inspect AI?

OpenBench provides:

*   **Reference implementations** of 20+ major benchmarks with consistent interfaces.
*   **Shared utilities** for common patterns (math scoring, multi-language support, etc.).
*   **Curated scorers** that work across different eval types.
*   **CLI tooling** optimized for running standardized benchmarks.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

OpenBench focuses on:

*   **Shared components**: Common scorers, solvers, and datasets across benchmarks to reduce code duplication.
*   **Clean implementations**: Each eval is written for readability and reliability.
*   **Developer experience**: Simple CLI, consistent patterns, and easy extensibility.

### How can I run `bench` outside of the `uv` environment?

Run the following command:

```bash
uv run pip install -e .
```

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

Define the environment variable:

```bash
HF_TOKEN="<HUGGINGFACE_TOKEN>"
```

For full HuggingFace documentation, see the [HuggingFace docs on Authentication](https://huggingface.co/docs/hub/en/datasets-polars-auth).

## Development

```bash
# Clone the repo
git clone https://github.com/groq/openbench.git
cd openbench

# Setup with UV
uv venv && uv sync --dev
source .venv/bin/activate

# Run tests
pytest
```

## Contributing

Contribute to OpenBench!  Open issues and submit PRs at [github.com/groq/openbench](https://github.com/groq/openbench).

## Reproducibility Statement

We strive to implement evaluations as faithfully as possible. Note that there might be numerical discrepancies between OpenBench and external sources due to variations in model prompts, quantization, and inference approaches. Ensure you use the same OpenBench version for meaningful comparisons.  We welcome contributions to improve this tool.

## Acknowledgments

*   [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)
*   [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
*   [Hugging Face's lighteval](https://github.com/huggingface/lighteval)

## Citation

```bibtex
@software{openbench,
  title = {OpenBench: Open-source Evaluation Infrastructure for Language Models},
  author = {Sah, Aarush and {Groq Team}},
  year = {2025},
  url = {https://github.com/groq/openbench}
}
```

## License

MIT

---

Built with ❤️ by [Aarush Sah](https://github.com/AarushSah) and the [Groq](https://groq.com) team