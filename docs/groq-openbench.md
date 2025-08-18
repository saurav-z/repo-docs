# OpenBench: Benchmarking Language Models with Ease

**Quickly and reliably evaluate any language model, from any provider, with OpenBench—your open-source evaluation infrastructure.** ([Original Repo](https://github.com/groq/openbench))

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench provides a standardized and reproducible benchmarking framework for Large Language Models (LLMs). It supports 30+ evaluation suites, including knowledge, math, reasoning, coding, science, and long-context recall. It works seamlessly with all major model providers like Groq, OpenAI, Anthropic, Google, and local models via Ollama, ensuring easy, provider-agnostic evaluation.  Keep your evaluations private and secure with first-class support for local evaluations.

**Key Features:**

*   **Extensive Benchmark Suite:** Includes 35+ benchmarks like MMLU, HumanEval, and competition math.
*   **Provider-Agnostic:** Supports 30+ model providers, including Groq, OpenAI, Anthropic, and local models.
*   **Simple CLI:** Easy-to-use command-line interface for listing, describing, and running evaluations.
*   **Local Eval Support:** Run privatized benchmarks easily with `bench eval <path>`.
*   **Hugging Face Integration:** Push evaluation results directly to Hugging Face datasets.
*   **Extensible:** Easily add new benchmarks and metrics.

## Getting Started: Evaluate in Seconds

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

## Provider Support

OpenBench seamlessly integrates with various model providers. Simply set your API key and you're ready to evaluate:

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

# 30+ providers supported
```

### Supported Providers

| Provider            | Environment Variable | Example Model String              |
|---------------------|---------------------|-----------------------------------|
| AI21 Labs           | `AI21_API_KEY`      | `ai21/model-name`                 |
| Anthropic           | `ANTHROPIC_API_KEY` | `anthropic/model-name`            |
| AWS Bedrock         | AWS credentials     | `bedrock/model-name`              |
| Azure               | `AZURE_OPENAI_API_KEY` | `azure/<deployment-name>`       |
| Baseten             | `BASETEN_API_KEY`   | `baseten/model-name`              |
| Cerebras            | `CEREBRAS_API_KEY`  | `cerebras/model-name`             |
| Cohere              | `COHERE_API_KEY`    | `cohere/model-name`               |
| Crusoe              | `CRUSOE_API_KEY`    | `crusoe/model-name`               |
| DeepInfra           | `DEEPINFRA_API_KEY` | `deepinfra/model-name`            |
| Friendli            | `FRIENDLI_TOKEN`    | `friendli/model-name`             |
| Google              | `GOOGLE_API_KEY`    | `google/model-name`               |
| Groq                | `GROQ_API_KEY`      | `groq/model-name`                 |
| Hugging Face        | `HF_TOKEN`          | `huggingface/model-name`          |
| Hyperbolic          | `HYPERBOLIC_API_KEY`| `hyperbolic/model-name`           |
| Lambda              | `LAMBDA_API_KEY`    | `lambda/model-name`               |
| MiniMax             | `MINIMAX_API_KEY`   | `minimax/model-name`              |
| Mistral             | `MISTRAL_API_KEY`   | `mistral/model-name`              |
| Moonshot            | `MOONSHOT_API_KEY`  | `moonshot/model-name`             |
| Nebius              | `NEBIUS_API_KEY`    | `nebius/model-name`               |
| Nous Research       | `NOUS_API_KEY`      | `nous/model-name`                 |
| Novita AI           | `NOVITA_API_KEY`    | `novita/model-name`               |
| Ollama              | None (local)        | `ollama/model-name`               |
| OpenAI              | `OPENAI_API_KEY`    | `openai/model-name`               |
| OpenRouter          | `OPENROUTER_API_KEY`| `openrouter/model-name`           |
| Parasail            | `PARASAIL_API_KEY`  | `parasail/model-name`             |
| Perplexity          | `PERPLEXITY_API_KEY`| `perplexity/model-name`           |
| Reka                | `REKA_API_KEY`      | `reka/model-name`                 |
| SambaNova           | `SAMBANOVA_API_KEY` | `sambanova/model-name`            |
| Together AI         | `TOGETHER_API_KEY`  | `together/model-name`             |
| Vercel AI Gateway   | `AI_GATEWAY_API_KEY`| `vercel/creator-name/model-name` |
| vLLM                | None (local)        | `vllm/model-name`                 |

## Available Benchmarks

Use `bench list` to see the most up-to-date list of available evaluations.

| Category         | Benchmarks                                                                                                                                   |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Knowledge        | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version) |
| Coding           | HumanEval (164 problems)                                                                                                                  |
| Math             | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages) |
| Reasoning        | SimpleQA (factuality), MuSR (multi-step reasoning), DROP (discrete reasoning over paragraphs)                                                |
| Long Context     | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                           |
| Healthcare       | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant) |

## Configuration

Set API keys and configure defaults:

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/llama-3.1-70b
```

## Command-Line Interface

For a full list of commands and options, use `bench --help`.

| Command                  | Description                                        |
|--------------------------|----------------------------------------------------|
| `bench` or `openbench`   | Show main menu with available commands             |
| `bench list`             | List available evaluations, models, and flags      |
| `bench eval <benchmark>` | Run benchmark evaluation on a model                |
| `bench eval-retry`       | Retry a failed evaluation                          |
| `bench view`             | View logs from previous benchmark runs             |
| `bench eval <path>`      | Run your local/private evals built with Inspect AI |

### Key `eval` Command Options

| Option               | Environment Variable     | Default                                          | Description                                      |
|----------------------|--------------------------|--------------------------------------------------|--------------------------------------------------|
| `-M <args>`          | None                     | None                                             | Pass model-specific arguments (e.g., `-M reasoning_effort=high`) |
| `-T <args>`          | None                     | None                                             | Pass task-specific arguments to the benchmark   |
| `--model`            | `BENCH_MODEL`            | None (required)                                 | Model(s) to evaluate                             |
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
| `--hub-repo`         | `BENCH_HUB_REPO`         | `None`                                           | Push results to a Hugging Face Hub dataset                    |

## Building Your Own Evals

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/). Use Inspect AI to create custom evaluations; once you do, you can point OpenBench at them with `bench eval <path>` to run!

## Exporting Logs to Hugging Face

Easily export your OpenBench results to a Hugging Face Hub dataset:

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs 
```

## FAQ

### How does OpenBench differ from Inspect AI?

OpenBench provides:

*   Reference implementations of major benchmarks with consistent interfaces.
*   Shared utilities for common patterns (math scoring, multi-language support, etc.).
*   Curated scorers that work across different eval types.
*   CLI tooling optimized for running standardized benchmarks.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

OpenBench focuses on:

*   Shared components: Common scorers, solvers, and datasets across benchmarks reduce code duplication.
*   Clean implementations: Each eval is written for readability and reliability.
*   Developer experience: Simple CLI, consistent patterns, easy to extend.

### How can I run `bench` outside of the `uv` environment?

Run:

```bash
uv run pip install -e .
```

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

If you see "gated" errors, set:

```bash
HF_TOKEN="<HUGGINGFACE_TOKEN>"
```

## Development

```bash
# Clone the repo
git clone https://github.com/groq/openbench.git
cd openbench

# Setup with UV
uv venv && uv sync --dev
source .venv/bin/activate

# CRITICAL: Install pre-commit hooks (CI will fail without this!)
pre-commit install

# Run tests
pytest
```

## Contributing

See our [Contributing Guide](CONTRIBUTING.md) for details.

Quick links:

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

OpenBench is designed for faithful benchmark implementations. However, small numerical differences are possible due to various factors. Ensure you are using the same version of OpenBench for meaningful comparisons.

## Acknowledgments

*   [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)
*   [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
*   [Hugging Face's lighteval](https://github.com/huggingface/lighteval)

## Citation

```bibtex
@software{openbench,
  title = {OpenBench: Provider-agnostic, open-source evaluation infrastructure for language models},
  author = {Sah, Aarush},
  year = {2025},
  url = {https://openbench.dev}
}
```

## License

MIT