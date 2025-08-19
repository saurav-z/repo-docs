# OpenBench: The Open-Source LLM Evaluation Powerhouse 🚀

**Tired of vendor lock-in when evaluating language models?** OpenBench is a provider-agnostic, open-source evaluation infrastructure that lets you benchmark LLMs across 35+ standardized suites, supporting 30+ providers and your own private evaluations. [Explore OpenBench on GitHub](https://github.com/groq/openbench).

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench empowers you to rigorously evaluate LLMs from any provider, including Groq, OpenAI, Anthropic, and local models, ensuring standardized, reproducible results. This alpha release features rapid iteration and robust support for various tasks like knowledge, math, and reasoning.

## Key Features

*   **Provider-Agnostic:** Seamlessly benchmark models from 30+ providers.
*   **35+ Benchmarks:** Comprehensive suite covering knowledge, math, coding, and more.
*   **Easy-to-Use CLI:** Simple commands for listing, describing, and running evaluations.
*   **Local Eval Support:** Privacy-focused evaluation with support for your own custom benchmarks.
*   **Hugging Face Integration:** Push results directly to Hugging Face datasets.
*   **Extensible:** Easily add new benchmarks and metrics.

## What's New

*   **Expanded Provider Support:** Added 18 more model providers, including AI21, Baseten, Cohere, DeepInfra, and more.
*   **New Benchmarks:** Introduced DROP (reading comprehension) and experimental benchmarks.
*   **CLI Enhancements:** Added `openbench` alias, model/task argument flags, and a `--debug` mode.
*   **Developer Tools:** GitHub Actions integration and Inspect AI extension support.

## Get Started in 60 Seconds

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

## Evaluating Models from Different Providers

OpenBench supports various providers, making it easy to compare model performance.

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

# 30+ providers supported - see full list below
```

## Supported Model Providers

Configure OpenBench by setting the appropriate API key environment variable for your chosen provider:

| Provider          | Environment Variable    | Example Model String        |
|-------------------|-------------------------|-----------------------------|
| AI21 Labs         | `AI21_API_KEY`          | `ai21/model-name`           |
| Anthropic         | `ANTHROPIC_API_KEY`     | `anthropic/model-name`      |
| AWS Bedrock       | AWS credentials         | `bedrock/model-name`        |
| Azure             | `AZURE_OPENAI_API_KEY`  | `azure/<deployment-name>`   |
| Baseten           | `BASETEN_API_KEY`       | `baseten/model-name`        |
| Cerebras          | `CEREBRAS_API_KEY`      | `cerebras/model-name`       |
| Cohere            | `COHERE_API_KEY`        | `cohere/model-name`         |
| Crusoe            | `CRUSOE_API_KEY`        | `crusoe/model-name`         |
| DeepInfra         | `DEEPINFRA_API_KEY`     | `deepinfra/model-name`      |
| Friendli          | `FRIENDLI_TOKEN`        | `friendli/model-name`       |
| Google            | `GOOGLE_API_KEY`        | `google/model-name`         |
| Groq              | `GROQ_API_KEY`          | `groq/model-name`           |
| Hugging Face      | `HF_TOKEN`              | `huggingface/model-name`    |
| Hyperbolic        | `HYPERBOLIC_API_KEY`    | `hyperbolic/model-name`     |
| Lambda            | `LAMBDA_API_KEY`        | `lambda/model-name`         |
| MiniMax           | `MINIMAX_API_KEY`       | `minimax/model-name`        |
| Mistral           | `MISTRAL_API_KEY`       | `mistral/model-name`        |
| Moonshot          | `MOONSHOT_API_KEY`      | `moonshot/model-name`       |
| Nebius            | `NEBIUS_API_KEY`        | `nebius/model-name`         |
| Nous Research     | `NOUS_API_KEY`          | `nous/model-name`           |
| Novita AI         | `NOVITA_API_KEY`        | `novita/model-name`         |
| Ollama            | None (local)            | `ollama/model-name`         |
| OpenAI            | `OPENAI_API_KEY`        | `openai/model-name`         |
| OpenRouter        | `OPENROUTER_API_KEY`    | `openrouter/model-name`     |
| Parasail          | `PARASAIL_API_KEY`      | `parasail/model-name`       |
| Perplexity        | `PERPLEXITY_API_KEY`    | `perplexity/model-name`     |
| Reka              | `REKA_API_KEY`          | `reka/model-name`           |
| SambaNova         | `SAMBANOVA_API_KEY`     | `sambanova/model-name`      |
| Together AI       | `TOGETHER_API_KEY`      | `together/model-name`       |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY`    | `vercel/creator-name/model-name` |
| vLLM              | None (local)            | `vllm/model-name`           |

## Available Benchmarks

Use `bench list` for an up-to-date list.

| Category          | Benchmarks                                                                                                                                                              |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Knowledge         | MMLU (57 subjects), GPQA, SuperGPQA, OpenBookQA, HLE, HLE_text                                                                                                           |
| Coding            | HumanEval                                                                                                                                                               |
| Math              | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH, MATH-500, MGSM (multilingual grade school math)                                                                        |
| Reasoning         | SimpleQA, MuSR, DROP                                                                                                                                                   |
| Long Context      | OpenAI MRCR, OpenAI MRCR_2n, OpenAI MRCR_4, OpenAI MRCR_8n                                                                                                              |
| Healthcare        | HealthBench, HealthBench_hard, HealthBench_consensus                                                                                                                     |

## Configuration

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/llama-3.1-70b
```

## Commands and Options

Run `bench --help` for a complete list of commands.

| Command                  | Description                                        |
|--------------------------|----------------------------------------------------|
| `bench` or `openbench`   | Show main menu with available commands             |
| `bench list`             | List evaluations, models, and flags               |
| `bench eval <benchmark>` | Run benchmark evaluation on a model                |
| `bench eval-retry`       | Retry a failed evaluation                          |
| `bench view`             | View logs from previous benchmark runs             |
| `bench eval <path>`      | Run your local/private evals with Inspect AI        |

### Key `eval` Command Options

| Option               | Environment Variable     | Default                                          | Description                                      |
|----------------------|--------------------------|--------------------------------------------------|--------------------------------------------------|
| `-M <args>`          | None                     | None                                             | Pass model-specific arguments                    |
| `-T <args>`          | None                     | None                                             | Pass task-specific arguments                     |
| `--model`            | `BENCH_MODEL`            | None (required)                                 | Model(s) to evaluate                             |
| `--epochs`           | `BENCH_EPOCHS`           | `1`                                              | Number of epochs                                 |
| `--max-connections`  | `BENCH_MAX_CONNECTIONS`  | `10`                                             | Maximum parallel requests                       |
| `--temperature`      | `BENCH_TEMPERATURE`      | `0.6`                                            | Model temperature                                |
| `--top-p`            | `BENCH_TOP_P`            | `1.0`                                            | Model top-p                                      |
| `--max-tokens`       | `BENCH_MAX_TOKENS`       | `None`                                           | Maximum tokens for model response                |
| `--seed`             | `BENCH_SEED`             | `None`                                           | Seed for deterministic generation                |
| `--limit`            | `BENCH_LIMIT`            | `None`                                           | Limit evaluated samples                          |
| `--logfile`          | `BENCH_OUTPUT`           | `None`                                           | Output file for results                          |
| `--sandbox`          | `BENCH_SANDBOX`          | `None`                                           | Environment to run evaluation                    |
| `--timeout`          | `BENCH_TIMEOUT`          | `10000`                                          | Timeout for each API request (seconds)           |
| `--display`          | `BENCH_DISPLAY`          | `None`                                           | Display type                                     |
| `--reasoning-effort` | `BENCH_REASONING_EFFORT` | `None`                                           | Reasoning effort level                           |
| `--json`             | None                     | `False`                                          | Output results in JSON format                    |
| `--hub-repo`         | `BENCH_HUB_REPO`         | `None`                                           | Push results to a Hugging Face Hub dataset       |

## Building Your Own Evals

Leverage the power of [Inspect AI](https://inspect.aisi.org.uk/) to create custom evaluations. Once built, use `bench eval <path>` to run your private evaluations.

## Exporting Logs to Hugging Face

Share your results with the community or perform further analysis by exporting logs to a Hugging Face Hub dataset.

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

This will export the logs to a Hugging Face Hub dataset.

## Frequently Asked Questions (FAQ)

### How does OpenBench differ from Inspect AI?

OpenBench is a benchmark library built on Inspect AI, offering:

*   Reference implementations of 20+ major benchmarks with consistent interfaces.
*   Shared utilities for common patterns.
*   Curated scorers.
*   CLI tooling optimized for running standardized benchmarks.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

OpenBench focuses on:

*   Shared components to reduce code duplication.
*   Clean implementations for readability.
*   Developer experience with a simple CLI.

### How can I run `bench` outside of the `uv` environment?

Run `uv run pip install -e .`

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

Define the environment variable `HF_TOKEN="<HUGGINGFACE_TOKEN>"`

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

⚠️ **IMPORTANT**: Run `pre-commit install` after setup!

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details, including adding benchmarks, code style, and submitting PRs.

Quick links:

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

OpenBench strives to implement evaluations faithfully. Numerical discrepancies are possible due to variations in model prompts, quantization, etc. Compare results within OpenBench.

## Acknowledgments

Special thanks to:

*   **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)**
*   **[EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**
*   **[Hugging Face's lighteval](https://github.com/huggingface/lighteval)**

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

---

Built with ❤️ by [Aarush Sah](https://github.com/AarushSah) and the [Groq](https://groq.com) team