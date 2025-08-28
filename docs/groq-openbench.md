# OpenBench: The Ultimate LLM Evaluation Toolkit 🚀

**Tired of limited model evaluation and vendor lock-in?** OpenBench is a **provider-agnostic, open-source** evaluation infrastructure designed to empower you with standardized, reproducible benchmarking for Language Models (LLMs). 

[Get Started with OpenBench](https://github.com/groq/openbench)

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench allows you to effortlessly evaluate your models across 30+ providers and a vast array of benchmarks spanning knowledge, math, reasoning, coding, and more!  **Focus on your research, not the infrastructure.**

**Key Features:**

*   **Provider-Agnostic:** Works with 30+ model providers, including Groq, OpenAI, Anthropic, Google, and local models via Ollama and Hugging Face.
*   **35+ Benchmarks:** Test your LLMs on diverse benchmarks like MMLU, HumanEval, GPQA, and more.
*   **Simple CLI:** Easily run, manage, and view evaluations with the intuitive `bench` CLI.
*   **Extensible:** Seamlessly add new benchmarks and metrics to customize your evaluations.
*   **Local Eval Support:** Run private benchmarks with ease.
*   **Hugging Face Integration:** Directly push evaluation results to Hugging Face datasets for collaboration.

## Getting Started in 60 Seconds

**Prerequisite:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# 1. Create a virtual environment and install OpenBench (30 seconds)
uv venv
source .venv/bin/activate
uv pip install openbench

# 2. Set your API key (any provider!)
export GROQ_API_KEY=your_key  # or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# 3. Run your first eval (30 seconds)
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10

# Done! 🎉  Check results in ./logs/ or visualize them:
bench view
```

## Evaluate with Any Provider

OpenBench supports many different model providers!

```bash
# Groq (Blazing Fast!)
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

# 30+ providers supported - See list below!
```

## Supported Model Providers

OpenBench seamlessly integrates with 30+ model providers through Inspect AI. Configure your environment variables and start evaluating!

| Provider        | Environment Variable     | Example Model String         |
| --------------- | ------------------------ | ---------------------------- |
| AI21 Labs       | `AI21_API_KEY`          | `ai21/model-name`            |
| Anthropic       | `ANTHROPIC_API_KEY`       | `anthropic/model-name`       |
| AWS Bedrock     | AWS credentials          | `bedrock/model-name`         |
| Azure           | `AZURE_OPENAI_API_KEY`    | `azure/<deployment-name>`    |
| Baseten         | `BASETEN_API_KEY`         | `baseten/model-name`         |
| Cerebras        | `CEREBRAS_API_KEY`        | `cerebras/model-name`        |
| Cohere          | `COHERE_API_KEY`          | `cohere/model-name`          |
| Crusoe          | `CRUSOE_API_KEY`          | `crusoe/model-name`          |
| DeepInfra       | `DEEPINFRA_API_KEY`       | `deepinfra/model-name`       |
| Friendli        | `FRIENDLI_TOKEN`          | `friendli/model-name`        |
| Google          | `GOOGLE_API_KEY`          | `google/model-name`          |
| Groq            | `GROQ_API_KEY`            | `groq/model-name`            |
| Hugging Face    | `HF_TOKEN`              | `huggingface/model-name`     |
| Hyperbolic      | `HYPERBOLIC_API_KEY`      | `hyperbolic/model-name`      |
| Lambda          | `LAMBDA_API_KEY`          | `lambda/model-name`          |
| MiniMax         | `MINIMAX_API_KEY`         | `minimax/model-name`         |
| Mistral         | `MISTRAL_API_KEY`         | `mistral/model-name`         |
| Moonshot        | `MOONSHOT_API_KEY`        | `moonshot/model-name`        |
| Nebius          | `NEBIUS_API_KEY`          | `nebius/model-name`          |
| Nous Research   | `NOUS_API_KEY`            | `nous/model-name`            |
| Novita AI       | `NOVITA_API_KEY`          | `novita/model-name`          |
| Ollama          | None (local)             | `ollama/model-name`          |
| OpenAI          | `OPENAI_API_KEY`          | `openai/model-name`          |
| OpenRouter      | `OPENROUTER_API_KEY`      | `openrouter/model-name`      |
| Parasail        | `PARASAIL_API_KEY`        | `parasail/model-name`        |
| Perplexity      | `PERPLEXITY_API_KEY`      | `perplexity/model-name`      |
| Reka            | `REKA_API_KEY`            | `reka/model-name`            |
| SambaNova       | `SAMBANOVA_API_KEY`       | `sambanova/model-name`       |
| Together AI     | `TOGETHER_API_KEY`        | `together/model-name`        |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY`    | `vercel/creator-name/model-name` |
| vLLM            | None (local)             | `vllm/model-name`            |

## Available Benchmarks

Find an up-to-date list using `bench list`.

| Category        | Benchmarks                                                                                                                                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Knowledge       | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version)                                                               |
| Coding          | HumanEval (164 problems)                                                                                                                                                                                                                      |
| Math            | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages)                                 |
| Reasoning       | SimpleQA (factuality), MuSR (multi-step reasoning), DROP (discrete reasoning over paragraphs)                                                                                                                                                 |
| Long Context    | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                                                                                                        |
| Healthcare      | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant)                                                                                                                   |
| Cybersecurity   | CTI-Bench (complete cyber threat intelligence suite), CTI-Bench ATE (MITRE ATT&CK technique extraction), CTI-Bench MCQ (knowledge questions on CTI standards and best practices), CTI-Bench RCM (CVE to CWE vulnerability mapping), CTI-Bench VSP (CVSS score calculation) |

## Configuration

```bash
# Set API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/llama-3.1-70b
```

## CLI Commands and Options

For a comprehensive list of commands and options, use: `bench --help`

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

## Building Custom Evaluations

OpenBench leverages [Inspect AI](https://inspect.aisi.org.uk/) for its robust evaluation capabilities. For custom evaluations, refer to the [Inspect AI documentation](https://inspect.aisi.org.uk/). After building your private evaluations with Inspect AI, run them using `bench eval <path>`.

## Exporting Logs to Hugging Face

Share and analyze your results by exporting logs to a Hugging Face Hub dataset.

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

This saves your logs to a Hugging Face Hub dataset named `openbench-logs`.

## Frequently Asked Questions (FAQ)

### How does OpenBench differ from Inspect AI?

OpenBench provides:

*   Reference implementations for 20+ major benchmarks
*   Shared utilities for common patterns
*   Curated scorers for diverse eval types
*   A CLI optimized for running standardized benchmarks

Essentially, OpenBench is a benchmark library built on Inspect AI.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

Each tool serves a purpose. OpenBench focuses on:

*   Shared components to minimize code duplication.
*   Readable and reliable implementations.
*   A simple CLI and easy extensibility.

We built OpenBench for our specific evaluation needs. It provides a curated set of benchmarks on the Inspect AI foundation.

### How can I run `bench` outside of the `uv` environment?

Run this command after setting up your environment:

```bash
uv run pip install -e .
```

### I'm running into issues downloading a dataset from Hugging Face. How do I fix it?

Set the environment variable with your Hugging Face token:

```bash
HF_TOKEN="<HUGGINGFACE_TOKEN>"
```

Refer to the [Hugging Face authentication documentation](https://huggingface.co/docs/hub/en/datasets-polars-auth) for further details.

## Development

Clone the repository:

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

**IMPORTANT:** Run `pre-commit install` after setup to prevent CI failures.

## Contributing

Contributions are welcome! See our [Contributing Guide](CONTRIBUTING.md) for details on:

*   Setting up the development environment
*   Adding benchmarks and model providers
*   Code style and testing
*   Submitting issues and pull requests

Quick links:

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

OpenBench's evaluations strive for accuracy, but expect numerical differences due to minor variations in model prompts, quantization, and repurposing benchmarks for OpenBench.

OpenBench results are meant to be compared with other OpenBench results. Always ensure you are using the same OpenBench version.

We welcome improvements.

## Acknowledgments

Thanks to:

*   **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)** for the evaluation framework.
*   **[EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** for pioneering work.
*   **[Hugging Face's lighteval](https://github.com/huggingface/lighteval)** for the excellent infrastructure.

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