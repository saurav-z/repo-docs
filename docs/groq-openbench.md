# OpenBench: The Open-Source LLM Evaluation Powerhouse

**Tired of fragmented LLM benchmarking? OpenBench offers a comprehensive, provider-agnostic solution for standardized and reproducible language model evaluation.**

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[View the original repository on GitHub](https://github.com/groq/openbench)

OpenBench is your all-in-one toolkit for evaluating Large Language Models (LLMs). It provides a standardized and reproducible benchmarking infrastructure, supporting a wide range of models and evaluation suites. With OpenBench, you can easily compare models, explore their strengths and weaknesses, and ensure consistent and reliable results across different providers.

## Key Features

*   **✅ Comprehensive Benchmarks:** Evaluate your models across 30+ benchmarks, including MMLU, HumanEval, GPQA, and more, covering knowledge, math, reasoning, and reading comprehension.
*   **🌐 Provider-Agnostic:** Seamlessly works with 15+ model providers out of the box, including Groq, OpenAI, Anthropic, Google, and local models.
*   **⚙️ Simple CLI:** Easily manage your evaluations with intuitive commands like `bench list`, `bench describe`, and `bench eval`.
*   **🛠️ Local Eval Support:** Run private or custom evaluations with the `bench eval <path>` command, built using the Inspect AI framework, preserving data privacy.
*   **📊 Extensible:** Easily add new benchmarks and metrics to meet your specific needs.
*   **🚀 Built on Inspect AI:** Leverages the industry-standard evaluation framework for robust and reliable results.

## Getting Started: Evaluate a Model in Under a Minute

**Prerequisites:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

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

## Supported Model Providers

OpenBench offers flexibility in choosing your model provider:

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

## Benchmark Categories

OpenBench supports a wide range of benchmarks, categorized for easy selection:

| Category        | Benchmarks                                                                                                                                                                                                                                                                 |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Knowledge**   | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version)                                                                                   |
| **Coding**      | HumanEval (164 problems)                                                                                                                                                                                                                                                  |
| **Math**        | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages)                                       |
| **Reasoning**   | SimpleQA (factuality), MuSR (multi-step reasoning)                                                                                                                                                                                                                         |
| **Long Context** | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                                                                                                                                  |
| **Healthcare**  | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant)                                                                                                                                           |

## Configuration and Commands

### Configuration

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/llama-3.1-70b
```

### Commands

For a full list of commands and options, run `bench --help`.

| Command                  | Description                                        |
|--------------------------|----------------------------------------------------|
| `bench`                  | Show main menu with available commands             |
| `bench list`             | List available evaluations, models, and flags      |
| `bench eval <benchmark>` | Run benchmark evaluation on a model                |
| `bench view`             | View logs from previous benchmark runs             |
| `bench eval <path>`      | Run your local/private evals built with Inspect AI |

### `eval` Command Options

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
| `--hub-repo`         | `BENCH_HUB_REPO`         | `None`                                           | Push results to a Hugging Face Hub dataset                    |

## Building Your Own Evals

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/). You can create custom evaluations using their framework. After building your private evaluations, use `bench eval <path>` to run them within OpenBench.

## Exporting Logs to Hugging Face

Share your results and collaborate with the community by exporting logs to a Hugging Face Hub dataset:

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs 
```

## FAQ

### How does OpenBench differ from Inspect AI?

OpenBench enhances the Inspect AI framework with:
*   **Pre-built benchmarks:** Ready-to-use evaluations with consistent interfaces.
*   **Shared utilities:** Common scoring and dataset handling across benchmarks.
*   **Curated scorers:** Optimized for different evaluation types.
*   **User-friendly CLI:** Simplified benchmarking workflows.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

OpenBench provides:
*   **Focus on usability**: OpenBench is optimized for ease of use and extension, making it easy for developers to run tests.
*   **Improved DX:** Simplifies the process of running existing benchmarks.
*   **Focus on collaboration:** Making it easy for others to replicate results.

### How can I run `bench` outside of the `uv` environment?

```bash
uv run pip install -e .
```

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

```bash
HF_TOKEN="<HUGGINGFACE_TOKEN>"
```

## Development

For development, clone the repository, set up with UV, and run tests:

```bash
git clone https://github.com/groq/openbench.git
cd openbench
uv venv && uv sync --dev
source .venv/bin/activate
pytest
```

## Contributing

Contribute to OpenBench by opening issues and pull requests at [github.com/groq/openbench](https://github.com/groq/openbench).

## Reproducibility Statement

OpenBench strives for faithfulness in implementing evaluations but acknowledges potential numerical discrepancies. Results should be compared within the same OpenBench version.

## Acknowledgments

Special thanks to:

*   **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)**
*   **[EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**
*   **[Hugging Face's lighteval](https://github.com/huggingface/lighteval)**

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