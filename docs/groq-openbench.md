# OpenBench: Evaluate LLMs with Provider-Agnostic Benchmarking

**OpenBench provides a comprehensive and open-source framework for evaluating Language Models (LLMs) across various providers and benchmarks, fostering reproducible and standardized assessments.** [Visit the GitHub Repository](https://github.com/groq/openbench)

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench offers a unified evaluation environment supporting **30+ model providers** including Groq, OpenAI, Anthropic, Google, and local models via Ollama and Hugging Face. This allows you to benchmark your models across a wide array of tasks and providers.  It includes over **35 benchmarks** covering areas like knowledge, reasoning, coding, and more.  OpenBench provides first-class support for running your own private evaluations, ensuring the privacy of your data.

## Key Features

*   **Provider-Agnostic:** Seamlessly works with 30+ model providers.
*   **Extensive Benchmarks:** Supports 35+ benchmarks across diverse domains (knowledge, math, reasoning, etc.).
*   **Simple CLI:** Intuitive command-line interface for easy evaluation.
*   **Local Evaluation Support:**  Privately evaluate models.
*   **Hugging Face Integration:**  Push results directly to Hugging Face datasets.
*   **Extensible:** Easily add new benchmarks and metrics.
*   **Reproducible:**  Provides a standardized and reliable evaluation process.

## What's New in v0.3.0

*   **Expanded Provider Support:** Integration for AI21, Baseten, Cerebras, Cohere, Crusoe, DeepInfra, Friendli, Hugging Face, Hyperbolic, Lambda, MiniMax, Moonshot, Nebius, Nous, Novita, Parasail, Reka, SambaNova and more
*   **New Benchmarks:** Includes the DROP (reading comprehension) benchmark.  Experimental benchmarks can be accessed with the `--alpha` flag.
*   **CLI Enhancements:** Offers the `openbench` alias, along with `-M`/`-T` flags for specifying model/task arguments, and a `--debug` mode for evaluation retries.
*   **Developer Tools:**  Features GitHub Actions integration and Inspect AI extension support.

## Quick Start: Evaluate a Model in 60 Seconds

**Prerequisite:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

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

## Supported Providers

OpenBench integrates with a wide range of LLM providers through Inspect AI. Set the appropriate API key environment variable to get started:

| Provider          | Environment Variable  | Example Model String         |
|-------------------|-----------------------|------------------------------|
| AI21 Labs       | `AI21_API_KEY`        | `ai21/model-name`            |
| Anthropic         | `ANTHROPIC_API_KEY`   | `anthropic/model-name`       |
| AWS Bedrock       | AWS credentials       | `bedrock/model-name`         |
| Azure             | `AZURE_OPENAI_API_KEY`| `azure/<deployment-name>`    |
| Baseten           | `BASETEN_API_KEY`     | `baseten/model-name`         |
| Cerebras          | `CEREBRAS_API_KEY`    | `cerebras/model-name`        |
| Cohere            | `COHERE_API_KEY`      | `cohere/model-name`          |
| Crusoe            | `CRUSOE_API_KEY`      | `crusoe/model-name`          |
| DeepInfra         | `DEEPINFRA_API_KEY`   | `deepinfra/model-name`       |
| Friendli          | `FRIENDLI_TOKEN`      | `friendli/model-name`        |
| Google            | `GOOGLE_API_KEY`      | `google/model-name`          |
| Groq              | `GROQ_API_KEY`        | `groq/model-name`            |
| Hugging Face      | `HF_TOKEN`            | `huggingface/model-name`     |
| Hyperbolic        | `HYPERBOLIC_API_KEY`  | `hyperbolic/model-name`      |
| Lambda            | `LAMBDA_API_KEY`      | `lambda/model-name`          |
| MiniMax           | `MINIMAX_API_KEY`     | `minimax/model-name`         |
| Mistral           | `MISTRAL_API_KEY`     | `mistral/model-name`         |
| Moonshot          | `MOONSHOT_API_KEY`    | `moonshot/model-name`        |
| Nebius            | `NEBIUS_API_KEY`      | `nebius/model-name`          |
| Nous Research     | `NOUS_API_KEY`        | `nous/model-name`            |
| Novita AI         | `NOVITA_API_KEY`      | `novita/model-name`          |
| Ollama            | None (local)          | `ollama/model-name`          |
| OpenAI            | `OPENAI_API_KEY`      | `openai/model-name`          |
| OpenRouter        | `OPENROUTER_API_KEY`  | `openrouter/model-name`      |
| Parasail          | `PARASAIL_API_KEY`    | `parasail/model-name`        |
| Perplexity        | `PERPLEXITY_API_KEY`  | `perplexity/model-name`      |
| Reka              | `REKA_API_KEY`        | `reka/model-name`            |
| SambaNova         | `SAMBANOVA_API_KEY`   | `sambanova/model-name`       |
| Together AI       | `TOGETHER_API_KEY`    | `together/model-name`        |
| vLLM              | None (local)          | `vllm/model-name`            |

## Available Benchmarks

For a complete and up-to-date list, run `bench list`.

| Category          | Benchmarks                                                                                                                                                                                                                                                         |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Knowledge         | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE\_text (text-only version)                                                                             |
| Coding            | HumanEval (164 problems)                                                                                                                                                                                                                                           |
| Math              | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM\_en (English), MGSM\_latin (5 languages), MGSM\_non\_latin (6 languages)                                      |
| Reasoning         | SimpleQA (factuality), MuSR (multi-step reasoning), DROP (discrete reasoning over paragraphs)                                                                                                                                                                         |
| Long Context      | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR\_2n (2 needle), OpenAI MRCR\_4 (4 needle), OpenAI MRCR\_8n (8 needle)                                                                                                                                                 |
| Healthcare        | HealthBench (open-ended healthcare eval), HealthBench\_hard (challenging variant), HealthBench\_consensus (consensus variant)                                                                                                                                     |

## Commands and Options

Run `bench --help` for detailed information.

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

OpenBench leverages [Inspect AI](https://inspect.aisi.org.uk/).  Consult their [documentation](https://inspect.aisi.org.uk/) for creating custom evaluations.  You can then use OpenBench to execute your custom evaluations via `bench eval <path>`.

## Exporting Logs to Hugging Face

Share your evaluation results by exporting logs to a Hugging Face Hub dataset:

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## Frequently Asked Questions (FAQ)

### How does OpenBench differ from Inspect AI?

OpenBench builds on Inspect AI by providing:

*   Pre-built implementations for 20+ major benchmarks.
*   Shared utilities for common patterns.
*   Curated scorers that work across evaluation types.
*   A CLI optimized for standardized benchmarking.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

OpenBench focuses on:

*   Shared components across benchmarks.
*   Clean and readable implementations.
*   A user-friendly CLI and easy extensibility.

### How can I run `bench` outside of the `uv` environment?

Run: `uv run pip install -e .`

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

Set the `HF_TOKEN` environment variable:

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

# Run tests
pytest
```

## Contributing

Contribute by opening issues and pull requests at [github.com/groq/openbench](https://github.com/groq/openbench).

## Reproducibility Statement

OpenBench strives for faithful implementations; however, numerical discrepancies are possible. For meaningful comparisons, use the same OpenBench version.  Contributions are welcome.

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