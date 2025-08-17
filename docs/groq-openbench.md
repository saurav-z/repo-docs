# OpenBench: Evaluate LLMs Across Providers with Open-Source Benchmarking

**Unlock standardized, reproducible evaluations for your Language Models across 30+ providers with OpenBench – your all-in-one solution for LLM benchmarking.** ([Original Repo](https://github.com/groq/openbench))

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench provides a comprehensive, provider-agnostic, and open-source infrastructure for evaluating Large Language Models (LLMs). It supports over 30 model providers, including Groq, OpenAI, Anthropic, and local models, making it easy to benchmark and compare LLMs. With a growing library of 35+ evaluation suites, covering knowledge, math, reasoning, coding, and more, OpenBench offers the tools you need to understand your LLM's performance.

## Key Features

*   **Provider-Agnostic:** Seamlessly evaluate models from 30+ providers.
*   **Extensive Benchmarks:** Access 35+ benchmarks across various domains, including MMLU, HumanEval, and more.
*   **Simple CLI:** Utilize the `bench` CLI for easy evaluation, listing, and result viewing.
*   **Local Eval Support:** Run private evaluations with built-in support via Inspect AI.
*   **Hugging Face Integration:** Push results directly to Hugging Face datasets for sharing and analysis.
*   **Reproducible Results:** Ensure consistent evaluations with a standardized framework.

## Getting Started in 60 Seconds

Evaluate any model in minutes with a few quick steps:

```bash
# Install OpenBench (30 seconds)
uv venv
source .venv/bin/activate
uv pip install openbench

# Set your API key (any provider!)
export GROQ_API_KEY=your_key  # or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# Run your first evaluation (30 seconds)
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10

# View your results
bench view
```

## Supported Model Providers

OpenBench integrates with a wide range of model providers. Simply set the appropriate API key environment variable to get started:

| Provider          | Environment Variable | Example Model String        |
| ----------------- | --------------------- | --------------------------- |
| AI21 Labs         | `AI21_API_KEY`         | `ai21/model-name`           |
| Anthropic         | `ANTHROPIC_API_KEY`   | `anthropic/model-name`      |
| AWS Bedrock       | AWS credentials       | `bedrock/model-name`        |
| Azure             | `AZURE_OPENAI_API_KEY` | `azure/<deployment-name>`   |
| Baseten           | `BASETEN_API_KEY`      | `baseten/model-name`        |
| Cerebras          | `CEREBRAS_API_KEY`     | `cerebras/model-name`       |
| Cohere            | `COHERE_API_KEY`       | `cohere/model-name`         |
| Crusoe            | `CRUSOE_API_KEY`       | `crusoe/model-name`         |
| DeepInfra         | `DEEPINFRA_API_KEY`    | `deepinfra/model-name`      |
| Friendli          | `FRIENDLI_TOKEN`       | `friendli/model-name`       |
| Google            | `GOOGLE_API_KEY`       | `google/model-name`         |
| Groq              | `GROQ_API_KEY`         | `groq/model-name`           |
| Hugging Face      | `HF_TOKEN`             | `huggingface/model-name`    |
| Hyperbolic        | `HYPERBOLIC_API_KEY`   | `hyperbolic/model-name`     |
| Lambda            | `LAMBDA_API_KEY`       | `lambda/model-name`         |
| MiniMax           | `MINIMAX_API_KEY`      | `minimax/model-name`        |
| Mistral           | `MISTRAL_API_KEY`      | `mistral/model-name`        |
| Moonshot          | `MOONSHOT_API_KEY`     | `moonshot/model-name`       |
| Nebius            | `NEBIUS_API_KEY`       | `nebius/model-name`         |
| Nous Research     | `NOUS_API_KEY`         | `nous/model-name`           |
| Novita AI         | `NOVITA_API_KEY`       | `novita/model-name`         |
| Ollama            | None (local)          | `ollama/model-name`         |
| OpenAI            | `OPENAI_API_KEY`       | `openai/model-name`         |
| OpenRouter        | `OPENROUTER_API_KEY`   | `openrouter/model-name`     |
| Parasail          | `PARASAIL_API_KEY`     | `parasail/model-name`       |
| Perplexity        | `PERPLEXITY_API_KEY`   | `perplexity/model-name`     |
| Reka              | `REKA_API_KEY`         | `reka/model-name`           |
| SambaNova         | `SAMBANOVA_API_KEY`    | `sambanova/model-name`      |
| Together AI       | `TOGETHER_API_KEY`     | `together/model-name`       |
| vLLM              | None (local)          | `vllm/model-name`           |

## Available Benchmarks

OpenBench offers a diverse selection of benchmarks across various categories:

| Category      | Benchmarks                                                                                                                                                                         |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Knowledge     | MMLU (57 subjects), GPQA, SuperGPQA, OpenBookQA, HLE, HLE_text |
| Coding        | HumanEval                                                                                                                                                                          |
| Math          | AIME, HMMT, BRUMO, MATH, MATH-500, MGSM, MGSM_en, MGSM_latin, MGSM_non_latin                                                                                                        |
| Reasoning     | SimpleQA, MuSR, DROP                                                                                                                                                              |
| Long Context  | OpenAI MRCR, OpenAI MRCR_2n, OpenAI MRCR_4, OpenAI MRCR_8n                                                                                                                          |
| Healthcare    | HealthBench, HealthBench_hard, HealthBench_consensus                                                                                                                              |

For the most up-to-date list, use the command: `bench list`

## Command Line Interface

OpenBench provides a straightforward CLI for managing your evaluations.

### Core Commands

| Command                  | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `bench` or `openbench`   | Show main menu                                           |
| `bench list`             | List available evaluations, models, and flags            |
| `bench eval <benchmark>` | Run benchmark evaluation on a model                      |
| `bench eval-retry`       | Retry a failed evaluation                                |
| `bench view`             | View logs from previous benchmark runs                   |
| `bench eval <path>`      | Run local/private evals built with Inspect AI          |

### Key `eval` Command Options

| Option               | Environment Variable     | Default                  | Description                                                  |
| -------------------- | ------------------------ | ------------------------ | ------------------------------------------------------------ |
| `-M <args>`          | None                     | None                     | Pass model-specific arguments (e.g., `-M reasoning_effort=high`) |
| `-T <args>`          | None                     | None                     | Pass task-specific arguments to the benchmark                  |
| `--model`            | `BENCH_MODEL`            | None (required)          | Model(s) to evaluate                                         |
| `--epochs`           | `BENCH_EPOCHS`           | `1`                      | Number of epochs to run each evaluation                       |
| `--max-connections`  | `BENCH_MAX_CONNECTIONS`  | `10`                     | Maximum parallel requests to model                             |
| `--temperature`      | `BENCH_TEMPERATURE`      | `0.6`                    | Model temperature                                             |
| `--top-p`            | `BENCH_TOP_P`            | `1.0`                    | Model top-p                                                   |
| `--max-tokens`       | `BENCH_MAX_TOKENS`       | `None`                   | Maximum tokens for model response                             |
| `--seed`             | `BENCH_SEED`             | `None`                   | Seed for deterministic generation                             |
| `--limit`            | `BENCH_LIMIT`            | `None`                   | Limit evaluated samples (number or start,end)                 |
| `--logfile`          | `BENCH_OUTPUT`           | `None`                   | Output file for results                                       |
| `--sandbox`          | `BENCH_SANDBOX`          | `None`                   | Environment to run evaluation (local/docker)                  |
| `--timeout`          | `BENCH_TIMEOUT`          | `10000`                  | Timeout for each API request (seconds)                        |
| `--display`          | `BENCH_DISPLAY`          | `None`                   | Display type (full/conversation/rich/plain/none)            |
| `--reasoning-effort` | `BENCH_REASONING_EFFORT` | `None`                   | Reasoning effort level (low/medium/high)                      |
| `--json`             | None                     | `False`                  | Output results in JSON format                                  |
| `--hub-repo`         | `BENCH_HUB_REPO`         | `None`                   | Push results to a Hugging Face Hub dataset                     |

## Building Custom Evaluations

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/). Customize your evaluation process to best fit your use case. Create custom evaluations using Inspect AI's documentation. Then, run these with OpenBench using `bench eval <path>`.

## Exporting Logs to Hugging Face

Share and analyze your results by exporting logs directly to a Hugging Face Hub dataset.

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## Frequently Asked Questions (FAQ)

### How does OpenBench differ from Inspect AI?

OpenBench offers:

*   Reference implementations of 20+ major benchmarks with consistent interfaces.
*   Shared utilities for common patterns.
*   Curated scorers.
*   CLI tooling optimized for running standardized benchmarks.

Think of OpenBench as a benchmark library built on Inspect AI's strong foundation.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

OpenBench focuses on:

*   Shared components.
*   Clean implementations.
*   Developer experience (simple CLI, consistent patterns).

### How can I run `bench` outside of the `uv` environment?

Run this command: `uv run pip install -e .`

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

Define the environment variable:  `HF_TOKEN="<HUGGINGFACE_TOKEN>"`

## Development

To contribute to OpenBench:

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

We welcome contributions! Please submit issues and pull requests at [github.com/groq/openbench](https://github.com/groq/openbench).

## Reproducibility

Ensure consistent results by using the same version of OpenBench.

## Acknowledgements

We would like to thank:

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