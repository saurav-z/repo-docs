# OpenBench: Evaluate LLMs Reliably with Open Source 🚀

**OpenBench** is your all-in-one, open-source solution for evaluating language models, offering standardized and reproducible benchmarking across various providers and tasks. **[Explore the OpenBench Repository](https://github.com/groq/openbench)**

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench provides a comprehensive and versatile platform for benchmarking LLMs. It supports 30+ model providers, including Groq, OpenAI, Anthropic, and local models. This allows you to assess your models on over 35 evaluation suites spanning domains like knowledge, math, reasoning, and coding. With first-class support for your own local evaluations, you can ensure complete privacy while assessing your models.

## Key Features

*   **Provider-Agnostic**: Supports 30+ model providers out-of-the-box.
*   **Extensive Benchmark Suite**: Access over 35 benchmarks, including MMLU, HumanEval, and more.
*   **Simple CLI**: Streamlined command-line interface for easy evaluation.
*   **Local Eval Support**: Run privatized benchmarks with local evaluation capabilities.
*   **Hugging Face Integration**: Push results directly to Hugging Face datasets.
*   **Extensible**: Easily add new benchmarks and metrics.

## Key Updates

*   **Expanded Provider Support**: Now supports AI21, Baseten, Cerebras, Cohere, Crusoe, DeepInfra, Friendli, Hugging Face, Hyperbolic, Lambda, MiniMax, Moonshot, Nebius, Nous, Novita, Parasail, Reka, SambaNova, and more.
*   **New Benchmarks**: Introducing DROP (reading comprehension) and experimental benchmarks available with the `--alpha` flag.
*   **CLI Enhancements**: Includes an `openbench` alias, `-M`/`-T` flags for model/task arguments, and `--debug` mode for evaluation retries.
*   **Developer Tools**: Enhanced with GitHub Actions integration and Inspect AI extension support.

## Get Started: Evaluate a Model in 60 Seconds

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

## Provider Support and Configuration

OpenBench supports various model providers through the Inspect AI framework. Set the appropriate API key environment variable to use your preferred provider:

### Example Provider Usage

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
```

### Supported Providers

| Provider              | Environment Variable   | Example Model String             |
| --------------------- | ---------------------- | -------------------------------- |
| **AI21 Labs**         | `AI21_API_KEY`         | `ai21/model-name`                |
| **Anthropic**         | `ANTHROPIC_API_KEY`    | `anthropic/model-name`           |
| **AWS Bedrock**       | AWS credentials        | `bedrock/model-name`             |
| **Azure**             | `AZURE_OPENAI_API_KEY` | `azure/<deployment-name>`        |
| **Baseten**           | `BASETEN_API_KEY`      | `baseten/model-name`             |
| **Cerebras**          | `CEREBRAS_API_KEY`     | `cerebras/model-name`            |
| **Cohere**            | `COHERE_API_KEY`       | `cohere/model-name`              |
| **Crusoe**            | `CRUSOE_API_KEY`       | `crusoe/model-name`              |
| **DeepInfra**         | `DEEPINFRA_API_KEY`    | `deepinfra/model-name`           |
| **Friendli**          | `FRIENDLI_TOKEN`       | `friendli/model-name`            |
| **Google**            | `GOOGLE_API_KEY`       | `google/model-name`              |
| **Groq**              | `GROQ_API_KEY`         | `groq/model-name`                |
| **Hugging Face**      | `HF_TOKEN`             | `huggingface/model-name`         |
| **Hyperbolic**        | `HYPERBOLIC_API_KEY`   | `hyperbolic/model-name`          |
| **Lambda**            | `LAMBDA_API_KEY`       | `lambda/model-name`              |
| **MiniMax**           | `MINIMAX_API_KEY`      | `minimax/model-name`             |
| **Mistral**           | `MISTRAL_API_KEY`      | `mistral/model-name`             |
| **Moonshot**          | `MOONSHOT_API_KEY`     | `moonshot/model-name`            |
| **Nebius**            | `NEBIUS_API_KEY`       | `nebius/model-name`              |
| **Nous Research**     | `NOUS_API_KEY`         | `nous/model-name`                |
| **Novita AI**         | `NOVITA_API_KEY`       | `novita/model-name`              |
| **Ollama**            | None (local)           | `ollama/model-name`              |
| **OpenAI**            | `OPENAI_API_KEY`       | `openai/model-name`              |
| **OpenRouter**        | `OPENROUTER_API_KEY`   | `openrouter/model-name`          |
| **Parasail**          | `PARASAIL_API_KEY`     | `parasail/model-name`            |
| **Perplexity**        | `PERPLEXITY_API_KEY`   | `perplexity/model-name`          |
| **Reka**              | `REKA_API_KEY`         | `reka/model-name`                |
| **SambaNova**         | `SAMBANOVA_API_KEY`    | `sambanova/model-name`           |
| **Together AI**       | `TOGETHER_API_KEY`     | `together/model-name`            |
| **Vercel AI Gateway** | `AI_GATEWAY_API_KEY`   | `vercel/creator-name/model-name` |
| **vLLM**              | None (local)           | `vllm/model-name`                |

## Available Benchmarks

Access the latest benchmarks using `bench list`. Benchmarks are case-sensitive in the CLI.

### Benchmark Categories

*   **Knowledge**: MMLU, GPQA, SuperGPQA, OpenBookQA, HLE, HLE_text
*   **Coding**: HumanEval
*   **Math**: AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH, MATH-500, MGSM, MGSM_en, MGSM_latin, MGSM_non_latin
*   **Reasoning**: SimpleQA, MuSR, DROP, MMMU, MMMU_MCQ, MMMU_OPEN, MMMU_PRO, MMMU_PRO_VISION
*   **Long Context**: OpenAI MRCR, OpenAI MRCR_2n, OpenAI MRCR_4, OpenAI MRCR_8n
*   **Healthcare**: HealthBench, HealthBench_hard, HealthBench_consensus
*   **Cybersecurity**: CTI-Bench, CTI-Bench ATE, CTI-Bench MCQ, CTI-Bench RCM, CTI-Bench VSP

## Configuration

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/openai/gpt-oss-20b
```

## Command Line Interface (CLI)

Run `bench --help` for a full list of commands.

| Command                  | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `bench` or `openbench`   | Show main menu with available commands             |
| `bench list`             | List available evaluations, models, and flags      |
| `bench eval <benchmark>` | Run benchmark evaluation on a model                |
| `bench eval-retry`       | Retry a failed evaluation                          |
| `bench view`             | View logs from previous benchmark runs             |
| `bench eval <path>`      | Run your local/private evals built with Inspect AI |

### Key `eval` Command Options

| Option               | Environment Variable     | Default         | Description                                                      |
| -------------------- | ------------------------ | --------------- | ---------------------------------------------------------------- |
| `-M <args>`          | None                     | None            | Pass model-specific arguments (e.g., `-M reasoning_effort=high`) |
| `-T <args>`          | None                     | None            | Pass task-specific arguments to the benchmark                    |
| `--model`            | `BENCH_MODEL`            | `groq/openai/gpt-oss-20b` | Model(s) to evaluate                                             |
| `--epochs`           | `BENCH_EPOCHS`           | `1`             | Number of epochs to run each evaluation                          |
| `--max-connections`  | `BENCH_MAX_CONNECTIONS`  | `10`            | Maximum parallel requests to model                               |
| `--temperature`      | `BENCH_TEMPERATURE`      | `0.6`           | Model temperature                                                |
| `--top-p`            | `BENCH_TOP_P`            | `1.0`           | Model top-p                                                      |
| `--max-tokens`       | `BENCH_MAX_TOKENS`       | `None`          | Maximum tokens for model response                                |
| `--seed`             | `BENCH_SEED`             | `None`          | Seed for deterministic generation                                |
| `--limit`            | `BENCH_LIMIT`            | `None`          | Limit evaluated samples (number or start,end)                    |
| `--logfile`          | `BENCH_OUTPUT`           | `None`          | Output file for results                                          |
| `--sandbox`          | `BENCH_SANDBOX`          | `None`          | Environment to run evaluation (local/docker)                     |
| `--timeout`          | `BENCH_TIMEOUT`          | `10000`         | Timeout for each API request (seconds)                           |
| `--display`          | `BENCH_DISPLAY`          | `None`          | Display type (full/conversation/rich/plain/none)                 |
| `--reasoning-effort` | `BENCH_REASONING_EFFORT` | `None`          | Reasoning effort level (low/medium/high)                         |
| `--json`             | None                     | `False`         | Output results in JSON format                                    |
| `--log-format`       | `BENCH_LOG_FORMAT`       | `eval`          | Output logging format (eval/json)                                |
| `--hub-repo`         | `BENCH_HUB_REPO`         | `None`          | Push results to a Hugging Face Hub dataset                       |

## Grader Information

Certain benchmarks require a grader model (e.g. `simpleqa`, `hle`, `math`). To use these, you'll need to set your `OPENAI_API_KEY`.

| Benchmark        | Default Grader Model                  |
| :--------------- | :------------------------------------ |
| `simpleqa`       | `openai/gpt-4.1-2025-04-14`          |
| `hle`            | `openai/o3-mini-2025-01-31`          |
| `hle_text`       | `openai/o3-mini-2025-01-31`          |
| `browsecomp`     | `openai/gpt-4.1-2025-04-14`          |
| `healthbench`    | `openai/gpt-4.1-2025-04-14`          |
| `math`           | `openai/gpt-4-turbo-preview`         |
| `math_500`       | `openai/gpt-4-turbo-preview`         |

## Creating Custom Evaluations

OpenBench is built upon [Inspect AI](https://inspect.aisi.org.uk/). Learn to build custom evaluations from the [Inspect AI documentation](https://inspect.aisi.org.uk/).  Run your private evaluations with OpenBench using `bench eval <path>`.

## Exporting Logs to Hugging Face

Share your results to Hugging Face!

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## FAQ

### How does OpenBench differ from Inspect AI?

OpenBench provides:

*   **Reference implementations** of 20+ major benchmarks with consistent interfaces
*   **Shared utilities** for common patterns (math scoring, multi-language support, etc.)
*   **Curated scorers** that work across different eval types
*   **CLI tooling** optimized for running standardized benchmarks

It’s a benchmark library built on Inspect’s foundation.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

OpenBench focuses on:

*   **Shared components**: Reduce code duplication via shared scorers, solvers, and datasets.
*   **Clean implementations**: Eval code is written for readability and reliability.
*   **Developer experience**: Simple CLI, consistent patterns, easy to extend.

### How can I run `bench` outside of the `uv` environment?

```bash
uv run pip install -e .
```

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

Set your Hugging Face token if prompted:

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

⚠️ **IMPORTANT**: Run `pre-commit install` after setup!

## Contributing

See our [Contributing Guide](CONTRIBUTING.md) for details:

*   Set up the development environment
*   Add new benchmarks and model providers
*   Code style and testing requirements
*   Submit issues and pull requests

Quick links:

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

We aim to implement evaluations as faithfully as possible. Differences may exist, but results are meant to be compared with OpenBench.

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

---

Built with ❤️ by [Aarush Sah](https://github.com/AarushSah) and the [Groq](https://groq.com) team