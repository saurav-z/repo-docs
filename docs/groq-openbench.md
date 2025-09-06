# OpenBench: Benchmark LLMs with Ease Across Any Provider

**Tired of vendor lock-in when evaluating your language models?** OpenBench is a provider-agnostic, open-source evaluation framework that simplifies LLM benchmarking, offering standardized, reproducible results across 30+ providers. [Explore the OpenBench Repository](https://github.com/groq/openbench)

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench provides a comprehensive suite of benchmarks and tools, letting you compare models across various domains like knowledge, math, reasoning, and coding.  Whether you're using Groq, OpenAI, Anthropic, or a local model, OpenBench offers a unified experience.  It also includes first-class support for custom, private evaluations, allowing you to maintain data privacy.

## Key Features

*   **Extensive Benchmark Suite:** Access 35+ benchmarks including MMLU, HumanEval, and various math and reasoning challenges.
*   **Provider-Agnostic:** Supports 30+ model providers including Groq, OpenAI, Anthropic, and local models via Ollama and Hugging Face.
*   **Simple CLI:**  Easily evaluate models with intuitive commands like `bench eval`, `bench list`, and `bench view`.
*   **Hugging Face Integration:**  Push results directly to Hugging Face datasets for sharing and analysis.
*   **Custom Evaluation Support:**  Integrate your private evaluations built with Inspect AI.
*   **Developer-Friendly:**  Easy to extend with new benchmarks and metrics.

## What's New

*   **Expanded Provider Support:** Added support for AI21, Baseten, Cerebras, Cohere, Crusoe, DeepInfra, Friendli, Hugging Face, Hyperbolic, Lambda, MiniMax, Moonshot, Nebius, Nous, Novita, Parasail, Reka, SambaNova, and more.
*   **New Benchmarks:** Added the DROP reading comprehension benchmark.
*   **CLI Enhancements:** Improved the CLI for easier use.
*   **Developer Tools:** GitHub Actions integration and Inspect AI extension support.

## Quickstart: Evaluate a Model in 60 Seconds

Get up and running quickly with a minimal setup:

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

## Provider-Specific Examples

OpenBench seamlessly supports a wide range of model providers:

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

## Supported Providers and API Keys

Choose your provider and set the appropriate environment variable:

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

OpenBench offers a wide range of benchmarks categorized by domain.  Use `bench list` for the most up-to-date list.

> [!NOTE]
> Benchmark names are case-sensitive in the CLI.

| Category          | Benchmarks                                                                                                                                                                                                                                                                                                      |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge**     | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version)                                                                                                                              |
| **Coding**        | HumanEval (164 problems)                                                                                                                                                                                                                                                                                        |
| **Math**          | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages)                                                                              |
| **Reasoning**     | SimpleQA (factuality), MuSR (multi-step reasoning), DROP (discrete reasoning over paragraphs), MMMU (multi-modal reasoning with 30+ subjects), MMMU_MCQ (multiple choice version), MMMU_OPEN (open answer version), MMMU_PRO (more rigorous version of mmmu), MMMU_PRO_VISION (vision only version of mmmu_pro) |
| **Long Context**  | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                                                                                                                                                                         |
| **Healthcare**    | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant)                                                                                                                                                                                     |
| **Cybersecurity** | CTI-Bench (complete cyber threat intelligence suite), CTI-Bench ATE (MITRE ATT&CK technique extraction), CTI-Bench MCQ (knowledge questions on CTI standards and best practices), CTI-Bench RCM (CVE to CWE vulnerability mapping), CTI-Bench VSP (CVSS score calculation)                                      |

## Configuration Options

Customize your evaluations with environment variables and command-line options:

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/openai/gpt-oss-20b
```

## Command-Line Interface (CLI)

Use the `bench` command to manage your evaluations.  Run `bench --help` for a complete guide.

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

Some benchmarks use a grader model for scoring.  Set your `OPENAI_API_KEY` for those evaluations:

```bash
export OPENAI_API_KEY=your_openai_key
```

These benchmarks use a grader model:

| Benchmark | Default Grader Model |
| :--- | :--- |
| `simpleqa` | `openai/gpt-4.1-2025-04-14` |
| `hle` | `openai/o3-mini-2025-01-31` |
| `hle_text` | `openai/o3-mini-2025-01-31` |
| `browsecomp` | `openai/gpt-4.1-2025-04-14` |
| `healthbench` | `openai/gpt-4.1-2025-04-14` |
| `math` | `openai/gpt-4-turbo-preview` |
| `math_500` | `openai/gpt-4-turbo-preview` |

## Building Custom Evaluations

Leverage the power of [Inspect AI](https://inspect.aisi.org.uk/) to create your own evaluations. OpenBench allows you to run your custom Inspect AI-built evaluations using `bench eval <path>`.

## Exporting Logs to Hugging Face

Share your results and collaborate by exporting logs to a Hugging Face Hub dataset:

```bash
export HF_TOKEN=<your-huggingface-token>
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## Frequently Asked Questions (FAQ)

*   **How does OpenBench differ from Inspect AI?** OpenBench provides curated benchmarks, shared utilities, and a user-friendly CLI built on top of the Inspect AI framework.
*   **Why not just use Inspect AI, lm-evaluation-harness, or lighteval?** OpenBench offers shared components, clean implementations, a simple CLI, and a focus on developer experience.
*   **How can I run `bench` outside of the `uv` environment?** Run `uv run pip install -e .`.
*   **I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?** Define the environment variable `HF_TOKEN="<HUGGINGFACE_TOKEN>"` to authenticate with Hugging Face.

## Development

Follow these steps to set up the development environment:

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

Contribute to OpenBench! Check out the [Contributing Guide](CONTRIBUTING.md) for instructions.

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility

We aim to implement benchmarks faithfully.  However, expect slight numerical differences due to model variations and package differences.  Compare results within OpenBench versions.

## Acknowledgments

Thanks to:

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