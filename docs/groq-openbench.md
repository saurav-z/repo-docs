# OpenBench: Evaluate LLMs Reliably and Reproducibly

**OpenBench provides a unified, open-source platform for benchmarking language models, supporting over 30 providers, and built on Inspect AI.** ([See the original repository](https://github.com/groq/openbench))

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench is a comprehensive solution for evaluating Large Language Models (LLMs) across a wide range of benchmarks, offering standardized, reproducible results for informed decision-making. It supports an ever-growing list of model providers, including Groq, OpenAI, Anthropic, and local models, making it easy to compare performance across different architectures.

## Key Features

*   **Provider-Agnostic:** Seamlessly evaluate models from 30+ providers with consistent interfaces.
*   **Extensive Benchmark Suite:** Access over 35 benchmarks covering knowledge, math, reasoning, coding, and more.
*   **Easy-to-Use CLI:** Simplify model evaluation with a straightforward command-line interface.
*   **Local Eval Support:** Maintain privacy and customize benchmarks by running your own local evaluations.
*   **Hugging Face Integration:** Push results to Hugging Face datasets for sharing and analysis.
*   **Extensible:** Add your own custom benchmarks and metrics with ease.
*   **Built on Inspect AI:** Leveraging an industry-standard evaluation framework.

## What's New

*   **Expanded Provider Support:** Integration with AI21, Baseten, Cerebras, Cohere, Crusoe, DeepInfra, Friendli, Hugging Face, Hyperbolic, Lambda, MiniMax, Moonshot, Nebius, Nous, Novita, Parasail, Reka, SambaNova, and more!
*   **New Benchmarks:** Added DROP (reading comprehension) and experimental benchmarks available with the `--alpha` flag.
*   **CLI Enhancements:** `openbench` alias, `-M`/`-T` flags for model/task arguments, and `--debug` mode for improved evaluation retries.
*   **Developer Tools:** Improved support for GitHub Actions, plus integration with the Inspect AI extension.

## Quickstart: Evaluate a Model in 60 Seconds

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

## Using Different Providers

OpenBench simplifies evaluating models from diverse providers.  Below are examples of how to specify your model:

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

## Supported Providers

OpenBench supports a wide array of model providers via Inspect AI.  Simply set the appropriate API key environment variable.

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

OpenBench offers a broad selection of benchmarks categorized below. To get an up-to-date list of available evaluations, use the `bench list` command.

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
| **Community** | DetailBench                                     |

## Configuration

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/openai/gpt-oss-20b
```

## Commands and Options

For a comprehensive list of commands and options, use `bench --help`.

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

Some benchmarks utilize a grader model to evaluate model performance, which requires an additional API key.

To run these benchmarks, export your `OPENAI_API_key`:

```bash
export OPENAI_API_KEY=your_openai_key
```

The following benchmarks rely on a grader model:

| Benchmark | Default Grader Model |
| :--- | :--- |
| `simpleqa` | `openai/gpt-4.1-2025-04-14` |
| `hle` | `openai/o3-mini-2025-01-31` |
| `hle_text` | `openai/o3-mini-2025-01-31` |
| `browsecomp` | `openai/gpt-4.1-2025-04-14` |
| `healthbench` | `openai/gpt-4.1-2025-04-14` |
| `math` | `openai/gpt-4-turbo-preview` |
| `math_500` | `openai/gpt-4-turbo-preview` |
| `detailbench` | `gpt-5-mini-2025-08-07` |

## Building Your Own Evals

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/).  To develop custom evaluations, consult the [Inspect AI documentation](https://inspect.aisi.org.uk/).  You can then run your private evaluations using `bench eval <path>`.

## Exporting Logs to Hugging Face

OpenBench allows exporting your evaluation logs to a Hugging Face Hub dataset, enabling sharing and further analysis.

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

This exports logs to a Hugging Face Hub dataset named `openbench-logs`.

## FAQ

### How does OpenBench differ from Inspect AI?

OpenBench enhances the Inspect AI framework with:

*   **Reference Implementations:** Consistent interfaces for 20+ major benchmarks.
*   **Shared Utilities:** Common components like math scoring and multi-language support.
*   **Curated Scorers:**  Designed to work across various evaluation types.
*   **CLI Tooling:** Optimized for standardized benchmark execution.

Think of OpenBench as a benchmark library built on top of Inspect AI.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

OpenBench is tailored to:

*   **Shared Components:** Reduces code duplication by providing common scorers, solvers, and datasets.
*   **Clean Implementations:** Focuses on readable and reliable code for each evaluation.
*   **Developer Experience:** Provides a simple CLI, consistent patterns, and ease of extension.

OpenBench provides curated benchmarks built on Inspect AI's powerful foundation.

### How can I run `bench` outside of the `uv` environment?

Use the following command to make `bench` available outside of the `uv` environment:

```bash
uv run pip install -e .
```

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

If dataset downloads fail, or you encounter "gated" errors, set the environment variable:

```bash
HF_TOKEN="<HUGGINGFACE_TOKEN>"
```

Refer to the [Hugging Face documentation on Authentication](https://huggingface.co/docs/hub/en/datasets-polars-auth) for detailed information.

## Development

To develop OpenBench, clone the repository:

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

⚠️ **IMPORTANT**:  Ensure you run `pre-commit install` after setup to avoid CI failures.

## Contributing

Contributions are welcome!  Refer to the [Contributing Guide](CONTRIBUTING.md) for details:

*   Setting up the development environment
*   Adding new benchmarks and model providers
*   Code style and testing requirements
*   Submitting issues and pull requests

Quick links:

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

OpenBench aims to provide faithful implementations of benchmark evaluations.

Expect minor numerical variations between OpenBench's results and those from other sources.

Variations may stem from model prompts, quantization, inference, and adaptations for OpenBench's packages.

Therefore, compare OpenBench results with OpenBench results, not as a direct comparison to all external sources. Use the same version of OpenBench for meaningful comparisons.

We encourage improvements and contributions.

## Acknowledgments

OpenBench is made possible by:

*   **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)** - The core evaluation framework.
*   **[EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** - Pioneering work in LLM evaluation.
*   **[Hugging Face's lighteval](https://github.com/huggingface/lighteval)** - Excellent evaluation infrastructure.

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