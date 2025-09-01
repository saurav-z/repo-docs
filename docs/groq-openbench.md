# OpenBench: The Ultimate LLM Evaluation Toolkit 🚀

**Unlock the power of comprehensive LLM benchmarking with OpenBench, a provider-agnostic, open-source evaluation infrastructure.** Easily compare and evaluate language models across 35+ benchmarks with support for 30+ providers including Groq, OpenAI, Anthropic, and more! [Explore OpenBench on GitHub](https://github.com/groq/openbench).

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Key Features

*   **Provider-Agnostic:** Works seamlessly with 30+ model providers (Groq, OpenAI, Anthropic, and many more) through Inspect AI.
*   **Comprehensive Benchmarks:** Access 35+ standardized benchmarks spanning knowledge, reasoning, coding, math, and more (MMLU, HumanEval, GPQA, etc.).
*   **Simple CLI:** Easy-to-use command-line interface for listing, describing, and running evaluations.
*   **Local Eval Support:**  Run private evaluations built with Inspect AI with `bench eval <path>`.
*   **Extensible:** Easily add new benchmarks and metrics.
*   **Hugging Face Integration:** Push results directly to Hugging Face datasets for sharing and analysis.

## What's New in v0.3.0

*   **Expanded Provider Support:** AI21, Baseten, Cerebras, Cohere, Crusoe, DeepInfra, Friendli, Hugging Face, Hyperbolic, Lambda, MiniMax, Moonshot, Nebius, Nous, Novita, Parasail, Reka, SambaNova and more.
*   **New Benchmarks:** DROP (reading comprehension), with experimental benchmarks available via the `--alpha` flag.
*   **CLI Enhancements:** `openbench` alias, `-M`/`-T` flags for model/task arguments, `--debug` mode for retry.
*   **Developer Tools:** GitHub Actions integration, Inspect AI extension support.

## Quick Start: Evaluate a Model in 60 Seconds

Get started with these easy steps:

1.  **Install (30 seconds):**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install openbench
    ```
2.  **Set API Key:**  (Example for Groq)
    ```bash
    export GROQ_API_KEY=your_key  # Or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
    ```
3.  **Run Your First Eval (30 seconds):**
    ```bash
    bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10
    ```

    Check results in `./logs/` or with the interactive UI using `bench view`.  That's it! 🎉

## Model Provider Support

OpenBench supports a vast array of model providers. Ensure you set the correct environment variable for your chosen provider:

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

OpenBench offers a wide selection of benchmarks to evaluate different aspects of LLM performance.

> [!NOTE]
> Benchmark names are case-sensitive in the CLI.

| Category          | Benchmarks                                                                                                                                                                                                                                                                 |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge**     | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version)                                                                                         |
| **Coding**        | HumanEval (164 problems)                                                                                                                                                                                                                                                   |
| **Math**          | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages)                                         |
| **Reasoning**     | SimpleQA (factuality), MuSR (multi-step reasoning), DROP (discrete reasoning over paragraphs), MMMU (multi-modal reasoning with 30+ subjects)                                                                                                                              |
| **Long Context**  | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                                                                                                                                    |
| **Healthcare**    | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant)                                                                                                                                                |
| **Cybersecurity** | CTI-Bench (complete cyber threat intelligence suite), CTI-Bench ATE (MITRE ATT&CK technique extraction), CTI-Bench MCQ (knowledge questions on CTI standards and best practices), CTI-Bench RCM (CVE to CWE vulnerability mapping), CTI-Bench VSP (CVSS score calculation) |

## Configuration

Configure OpenBench with your API keys and set a default model to streamline your evaluations.

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/openai/gpt-oss-20b
```

## Commands and Options

Explore the powerful `bench` CLI for all your evaluation needs.  Run `bench --help` for a full list.

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

Certain benchmarks utilize a grader model for scoring.  Ensure you have your OpenAI API key configured to run these:

```bash
export OPENAI_API_KEY=your_openai_key
```

The following benchmarks leverage a grader:

| Benchmark | Default Grader Model |
| :--- | :--- |
| `simpleqa` | `openai/gpt-4.1-2025-04-14` |
| `hle` | `openai/o3-mini-2025-01-31` |
| `hle_text` | `openai/o3-mini-2025-01-31` |
| `browsecomp` | `openai/gpt-4.1-2025-04-14` |
| `healthbench` | `openai/gpt-4.1-2025-04-14` |
| `math` | `openai/gpt-4-turbo-preview` |
| `math_500` | `openai/gpt-4-turbo-preview` |

## Building Your Own Evaluations

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/). Leverage Inspect AI to create custom evaluations and then integrate them seamlessly into OpenBench: `bench eval <path>`.

## Exporting Logs to Hugging Face

Share your results with the community by pushing logs directly to a Hugging Face Hub dataset.

```bash
export HF_TOKEN=<your-huggingface-token>
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

This will export results to the Hub dataset with the name `openbench-logs`.

## Frequently Asked Questions (FAQ)

**Q: How does OpenBench differ from Inspect AI?**

*   OpenBench provides:
    *   Reference implementations of 20+ benchmarks with a consistent interface.
    *   Shared utilities for common patterns.
    *   Curated scorers that work across different eval types.
    *   CLI tooling optimized for standardized benchmarking.

**Q: Why not use Inspect AI, lm-evaluation-harness, or lighteval?**

*   OpenBench focuses on:
    *   Shared components for reduced code duplication.
    *   Clean, readable implementations.
    *   Developer experience with a simple CLI and easy extensibility.

**Q: How do I run `bench` outside of the `uv` environment?**

*   Run `uv run pip install -e .`

**Q: I'm getting errors downloading datasets from Hugging Face, how do I fix this?**

*   Set the `HF_TOKEN` environment variable with your Hugging Face token.

## Development

Get started with OpenBench development.

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

**Important:**  Ensure `pre-commit install` is run after setup.  CI will fail otherwise.

## Contributing

We welcome your contributions.  See our [Contributing Guide](CONTRIBUTING.md) for setup, adding benchmarks, code style, and submitting pull requests.

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

OpenBench aims for faithful implementations. However, slight score discrepancies may occur due to differences in model prompts, quantization, or inference approaches.  Compare OpenBench results with OpenBench results; using the same OpenBench version is crucial.  We welcome improvements!

## Acknowledgments

*   [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) - Evaluation framework
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