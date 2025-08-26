# OpenBench: The Open-Source LLM Evaluation Powerhouse

**Unlock standardized, reproducible benchmarking for Language Models (LLMs) across diverse providers!** OpenBench is a provider-agnostic, open-source evaluation infrastructure that empowers you to assess LLMs with ease.

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Tired of vendor lock-in and inconsistent benchmarking? OpenBench offers a streamlined solution. **Evaluate your favorite LLMs across 35+ benchmarks, including knowledge, math, reasoning, and more, with out-of-the-box support for 30+ model providers.** [Explore the OpenBench Repository](https://github.com/groq/openbench).

## Key Features:

*   **Broad Provider Support:** Seamlessly integrates with Groq, OpenAI, Anthropic, Google, AWS Bedrock, Azure, local models (Ollama, Hugging Face), and 30+ other providers.
*   **Extensive Benchmark Suite:** Access 35+ benchmarks, including MMLU, HumanEval, and more, covering a wide range of LLM capabilities.
*   **Simple and Powerful CLI:** Easily run evaluations with the `bench` or `openbench` command, offering intuitive commands and options.
*   **Reproducible Results:** Built on the foundation of [Inspect AI](https://inspect.aisi.org.uk/), ensuring reliable and consistent evaluations.
*   **Extensible Architecture:** Easily add new benchmarks and integrate with custom evaluation pipelines.
*   **Local Evaluation Support:** Run privatized benchmarks with local eval support to preserve your data's privacy.
*   **Hugging Face Integration:** Effortlessly push your evaluation results to Hugging Face datasets for sharing and further analysis.

## Getting Started: Evaluate a Model in Under a Minute!

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

## Model Provider Support:

OpenBench supports a wide range of model providers through Inspect AI.

### Supported Providers:

| Provider           | Environment Variable       | Example Model String        |
| ------------------ | -------------------------- | --------------------------- |
| **AI21 Labs**      | `AI21_API_KEY`            | `ai21/model-name`          |
| **Anthropic**       | `ANTHROPIC_API_KEY`        | `anthropic/model-name`      |
| **AWS Bedrock**     | AWS credentials            | `bedrock/model-name`        |
| **Azure**           | `AZURE_OPENAI_API_KEY`     | `azure/<deployment-name>`  |
| **Baseten**         | `BASETEN_API_KEY`          | `baseten/model-name`        |
| **Cerebras**        | `CEREBRAS_API_KEY`         | `cerebras/model-name`       |
| **Cohere**          | `COHERE_API_KEY`           | `cohere/model-name`         |
| **Crusoe**          | `CRUSOE_API_KEY`           | `crusoe/model-name`         |
| **DeepInfra**       | `DEEPINFRA_API_KEY`        | `deepinfra/model-name`      |
| **Friendli**        | `FRIENDLI_TOKEN`           | `friendli/model-name`       |
| **Google**          | `GOOGLE_API_KEY`           | `google/model-name`         |
| **Groq**            | `GROQ_API_KEY`             | `groq/model-name`           |
| **Hugging Face**    | `HF_TOKEN`                 | `huggingface/model-name`    |
| **Hyperbolic**      | `HYPERBOLIC_API_KEY`       | `hyperbolic/model-name`     |
| **Lambda**          | `LAMBDA_API_KEY`           | `lambda/model-name`         |
| **MiniMax**         | `MINIMAX_API_KEY`          | `minimax/model-name`        |
| **Mistral**         | `MISTRAL_API_KEY`          | `mistral/model-name`        |
| **Moonshot**        | `MOONSHOT_API_KEY`         | `moonshot/model-name`       |
| **Nebius**          | `NEBIUS_API_KEY`           | `nebius/model-name`         |
| **Nous Research**   | `NOUS_API_KEY`             | `nous/model-name`           |
| **Novita AI**       | `NOVITA_API_KEY`           | `novita/model-name`         |
| **Ollama**          | None (local)              | `ollama/model-name`         |
| **OpenAI**          | `OPENAI_API_KEY`           | `openai/model-name`         |
| **OpenRouter**      | `OPENROUTER_API_KEY`       | `openrouter/model-name`     |
| **Parasail**        | `PARASAIL_API_KEY`         | `parasail/model-name`       |
| **Perplexity**      | `PERPLEXITY_API_KEY`       | `perplexity/model-name`     |
| **Reka**            | `REKA_API_KEY`             | `reka/model-name`           |
| **SambaNova**       | `SAMBANOVA_API_KEY`        | `sambanova/model-name`      |
| **Together AI**     | `TOGETHER_API_KEY`         | `together/model-name`       |
| **Vercel AI Gateway** | `AI_GATEWAY_API_KEY`      | `vercel/creator-name/model-name` |
| **vLLM**            | None (local)              | `vllm/model-name`           |

## Available Benchmarks:

For a full and up-to-date list, run `bench list`.

| Category         | Benchmarks                                                                                                                                                                 |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge**    | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version) |
| **Coding**       | HumanEval (164 problems)                                                                                                                                                   |
| **Math**         | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages) |
| **Reasoning**    | SimpleQA (factuality), MuSR (multi-step reasoning), DROP (discrete reasoning over paragraphs)                                                                             |
| **Long Context** | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                                     |
| **Healthcare**   | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant)                                              |
| **Cybersecurity** | CTI-Bench (complete cyber threat intelligence suite), CTI-Bench ATE (MITRE ATT&CK technique extraction), CTI-Bench MCQ (knowledge questions on CTI standards and best practices), CTI-Bench RCM (CVE to CWE vulnerability mapping), CTI-Bench VSP (CVSS score calculation) |

## CLI Commands and Options:

Use `bench --help` for detailed information on available commands and options.

| Command                   | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `bench` or `openbench`    | Show main menu with available commands                 |
| `bench list`              | List available evaluations, models, and flags          |
| `bench eval <benchmark>`  | Run benchmark evaluation on a model                    |
| `bench eval-retry`        | Retry a failed evaluation                              |
| `bench view`              | View logs from previous benchmark runs                 |
| `bench eval <path>`       | Run your local/private evals built with Inspect AI     |

### Key `eval` Command Options:

| Option               | Environment Variable     | Default                                          | Description                                      |
| -------------------- | -------------------------- | -------------------------------------------------- | -------------------------------------------------- |
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

## Building Your Own Evals:

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/). Create custom evaluations using their documentation.  You can then integrate your private benchmarks with OpenBench via `bench eval <path>`.

## Exporting Logs to Hugging Face:

Share your evaluation results with the community by exporting logs to a Hugging Face Hub dataset.

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

This example exports logs to the `openbench-logs` dataset in your Hugging Face Hub.

## FAQ:

### How does OpenBench differ from Inspect AI?

OpenBench provides:
- **Reference implementations** of 20+ major benchmarks with consistent interfaces
- **Shared utilities** for common patterns (math scoring, multi-language support, etc.)
- **Curated scorers** that work across different eval types
- **CLI tooling** optimized for running standardized benchmarks

Think of it as a benchmark library built on Inspect's excellent foundation.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

Different tools for different needs! OpenBench focuses on:

- **Shared components**: Common scorers, solvers, and datasets across benchmarks reduce code duplication
- **Clean implementations**: Each eval is written for readability and reliability
- **Developer experience**: Simple CLI, consistent patterns, easy to extend

We built OpenBench because we needed evaluation code that was easy to understand, modify, and trust. It's a curated set of benchmarks built on Inspect AI's excellent foundation.

### How can I run `bench` outside of the `uv` environment?

Run the following command:

```bash
uv run pip install -e .
```

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

Define the environment variable:

```bash
HF_TOKEN="<HUGGINGFACE_TOKEN>"
```

For more information, consult the [HuggingFace documentation](https://huggingface.co/docs/hub/en/datasets-polars-auth).

## Development:

*Clone the repository and set up the development environment.*

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

**Important:** Run `pre-commit install` after setup.

## Contributing:

See our [Contributing Guide](CONTRIBUTING.md) for detailed instructions.

Quick links:
- [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
- [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement:

OpenBench aims for faithful benchmark implementations, but numerical discrepancies may occur. For meaningful comparisons, use the same OpenBench version. Contributions are welcomed.

## Acknowledgments:

*   [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)
*   [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
*   [Hugging Face's lighteval](https://github.com/huggingface/lighteval)

## Citation:

```bibtex
@software{openbench,
  title = {OpenBench: Provider-agnostic, open-source evaluation infrastructure for language models},
  author = {Sah, Aarush},
  year = {2025},
  url = {https://openbench.dev}
}
```

## License:

MIT

---

Built with ❤️ by [Aarush Sah](https://github.com/AarushSah) and the [Groq](https://groq.com) team