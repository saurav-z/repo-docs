# OpenBench: The Ultimate LLM Benchmarking Toolkit

**Evaluate any language model, from any provider, with OpenBench – your all-in-one solution for standardized and reproducible LLM benchmarking.**  ([Original Repository](https://github.com/groq/openbench))

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench is a powerful, open-source framework designed to benchmark Large Language Models (LLMs) with unmatched flexibility. It supports 30+ model providers, including Groq, OpenAI, Anthropic, Google, and local models, providing a comprehensive suite of evaluation tools.  This allows you to compare performance across various LLMs using consistent and reliable metrics.

**Key Features:**

*   **Provider-Agnostic:**  Seamlessly evaluate models from diverse providers with simple API key configurations.
*   **35+ Benchmarks:** Access a comprehensive selection of benchmarks spanning knowledge, math, reasoning, coding, and more. Includes support for MMLU, GPQA, HumanEval, and many more.
*   **Simple CLI:**  Effortlessly run evaluations with user-friendly commands like `bench eval`, `bench list`, and `bench view`.
*   **Extensible:** Easily add new benchmarks, metrics, and model providers to customize your evaluation process.
*   **Local Evaluation Support:** Preserve your privacy by running evaluations with your own local models.
*   **Hugging Face Integration:** Push your evaluation results directly to Hugging Face datasets for easy sharing and analysis.
*   **Built on Inspect AI:** Leverage the industry-standard evaluation framework for robust and reliable results.

## What's New

*   **Expanded Provider Support:** Added support for 18 additional model providers, including AI21, Cerebras, Cohere, DeepInfra and more.
*   **New Benchmarks:** Introduced new benchmarks, such as DROP for reading comprehension.
*   **CLI Enhancements:** Improved CLI functionality with features like aliases and debugging mode.
*   **Developer Tools:** Enhanced development workflow with GitHub Actions integration and Inspect AI extension support.

## Getting Started

Evaluate your first model in under a minute with these simple steps:

```bash
# Create a virtual environment and install OpenBench
uv venv
source .venv/bin/activate
uv pip install openbench

# Set your API key (e.g., GROQ, OpenAI, etc.)
export GROQ_API_KEY=your_key

# Run your first evaluation
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10

# View your results
bench view
```

## Provider Configuration

OpenBench offers wide support for various model providers. Set the corresponding API key environment variable to get started:

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

OpenBench supports a wide range of benchmarks, categorized for different evaluation needs:

| Category          | Benchmarks                                                                                                                                                                                                                                                                                                      |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge**     | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version)                                                                                                                              |
| **Coding**        | HumanEval (164 problems)                                                                                                                                                                                                                                                                                        |
| **Math**          | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages)                                                                              |
| **Reasoning**     | SimpleQA (factuality), MuSR (multi-step reasoning), DROP (discrete reasoning over paragraphs), MMMU (multi-modal reasoning with 30+ subjects), MMMU_MCQ (multiple choice version), MMMU_OPEN (open answer version), MMMU_PRO (more rigorous version of mmmu), MMMU_PRO_VISION (vision only version of mmmu_pro) |
| **Long Context**  | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                                                                                                                                                                         |
| **Healthcare**    | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant)                                                                                                                                                                                     |
| **Cybersecurity** | CTI-Bench (complete cyber threat intelligence suite), CTI-Bench ATE (MITRE ATT&CK technique extraction), CTI-Bench MCQ (knowledge questions on CTI standards and best practices), CTI-Bench RCM (CVE to CWE vulnerability mapping), CTI-Bench VSP (CVSS score calculation)                                      |

For a complete and up-to-date list, use the `bench list` command.

## Commands and Options

Explore the core functionalities of OpenBench using the CLI:

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

Some benchmarks require a grader model. You'll need an `OPENAI_API_KEY` for those.

```bash
export OPENAI_API_KEY=your_openai_key
```

The following benchmarks use a grader model:

| Benchmark | Default Grader Model |
| :--- | :--- |
| `simpleqa` | `openai/gpt-4.1-2025-04-14` |
| `hle` | `openai/o3-mini-2025-01-31` |
| `hle_text` | `openai/o3-mini-2025-01-31` |
| `browsecomp` | `openai/gpt-4.1-2025-04-14` |
| `healthbench` | `openai/gpt-4.1-2025-04-14` |
| `math` | `openai/gpt-4-turbo-preview` |
| `math_500` | `openai/gpt-4-turbo-preview` |

## Custom Evaluations

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/). Create custom evaluations and run them with OpenBench:  `bench eval <path>`.

## Exporting Logs

Export results to a Hugging Face Hub dataset:

```bash
export HF_TOKEN=<your-huggingface-token>
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## FAQ

**Q: How does OpenBench differ from Inspect AI?**

*   OpenBench offers reference implementations, shared utilities, curated scorers, and a user-friendly CLI on top of Inspect AI.

**Q: Why not use other evaluation tools?**

*   OpenBench focuses on shared components, clean implementations, and a great developer experience.

**Q: How can I run `bench` outside the `uv` environment?**

```bash
uv run pip install -e .
```

**Q: I'm having issues with Hugging Face dataset downloads.**

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

## Contributing

Contribute to OpenBench! See our [Contributing Guide](CONTRIBUTING.md).

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility and Acknowledgments

OpenBench aims for faithful implementations.  Results are best compared within OpenBench versions.

Thanks to [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and [Hugging Face's lighteval](https://github.com/huggingface/lighteval).

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

Built with ❤️ by [Aarush Sah](https://github.com/AarushSah) and the [Groq](https://groq.com) team.