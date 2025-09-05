# OpenBench: The Open-Source LLM Evaluation Powerhouse

**Effortlessly benchmark and compare Large Language Models (LLMs) across various providers with OpenBench, a provider-agnostic, open-source evaluation infrastructure.** [Explore OpenBench on GitHub](https://github.com/groq/openbench)

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench simplifies LLM evaluation with standardized, reproducible benchmarking across diverse models and providers.  It offers over 35 evaluation suites, including knowledge, reasoning, coding, and more, with first-class support for local evaluations.  **OpenBench is compatible with over 30 model providers, including Groq, OpenAI, Anthropic, and local models through Ollama and Hugging Face.**

## Key Features

*   **Extensive Benchmark Suite:** Evaluate models on 35+ benchmarks (and growing!), covering a wide range of capabilities.
*   **Provider Agnostic:** Seamlessly test models from 30+ providers without code modifications.
*   **Simple CLI:** Utilize easy-to-use commands like `bench eval`, `bench list`, and `bench view`.
*   **Local Evaluation Support:** Run evaluations on your own data and models while preserving privacy.
*   **Hugging Face Integration:** Push results directly to Hugging Face datasets for easy sharing and analysis.
*   **Extensible Architecture:** Easily add new benchmarks, metrics, and model providers.

## What's New in v0.3.0

*   **Expanded Provider Support:** Added support for 18 more model providers, including AI21, Baseten, Cerebras, and more.
*   **New Benchmarks:** Introduced the DROP reading comprehension benchmark.
*   **CLI Enhancements:** Included `openbench` alias, `-M`/`-T` flags for model/task arguments, and `--debug` mode.
*   **Enhanced Developer Tools:**  GitHub Actions integration, Inspect AI extension support

## Get Started: Evaluate a Model in Seconds

Quickly evaluate models with these easy steps:

1.  **Install:**  Create a virtual environment and install OpenBench (using [uv](https://docs.astral.sh/uv/getting-started/installation/)):

    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install openbench
    ```

2.  **Set API Key:**  Set the appropriate environment variable for your model provider (e.g., `export GROQ_API_KEY=your_key` for Groq models).
3.  **Run Evaluation:** Execute a benchmark:

    ```bash
    bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10
    ```

4.  **View Results:**  Access results in the `./logs/` directory or visualize them using:

    ```bash
    bench view
    ```

## Supported Providers and Model Format

OpenBench supports a wide array of model providers. Set the corresponding API key as an environment variable, then specify the model name.

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

Use `bench list` to see the full and up-to-date list.

| Category          | Benchmarks                                                                                                                                                                                                                                                                 |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge**     | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version)                                                                                         |
| **Coding**        | HumanEval (164 problems)                                                                                                                                                                                                                                                   |
| **Math**          | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages)                                         |
| **Reasoning**     | SimpleQA (factuality), MuSR (multi-step reasoning), DROP (discrete reasoning over paragraphs), MMMU (multi-modal reasoning with 30+ subjects)                                                                                                                              |
| **Long Context**  | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                                                                                                                                    |
| **Healthcare**    | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant)                                                                                                                                                |
| **Cybersecurity** | CTI-Bench (complete cyber threat intelligence suite), CTI-Bench ATE (MITRE ATT&CK technique extraction), CTI-Bench MCQ (knowledge questions on CTI standards and best practices), CTI-Bench RCM (CVE to CWE vulnerability mapping), CTI-Bench VSP (CVSS score calculation) |

## Configuration and Command-Line Options

Configure OpenBench with environment variables and control evaluations using the `bench` command.  See the `--help` option for a complete list of commands.

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

Certain benchmarks rely on grader models for scoring. These require the `OPENAI_API_KEY`.

| Benchmark         | Default Grader Model           |
| ----------------- | ------------------------------ |
| `simpleqa`        | `openai/gpt-4.1-2025-04-14` |
| `hle`             | `openai/o3-mini-2025-01-31` |
| `hle_text`        | `openai/o3-mini-2025-01-31` |
| `browsecomp`      | `openai/gpt-4.1-2025-04-14` |
| `healthbench`     | `openai/gpt-4.1-2025-04-14` |
| `math`            | `openai/gpt-4-turbo-preview` |
| `math_500`        | `openai/gpt-4-turbo-preview` |

## Building Custom Evaluations

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/).  Create your own custom evaluations with Inspect AI, and then use `bench eval <path>` to run them.

## Exporting Logs to Hugging Face

Share your evaluation results by exporting logs to a Hugging Face Hub dataset:

```bash
export HF_TOKEN=<your-huggingface-token>
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## FAQ

*   **How is OpenBench different from Inspect AI?** OpenBench provides a benchmark library, shared utilities, curated scorers, and a user-friendly CLI, all built on Inspect AI's foundation.
*   **Why not use Inspect AI, lm-evaluation-harness, or lighteval?**  OpenBench focuses on shared components, clean implementations, and a great developer experience to provide a curated and reliable benchmark suite.
*   **How to run `bench` outside the `uv` environment:** `uv run pip install -e .`
*   **Dataset download issues from HuggingFace:** Set the `HF_TOKEN="<HUGGINGFACE_TOKEN>"` environment variable.

## Development

Contribute to OpenBench by cloning the repository, setting up the environment (using `uv`), installing pre-commit hooks, and running tests. Review the [Contributing Guide](CONTRIBUTING.md) for instructions.

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for detailed instructions.

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

OpenBench strives to faithfully implement evaluations but acknowledges potential numerical discrepancies compared to other sources.  For meaningful comparisons, use the same OpenBench version.

## Acknowledgements

-   [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)
-   [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
-   [Hugging Face's lighteval](https://github.com/huggingface/lighteval)

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