# OpenBench: The Ultimate LLM Evaluation Toolkit 🚀

**Effortlessly benchmark and compare Large Language Models (LLMs) across 30+ providers with OpenBench.** This open-source, provider-agnostic toolkit empowers you to rigorously evaluate LLMs using standardized benchmarks and ensure reproducible results. [Explore the original repository](https://github.com/groq/openbench).

*   **Key Features:**
    *   **Provider-Agnostic:** Supports 30+ model providers including Groq, OpenAI, Anthropic, Google, and local models.
    *   **35+ Benchmarks:** Includes industry-standard benchmarks like MMLU, HumanEval, and more, covering knowledge, reasoning, coding, and other critical areas.
    *   **Simple CLI:** Easy-to-use command-line interface for streamlined evaluation: `bench list`, `bench describe`, `bench eval`.
    *   **Extensible:** Easily add new benchmarks and custom metrics to tailor your evaluation.
    *   **Local Eval Support:** Run private evaluations using the Inspect AI framework with `bench eval <path>`.
    *   **Hugging Face Integration:** Directly push your evaluation results to Hugging Face datasets for sharing and analysis.

## Quick Start: Evaluate a Model in Under a Minute

Get started with OpenBench in just a few steps:

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

## Key Improvements in v0.3.0

*   **Expanded Provider Support:** Integration with 18 more model providers.
*   **New Benchmarks:** Introduction of the DROP reading comprehension benchmark and experimental benchmarks available with the `--alpha` flag.
*   **Enhanced CLI:** Added an `openbench` alias, model/task arguments with `-M` and `-T` flags, and a `--debug` mode for retry functionalities.
*   **Improved Developer Tools:** Incorporates GitHub Actions and Inspect AI extension support.

## Available Benchmarks

OpenBench offers a diverse range of benchmarks across critical LLM evaluation categories:

| Category            | Benchmarks                                                                                                                                                                                                      |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge**       | MMLU, GPQA, SuperGPQA, OpenBookQA, HLE, HLE_text                                                                                                                                                                  |
| **Coding**          | HumanEval                                                                                                                                                                                                     |
| **Math**            | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH, MATH-500, MGSM, MGSM_en, MGSM_latin, MGSM_non_latin                                                                                                              |
| **Reasoning**       | SimpleQA, MuSR, DROP                                                                                                                                                                                                      |
| **Long Context**    | OpenAI MRCR, OpenAI MRCR_2n, OpenAI MRCR_4, OpenAI MRCR_8n                                                                                                                                                   |
| **Healthcare**      | HealthBench, HealthBench_hard, HealthBench_consensus                                                                                                                                                            |
| **Cybersecurity**   | CTI-Bench, CTI-Bench ATE, CTI-Bench MCQ, CTI-Bench RCM, CTI-Bench VSP                                                                                                                                            |

For the most up-to-date list, use the command: `bench list`

## Supported Model Providers

OpenBench seamlessly integrates with a wide array of model providers through Inspect AI.  Simply set the appropriate API key environment variable to get started:

| Provider             | Environment Variable      | Example Model String                |
| -------------------- | ------------------------- | ----------------------------------- |
| AI21 Labs            | `AI21_API_KEY`            | `ai21/model-name`                   |
| Anthropic            | `ANTHROPIC_API_KEY`       | `anthropic/model-name`              |
| AWS Bedrock          | AWS credentials           | `bedrock/model-name`                |
| Azure                | `AZURE_OPENAI_API_KEY`    | `azure/<deployment-name>`           |
| Baseten              | `BASETEN_API_KEY`         | `baseten/model-name`                |
| Cerebras             | `CEREBRAS_API_KEY`        | `cerebras/model-name`               |
| Cohere               | `COHERE_API_KEY`          | `cohere/model-name`                 |
| Crusoe               | `CRUSOE_API_KEY`          | `crusoe/model-name`                 |
| DeepInfra            | `DEEPINFRA_API_KEY`       | `deepinfra/model-name`              |
| Friendli             | `FRIENDLI_TOKEN`          | `friendli/model-name`               |
| Google               | `GOOGLE_API_KEY`          | `google/model-name`                 |
| Groq                 | `GROQ_API_KEY`            | `groq/model-name`                   |
| Hugging Face         | `HF_TOKEN`                | `huggingface/model-name`            |
| Hyperbolic           | `HYPERBOLIC_API_KEY`      | `hyperbolic/model-name`             |
| Lambda               | `LAMBDA_API_KEY`          | `lambda/model-name`                 |
| MiniMax              | `MINIMAX_API_KEY`         | `minimax/model-name`                |
| Mistral              | `MISTRAL_API_KEY`         | `mistral/model-name`                |
| Moonshot             | `MOONSHOT_API_KEY`        | `moonshot/model-name`               |
| Nebius               | `NEBIUS_API_KEY`          | `nebius/model-name`                 |
| Nous Research        | `NOUS_API_KEY`            | `nous/model-name`                   |
| Novita AI            | `NOVITA_API_KEY`          | `novita/model-name`                 |
| Ollama               | None (local)              | `ollama/model-name`                 |
| OpenAI               | `OPENAI_API_KEY`          | `openai/model-name`                 |
| OpenRouter           | `OPENROUTER_API_KEY`      | `openrouter/model-name`             |
| Parasail             | `PARASAIL_API_KEY`        | `parasail/model-name`               |
| Perplexity           | `PERPLEXITY_API_KEY`      | `perplexity/model-name`             |
| Reka                 | `REKA_API_KEY`            | `reka/model-name`                   |
| SambaNova            | `SAMBANOVA_API_KEY`       | `sambanova/model-name`              |
| Together AI          | `TOGETHER_API_KEY`        | `together/model-name`               |
| Vercel AI Gateway    | `AI_GATEWAY_API_KEY`      | `vercel/creator-name/model-name`    |
| vLLM                 | None (local)              | `vllm/model-name`                   |

## Configuration and Commands

Customize your OpenBench experience with environment variables and explore a variety of command-line options:

### Key `eval` Command Options:

| Option                | Environment Variable      | Default                    | Description                                      |
| --------------------- | ------------------------- | -------------------------- | ------------------------------------------------ |
| `-M <args>`           | None                      | None                       | Pass model-specific arguments                    |
| `-T <args>`           | None                      | None                       | Pass task-specific arguments                    |
| `--model`             | `BENCH_MODEL`             | None (required)            | Model(s) to evaluate                            |
| `--epochs`            | `BENCH_EPOCHS`            | `1`                        | Number of epochs                                |
| `--max-connections`   | `BENCH_MAX_CONNECTIONS`   | `10`                       | Maximum parallel requests                      |
| `--temperature`       | `BENCH_TEMPERATURE`       | `0.6`                      | Model temperature                                |
| `--top-p`             | `BENCH_TOP_P`             | `1.0`                      | Model top-p                                       |
| `--max-tokens`        | `BENCH_MAX_TOKENS`        | `None`                     | Maximum tokens for model response                |
| `--seed`              | `BENCH_SEED`              | `None`                     | Seed for deterministic generation                |
| `--limit`             | `BENCH_LIMIT`             | `None`                     | Limit evaluated samples (number or start,end)   |
| `--logfile`           | `BENCH_OUTPUT`            | `None`                     | Output file for results                          |
| `--sandbox`           | `BENCH_SANDBOX`           | `None`                     | Environment for evaluation (local/docker)        |
| `--timeout`           | `BENCH_TIMEOUT`           | `10000`                    | Timeout for each API request (seconds)           |
| `--display`           | `BENCH_DISPLAY`           | `None`                     | Display type                                    |
| `--reasoning-effort`  | `BENCH_REASONING_EFFORT`  | `None`                     | Reasoning effort level                         |
| `--json`              | None                      | `False`                    | Output results in JSON format                    |
| `--hub-repo`          | `BENCH_HUB_REPO`          | `None`                     | Push results to a Hugging Face Hub dataset       |

## Building Custom Evaluations

OpenBench leverages the robust [Inspect AI](https://inspect.aisi.org.uk/) framework. Create custom evaluations by consulting their documentation and then run them with `bench eval <path>`.

## Exporting Logs to Hugging Face

Share your results and facilitate further analysis by exporting logs to a Hugging Face Hub dataset:

```bash
export HF_TOKEN=<your-huggingface-token>
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## Frequently Asked Questions

### How does OpenBench differ from Inspect AI?

OpenBench is a benchmark library built on Inspect AI, providing pre-built benchmarks, shared utilities, curated scorers, and a user-friendly CLI.

### How can I run `bench` outside of the `uv` environment?

Run `uv run pip install -e .`

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

Define the environment variable `HF_TOKEN="<HUGGINGFACE_TOKEN>"`

## Development

To contribute and develop OpenBench, follow these steps:

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

Your contributions are welcome! Review the [Contributing Guide](CONTRIBUTING.md) for details on development, adding benchmarks, code style, and submitting pull requests.

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

OpenBench is designed for faithful reproduction of results, but minor discrepancies with other sources may occur due to factors such as prompt variations and inference differences. Always compare results using the same OpenBench version.

## Acknowledgments

We thank:

*   **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)**
*   **[EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**
*   **[Hugging Face's lighteval](https://github.com/huggingface/lighteval)**

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