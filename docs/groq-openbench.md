# OpenBench: The Open-Source LLM Benchmarking Powerhouse

**Evaluate any language model, on any provider, with reproducible, standardized benchmarks.** 🚀

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench is a **provider-agnostic**, open-source evaluation framework designed for **rigorous and reproducible benchmarking** of Large Language Models (LLMs).  With support for **30+ model providers**, including Groq, OpenAI, Anthropic, and local models via Ollama and Hugging Face, OpenBench allows you to benchmark any LLM. Explore **35+ pre-built evaluation suites**, covering diverse areas like knowledge, math, reasoning, and coding, and leverage local evaluation support to preserve your privacy.

[**Explore OpenBench on GitHub**](https://github.com/groq/openbench)

## Key Features:

*   **Extensive Benchmark Coverage:** Evaluate LLMs across 35+ benchmarks, including MMLU, HumanEval, and more, covering diverse domains.
*   **Provider Agnostic:** Seamlessly test models from 30+ providers, including Groq, OpenAI, Anthropic, Google, and local models.
*   **Simple CLI:**  Easily run and manage evaluations with intuitive commands like `bench eval`, `bench list`, and `bench view`.
*   **Reproducible Results:** Ensure consistent and reliable benchmarking with standardized evaluation procedures.
*   **Local Evaluation Support:**  Privatize benchmarks and run custom evaluations with `bench eval <path>`.
*   **Hugging Face Integration:**  Push your evaluation results directly to Hugging Face Datasets.
*   **Extensible:** Easily add new benchmarks and metrics to customize your evaluation process.

## Getting Started: Evaluate a Model in 60 Seconds

Quickly evaluate your first model with these simple steps:

1.  **Install OpenBench (30 seconds):**

    ```bash
    # Create a virtual environment and install OpenBench
    uv venv
    source .venv/bin/activate
    uv pip install openbench
    ```

2.  **Set Your API Key (any provider!)**

    ```bash
    export GROQ_API_KEY=your_key  # or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
    ```

3.  **Run Your First Evaluation (30 seconds)**

    ```bash
    bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10
    ```

    **That's it!** 🎉 View your results in `./logs/` or using the interactive UI with `bench view`.

## Provider Support

OpenBench supports 30+ model providers, including:

*   AI21 Labs
*   Anthropic
*   AWS Bedrock
*   Azure
*   Baseten
*   Cerebras
*   Cohere
*   Crusoe
*   DeepInfra
*   Friendli
*   Google
*   Groq
*   Hugging Face
*   Hyperbolic
*   Lambda
*   MiniMax
*   Mistral
*   Moonshot
*   Nebius
*   Nous Research
*   Novita AI
*   Ollama (local)
*   OpenAI
*   OpenRouter
*   Parasail
*   Perplexity
*   Reka
*   SambaNova
*   Together AI
*   vLLM (local)

Set the appropriate API key environment variable for your chosen provider, and you're ready to benchmark!

**See the original README for a comprehensive list of providers and example model strings.**

## Available Benchmarks

OpenBench offers a wide range of benchmarks categorized by domain:

*   **Knowledge:** MMLU, GPQA, SuperGPQA, OpenBookQA, HLE, HLE_text
*   **Coding:** HumanEval
*   **Math:** AIME, HMMT, BRUMO, MATH, MATH-500, MGSM, MGSM_en, MGSM_latin, MGSM_non_latin
*   **Reasoning:** SimpleQA, MuSR, DROP
*   **Long Context:** OpenAI MRCR, OpenAI MRCR_2n, OpenAI MRCR_4, OpenAI MRCR_8n
*   **Healthcare:** HealthBench, HealthBench_hard, HealthBench_consensus

**See the original README for more details on each benchmark.**

## Advanced Configuration & Commands

**Use `bench --help` for a complete list of all commands and options.**

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

## Building Custom Evaluations

OpenBench is built upon [Inspect AI](https://inspect.aisi.org.uk/).  Use Inspect AI to create your own evaluations, then use OpenBench to run them. `bench eval <path>`

## Exporting Logs to Hugging Face

Share your evaluation results with the community by exporting them to a Hugging Face Hub dataset:

```bash
export HF_TOKEN=<your-huggingface-token>
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## FAQ

**(See Original README for answers to common questions)**

*   How does OpenBench differ from Inspect AI?
*   Why not just use Inspect AI, lm-evaluation-harness, or lighteval?
*   How can I run `bench` outside of the `uv` environment?
*   I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

## Development

**(See Original README for details on development setup, testing, contributing, and license)**

## Acknowledgments

OpenBench is made possible by:

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