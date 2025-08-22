# OpenBench: Evaluate LLMs with Ease and Precision 🚀

**Unlock the power of standardized and reproducible LLM benchmarking across 30+ providers with OpenBench!**

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench is a provider-agnostic, open-source evaluation infrastructure that simplifies and standardizes the benchmarking of Large Language Models (LLMs). It offers a comprehensive suite of over 35 benchmarks covering knowledge, math, reasoning, coding, and more, allowing you to evaluate any model from 30+ providers, including Groq, OpenAI, Anthropic, Google, and local models.  [Explore OpenBench on GitHub](https://github.com/groq/openbench).

**Key Features:**

*   **Provider Agnostic:** Seamlessly supports 30+ model providers, eliminating vendor lock-in.
*   **Extensive Benchmarks:** Evaluate models across 35+ benchmarks, including MMLU, HumanEval, and more.
*   **Simple CLI:**  Easy-to-use command-line interface for streamlined evaluation workflows.
*   **Local Eval Support:** Run private benchmarks while preserving privacy.
*   **Hugging Face Integration:** Push results directly to Hugging Face datasets for sharing and analysis.
*   **Extensible Architecture:** Easily add new benchmarks and integrate with existing evaluation frameworks.
*   **Reproducibility:** Built upon Inspect AI for reliable and reproducible results.

## Quick Start: Evaluate a Model in Seconds

Get started with OpenBench in under a minute!

1.  **Install:** `uv venv && source .venv/bin/activate && uv pip install openbench`
2.  **Set API Key:** `export GROQ_API_KEY=your_key` (or relevant provider key)
3.  **Run Evaluation:** `bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10`
4.  **View Results:** `./logs/` or use `bench view` to visualize in an interactive UI.

## Model Provider Support

OpenBench supports a wide array of model providers:

| Provider          | Example Model String          |
| ----------------- | ----------------------------- |
| Groq              | `groq/llama-3.3-70b-versatile` |
| OpenAI            | `openai/o3-2025-04-16`       |
| Anthropic         | `anthropic/claude-sonnet-4-20250514`       |
| Google            | `google/gemini-2.5-pro`        |
| Hugging Face      | `huggingface/gpt-oss-120b:groq` |
| **And 30+ more!** |  See the full list below.      |

## Supported Providers

OpenBench supports 30+ model providers via environment variables.

**(See detailed table in original README for complete provider and environment variable mappings)**

## Available Benchmarks

OpenBench offers a diverse range of benchmarks for comprehensive model evaluation.

**(See detailed table in original README for a full benchmark list.)**

## Commands and Options

**(See detailed table in original README for full usage instructions.)**

Key Commands:

*   `bench list`: Lists available evaluations, models, and flags.
*   `bench eval <benchmark>`: Runs a benchmark evaluation on a model.
*   `bench eval-retry`: Retries a failed evaluation.
*   `bench view`: Views logs from previous benchmark runs.

## Building Custom Evals

OpenBench is built on top of [Inspect AI](https://inspect.aisi.org.uk/). Leverage Inspect AI's documentation to build your custom evaluations. Run your own private evaluations with Inspect AI using `bench eval <path>`.

## Exporting Results to Hugging Face

Share your evaluation results by exporting them to a Hugging Face Hub dataset:

```bash
export HF_TOKEN=<your-huggingface-token>
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## FAQ

**(See original README for a comprehensive FAQ section.)**

## Development

**(See original README for development instructions.)**

## Contributing

**(See original README for contributing guidelines.)**

## Reproducibility Statement

**(See original README for the reproducibility statement.)**

## Acknowledgments

**(See original README for the acknowledgements.)**

## Citation

**(See original README for the citation details.)**

## License

MIT