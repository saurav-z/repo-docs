# OpenBench: The Open-Source LLM Evaluation Powerhouse

**Tired of vendor lock-in and inconsistent LLM benchmark results?** OpenBench is your solution: a provider-agnostic, open-source evaluation infrastructure for language models, offering standardized and reproducible benchmarking across a vast range of models and tasks. Explore the original repo: [https://github.com/groq/openbench](https://github.com/groq/openbench).

## Key Features:

*   **🚀 35+ Benchmarks:** Evaluate LLMs across diverse categories, including knowledge, math, reasoning, coding, and more.
*   **🌐 Provider-Agnostic:** Works seamlessly with 30+ model providers, including Groq, OpenAI, Anthropic, and local models.
*   **⚙️ Simple CLI:** Easily run, manage, and view benchmark results with intuitive commands like `bench eval` and `bench view`.
*   **📦 Extensible & Customizable:** Add new benchmarks, metrics, and even run your own private evaluations with ease.
*   **📤 Hugging Face Integration:** Directly push evaluation results to Hugging Face Datasets for sharing and analysis.
*   **🔒 Local Eval Support:** Run privatized benchmarks with `bench eval <path>`.
*   **⚡ Integrated Developer Tools**: GitHub Actions Integration and Inspect AI extension support.

## What's New in v0.3.0

*   **📡 18 More Model Providers**: Expanded support for a wider range of providers.
*   **🧪 New Benchmarks**: Including the DROP reading comprehension benchmark.
*   **⚡ CLI Enhancements**: Improved command-line interface for streamlined workflows.
*   **🔧 Developer Tools**: Integration of GitHub Actions and Inspect AI extension support.

## Get Started in 60 Seconds!

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

## Detailed Features & Functionality

### Supported Providers

OpenBench supports 30+ model providers through Inspect AI. Setting the appropriate API key environment variable allows you to utilize your choice of model.

### Available Benchmarks

OpenBench offers a comprehensive set of benchmarks, categorized for easy navigation. (See the list in original README)

### Commands and Options

Easily run benchmark evaluations with the following options:

| Command                  | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `bench` or `openbench`   | Show main menu with available commands             |
| `bench list`             | List available evaluations, models, and flags      |
| `bench eval <benchmark>` | Run benchmark evaluation on a model                |
| `bench eval-retry`       | Retry a failed evaluation                          |
| `bench view`             | View logs from previous benchmark runs             |
| `bench eval <path>`      | Run your local/private evals built with Inspect AI |

**Key `eval` Command Options:** (See the table in original README)

### Grader Information

Some benchmarks employ a grader model for evaluation. To run these benchmarks, you'll need to export the appropriate `OPENAI_API_KEY`. (See the tables in the original README)

### Configuration

Set up your API keys to prepare for model evaluation:

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/openai/gpt-oss-20b
```

## Building Your Own Evals

OpenBench is built upon [Inspect AI](https://inspect.aisi.org.uk/). Integrate custom evaluations by utilizing `bench eval <path>` to point OpenBench at your existing work.

## Exporting Logs to Hugging Face

OpenBench can export logs to a Hugging Face Hub dataset.

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

## FAQ

(See FAQ from original README)

## Development

Follow these instructions to set up your development environment.

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

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed instructions.

Quick links:

*   [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
*   [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

OpenBench aims to implement evaluations as faithfully as possible. (See the statement in the original README)

## Acknowledgments

(See the acknowledgements from original README)

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