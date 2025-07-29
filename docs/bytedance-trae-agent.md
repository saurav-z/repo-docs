<!-- SEO-optimized README -->
# Trae Agent: Your AI-Powered Software Engineering Assistant

**Tired of repetitive coding tasks? Trae Agent is an LLM-powered assistant that simplifies software engineering workflows with natural language commands.**  Explore the original project on GitHub: [bytedance/trae-agent](https://github.com/bytedance/trae-agent).

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent is an innovative AI agent designed to streamline software development by executing tasks and managing workflows based on natural language instructions.  Its modular design fosters easy modification and extension, making it a great tool for exploring and developing AI agent capabilities.

**Project Status:** Actively under development. See [docs/roadmap.md](docs/roadmap.md) and [CONTRIBUTING](CONTRIBUTING.md) for more information.

**Key Features:**

*   **🤖 Multi-LLM Support:** Works with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama and Google Gemini APIs, offering flexibility in model selection.
*   **🛠️ Rich Tool Ecosystem:**  Includes file editing, bash execution, sequential thinking capabilities and more, facilitating comprehensive software engineering tasks.
*   **🎯 Interactive Mode:** Engage in conversational development with a user-friendly interface for iterative refinement.
*   **📊 Trajectory Recording:** Comprehensive logging of agent actions for debugging, analysis, and understanding agent behavior.
*   **⚙️ Flexible Configuration:**  Utilizes JSON-based configuration with environment variable support for easy customization.
*   **🚀 Easy Installation:** Simple pip-based installation process to get you started quickly.
*   🌊 **Lakeview**: Provides short and concise summarisation for agent steps.

## 🚀 Getting Started

### Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) for project setup.

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv venv
uv sync --all-extras
```

or

```bash
make uv-venv
make uv-sync
```

### Setup API Keys

Configure Trae Agent with the configuration file or environment variables.

**Configuration Setup (Recommended):**

1.  **Copy the example configuration:**

    ```bash
    cp trae_config.json.example trae_config.json
    ```

2.  **Edit `trae_config.json`**: Replace placeholder values with your API keys and credentials.

**Note:** `trae_config.json` is ignored by git.

**Environment Variables:**

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Doubao (and OpenAI-compatible providers)
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="your-model-provider-base-url"

# OpenRouter
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# Optional: OpenRouter rankings
export OPENROUTER_SITE_URL="https://your-site.com"
export OPENROUTER_SITE_NAME="Your App Name"

# Optional: OpenAI Compatible API
export OPENAI_BASE_URL="your-openai-compatible-api-base-url"
```

Consider using [python-dotenv](https://pypi.org/project/python-dotenv/) for managing API keys securely within a `.env` file.

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## 📖 Usage Guide

### Command Line Interface

The `trae-cli` command provides various subcommands.

#### `trae run` - Execute Tasks

```bash
# Basic execution
trae-cli run "Create a Python script that calculates fibonacci numbers"

# Specify provider and model
trae-cli run "Fix the bug in main.py" --provider anthropic --model claude-sonnet-4-20250514

# Use OpenRouter
trae-cli run "Optimize this code" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Add documentation" --provider openrouter --model "anthropic/claude-3-5-sonnet"
trae-cli run "Implement a data parsing function" --provider google --model gemini-2.5-pro

# Custom working directory
trae-cli run "Add unit tests for the utils module" --working-dir /path/to/project

# Save trajectory
trae-cli run "Refactor the database module" --trajectory-file debug_session.json

# Force patches
trae-cli run "Update the API endpoints" --must-patch
```

#### `trae interactive` - Interactive Mode

```bash
# Start interactive mode
trae-cli interactive

# Custom configuration
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

In interactive mode, you can:

*   Enter task descriptions.
*   Use commands like `status`, `help`, `clear`, and `exit`/`quit`.

#### `trae show-config` - View Configuration

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

### Configuration

Trae Agent uses a JSON configuration file (`trae_config.json`). Review the example for details.

**WARNING:** For Doubao users, use `base_url=https://ark.cn-beijing.volces.com/api/v3/`.

**Configuration Priority:** Command-line arguments > Configuration file > Environment variables > Default values.

```bash
# GPT-4 via OpenRouter
trae-cli run "Write a Python script" --provider openrouter --model "openai/gpt-4o"

# Claude via OpenRouter
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"

# Gemini via OpenRouter
trae-cli run "Generate docs" --provider openrouter --model "google/gemini-pro"

# Gemini Directly
trae-cli run "Analyze this dataset" --provider google --model gemini-2.5-flash

# Qwen via Ollama
trae-cli run "Comment this code" --provider ollama --model "qwen3"
```

**Popular OpenRouter Models:**

*   `openai/gpt-4o` - Latest GPT-4 model
*   `anthropic/claude-3-5-sonnet` - For coding tasks
*   `google/gemini-pro` - Reasoning capabilities
*   `meta-llama/llama-3.1-405b` - Open source alternative
*   `openai/gpt-4o-mini` - Fast and cost-effective

### Environment Variables

*   `OPENAI_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `GOOGLE_API_KEY`
*   `OPENROUTER_API_KEY`
*   `OPENROUTER_SITE_URL` (Optional)
*   `OPENROUTER_SITE_NAME` (Optional)

## 🛠️ Available Tools

Trae Agent offers an extensive set of tools for a variety of tasks.  See [docs/tools.md](docs/tools.md) for details.

## 📊 Trajectory Recording

```bash
# Auto-generated
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:

*   LLM interactions
*   Agent steps
*   Tool usage
*   Metadata

See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

1.  Fork the repository.
2.  Set up a development environment: `make install-dev`
3.  Create a feature branch.
4.  Make changes and add tests.
5.  Pre-commit check: `make pre-commit` or `make uv-pre-commit`, and fix formatting with `make fix-format` if necessary.
6.  Commit and push changes.
7.  Open a Pull Request.

### Development Guidelines

*   Follow PEP 8.
*   Add tests.
*   Update documentation.
*   Use type hints.
*   Ensure all tests pass.

## 📋 Requirements

*   Python 3.12+
*   API keys (OpenAI, Anthropic, OpenRouter, Google Gemini)

## 🔧 Troubleshooting

### Common Issues

**Import Errors:**

```bash
PYTHONPATH=. trae-cli run "your task"
```

**API Key Issues:**

```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
echo $OPENROUTER_API_KEY
trae-cli show-config
```

**Permission Errors:**

```bash
chmod +x /path/to/your/project
```

**Command not found Errors:**

```bash
uv run trae-cli `xxxxx`
```

## 📄 License

MIT License.  See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Thanks to Anthropic for the [anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts) project.