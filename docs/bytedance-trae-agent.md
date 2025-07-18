# Trae Agent: Your AI-Powered Software Engineering Assistant

**Trae Agent** is a cutting-edge, LLM-based agent designed to streamline software engineering tasks, offering a powerful CLI for executing complex workflows with natural language instructions. [Explore the original repository](https://github.com/bytedance/trae-agent) for the latest updates and contributions.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)

## Key Features

*   **Multi-LLM Support:** Works with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama and Google Gemini APIs
*   **Rich Tool Ecosystem:** Includes file editing, bash execution, sequential thinking, and more, extending agent capabilities.
*   **Interactive Mode:** Conversational interface for iterative development and real-time feedback.
*   **Trajectory Recording:** Detailed logging of agent actions for comprehensive debugging and analysis.
*   **Flexible Configuration:** JSON-based configuration with environment variable support for easy customization.
*   **Easy Installation:** Simple pip-based installation for quick setup and deployment.
*   **Lakeview Summarization:** Provides concise summaries for each agent step.

## Why Trae Agent?

Trae Agent is designed with a transparent and modular architecture, making it a prime choice for research and development, offering a flexible platform to study AI agent architectures and build novel agent capabilities.

## Quick Start

### Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) to setup the project.

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv venv
uv sync --all-extras
```

or use make.

```bash
make uv-venv
make uv-sync
```

### Setup API Keys

Configure Trae Agent using a config file or environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Doubao (also works with other OpenAI-compatible model providers)
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="your-model-provider-base-url"

# For OpenRouter
export OPENROUTER_API_KEY="your-openrouter-api-key"

# For Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# Optional: For OpenRouter rankings
export OPENROUTER_SITE_URL="https://your-site.com"
export OPENROUTER_SITE_NAME="Your App Name"

# Optional: If you want to use a specific openai compatible api provider, you can set the base url here
export OPENAI_BASE_URL="your-openai-compatible-api-base-url"
```

For security, use [python-dotenv](https://pypi.org/project/python-dotenv/) to store `MODEL_API_KEY="My API Key"` in a `.env` file.

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## Usage

### Command Line Interface

*   `trae run`: Execute a task with various options.
    ```bash
    # Basic task execution
    trae-cli run "Create a Python script that calculates fibonacci numbers"

    # With specific provider and model
    trae-cli run "Fix the bug in main.py" --provider anthropic --model claude-sonnet-4-20250514

    # Using OpenRouter with any supported model
    trae-cli run "Optimize this code" --provider openrouter --model "openai/gpt-4o"
    trae-cli run "Add documentation" --provider openrouter --model "anthropic/claude-3-5-sonnet"

    # Using Google Gemini
    trae-cli run "Implement a data parsing function" --provider google --model gemini-2.5-pro

    # With custom working directory
    trae-cli run "Add unit tests for the utils module" --working-dir /path/to/project

    # Save trajectory for debugging
    trae-cli run "Refactor the database module" --trajectory-file debug_session.json

    # Force to generate patches
    trae-cli run "Update the API endpoints" --must-patch
    ```
*   `trae interactive`: Engage in an interactive session.
    ```bash
    # Start interactive session
    trae-cli interactive

    # With custom configuration
    trae-cli interactive --provider openai --model gpt-4o --max-steps 30
    ```
    Within interactive mode, use commands such as `status`, `help`, `clear`, `exit`, and `quit`.
*   `trae show-config`: View the current configuration status.
    ```bash
    trae-cli show-config

    # With custom config file
    trae-cli show-config --config-file my_config.json
    ```

### Configuration

Use a JSON configuration file (`trae_config.json`).  Configuration follows this priority: command-line arguments, config file, environment variables, and default values.

**WARNING:**  For Doubao users, use `base_url=https://ark.cn-beijing.volces.com/api/v3/`.

**Popular OpenRouter Models:**

*   `openai/gpt-4o`
*   `anthropic/claude-3-5-sonnet`
*   `google/gemini-pro`
*   `meta-llama/llama-3.1-405b`
*   `openai/gpt-4o-mini`

### Environment Variables

*   `OPENAI_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `GOOGLE_API_KEY`
*   `OPENROUTER_API_KEY`
*   `OPENROUTER_SITE_URL` (Optional)
*   `OPENROUTER_SITE_NAME` (Optional)

## Available Tools

Trae Agent offers a robust set of tools, including file editing, bash execution, and more. See [docs/tools.md](docs/tools.md) for details.

## Trajectory Recording

Automatic trajectory recording for debugging and analysis.

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

```bash
make install-dev
make pre-commit
```

## Requirements

*   Python 3.12+
*   API keys for your chosen provider (OpenAI, Anthropic, OpenRouter, Google Gemini).

## Troubleshooting

*   **Import Errors:** Try `PYTHONPATH=. trae-cli run "your task"`.
*   **API Key Issues:** Verify keys with `echo $OPENAI_API_KEY` and check configuration with `trae-cli show-config`.
*   **Permission Errors:** Ensure proper file permissions with `chmod +x /path/to/your/project`.
*   **Command not found Errors:** try `uv run trae-cli `xxxxx``

## License

This project is licensed under the MIT License (see [LICENSE](LICENSE)).

## Acknowledgments

Thanks to Anthropic for the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project.