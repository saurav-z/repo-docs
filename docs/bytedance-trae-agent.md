# Trae Agent: Your LLM-Powered Software Engineering Assistant

**Trae Agent**, available on [GitHub](https://github.com/bytedance/trae-agent), is a powerful AI agent designed to assist with software engineering tasks, offering a flexible and extensible platform for developers and researchers alike.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent is a research-friendly AI agent that lets you automate complex software engineering workflows with natural language commands.

## Key Features

*   **Multi-LLM Support:** Works seamlessly with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama and Google Gemini APIs.
*   **Rich Tool Ecosystem:** Includes file editing, bash execution, sequential thinking, and more, enabling a wide range of engineering tasks.
*   **Interactive Mode:** Provides a conversational interface for iterative development and refinement of code.
*   **Detailed Trajectory Recording:** Logs all agent actions for debugging, analysis, and understanding agent behavior.
*   **Flexible Configuration:** JSON-based configuration with environment variable support for easy customization.
*   **Easy Installation:** Simple pip-based installation for quick setup.
*   **Lakeview:** Provides short and concise summarisation for agent steps

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

### API Key Setup

Configure API keys using either a configuration file or environment variables.  Refer to the documentation for details.

**Environment Variables Example:**

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

Consider using [python-dotenv](https://pypi.org/project/python-dotenv/) to manage API keys securely.

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

The core command is `trae-cli`, offering the following subcommands:

#### `trae run` - Execute a Task

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

#### `trae interactive` - Interactive Mode

```bash
# Start interactive session
trae-cli interactive

# With custom configuration
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

Interactive mode allows you to provide task descriptions and interact with the agent.

#### `trae show-config` - Configuration Status

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

### Configuration

Trae Agent uses a JSON configuration file. The configuration file's location and structure can be found in `trae_config.json`. Command-line arguments take the highest priority, followed by the config file, environment variables, and lastly, the default values.

**Important for Doubao Users:** Use the following base URL:

```
base_url=https://ark.cn-beijing.volces.com/api/v3/
```

**Examples of using LLMs and OpenRouter**
```bash
# Use GPT-4 through OpenRouter
trae-cli run "Write a Python script" --provider openrouter --model "openai/gpt-4o"

# Use Claude through OpenRouter
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"

# Use Gemini through OpenRouter
trae-cli run "Generate docs" --provider openrouter --model "google/gemini-pro"

# Use Gemini directly
trae-cli run "Analyze this dataset" --provider google --model gemini-2.5-flash

# Use Qwen through Ollama
trae-cli run "Comment this code" --provider ollama --model "qwen3"
```

**Popular OpenRouter Models:**

*   `openai/gpt-4o` - Latest GPT-4 model
*   `anthropic/claude-3-5-sonnet` - Excellent for coding tasks
*   `google/gemini-pro` - Strong reasoning capabilities
*   `meta-llama/llama-3.1-405b` - Open source alternative
*   `openai/gpt-4o-mini` - Fast and cost-effective

### Environment Variables

*   `OPENAI_API_KEY` - OpenAI API key
*   `ANTHROPIC_API_KEY` - Anthropic API key
*   `GOOGLE_API_KEY` - Google Gemini API key
*   `OPENROUTER_API_KEY` - OpenRouter API key
*   `OPENROUTER_SITE_URL` - (Optional) Your site URL for OpenRouter rankings
*   `OPENROUTER_SITE_NAME` - (Optional) Your site name for OpenRouter rankings

## Available Tools

Trae Agent offers a robust set of tools for file editing, bash execution, structured thinking, and JSON manipulation. New tools are actively being developed, and existing tools are continuously enhanced.  See `docs/tools.md` for details.

## Trajectory Recording

Trae Agent automatically records detailed execution trajectories for debugging and analysis.

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files capture:

*   LLM Interactions
*   Agent Steps
*   Tool Usage
*   Metadata

See `docs/TRAJECTORY_RECORDING.md` for more details.

## Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

1.  Fork the repository.
2.  Set up a development environment:
    ```bash
    make install-dev
    ```
3.  Create a feature branch.
4.  Make your changes and add tests.
5.  Pre-commit check:
    ```bash
    make pre-commit
    or:
    make uv-pre-commit
    ```
    If formatting errors occur, try:
    ```bash
    make fix-format
    ```
6.  Commit your changes.
7.  Push to your branch.
8.  Open a Pull Request.

### Development Guidelines

*   Follow PEP 8 style.
*   Add tests for new features.
*   Update documentation as needed.
*   Use type hints where appropriate.
*   Ensure all tests pass.

## Requirements

*   Python 3.12+
*   API key for your chosen provider (OpenAI, Anthropic, OpenRouter, Google Gemini).

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Try setting PYTHONPATH
PYTHONPATH=. trae-cli run "your task"
```

**API Key Issues:**

```bash
# Verify your API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
echo $OPENROUTER_API_KEY

# Check configuration
trae-cli show-config
```

**Permission Errors:**
```bash
# Ensure proper permissions for file operations
chmod +x /path/to/your/project
```

**Command not found Errors:**
```bash
# you can try
uv run trae-cli `xxxxx`
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank Anthropic for the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project.