# Trae Agent: Your AI-Powered Software Engineering Assistant

**Trae Agent empowers software engineers to tackle complex tasks with natural language, offering a powerful CLI and modular design for research and development.** ([Original Repo](https://github.com/bytedance/trae-agent))

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent is an LLM-based agent designed to streamline software engineering tasks. This research-friendly framework is ideal for studying AI agent architectures and exploring innovative agent capabilities.

## Key Features

*   **🤖 Multi-LLM Support:** Works with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs, providing flexibility in model selection.
*   **🛠️ Rich Tool Ecosystem:** Includes file editing, bash execution, sequential thinking, and more, enabling comprehensive task automation.
*   **🎯 Interactive Mode:** Offers a conversational interface for iterative development and exploration.
*   **📊 Trajectory Recording:** Logs detailed agent actions for thorough debugging, analysis, and reproducibility.
*   **⚙️ Flexible Configuration:** Utilizes JSON-based configuration with environment variable support for easy customization.
*   **🚀 Easy Installation:** Simple pip-based installation process to get you up and running quickly.
*   **🌊 Lakeview:** Provides short and concise summarisation for agent steps

## Quick Start

### Installation

We recommend using [uv](https://docs.astral.sh/uv/) to set up the project.

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv venv
uv sync --all-extras
```

Alternatively, use `make`:

```bash
make uv-venv
make uv-sync
```

### API Key Setup

Configure Trae Agent using a config file, or set API keys as environment variables:

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

Consider using [python-dotenv](https://pypi.org/project/python-dotenv/) to store `MODEL_API_KEY="My API Key"` in your `.env` file.

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

The `trae` command offers various subcommands:

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

Interactive mode commands:

*   Type a task description
*   `status`
*   `help`
*   `clear`
*   `exit` / `quit`

#### `trae show-config` - Configuration Status

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

### Configuration

Trae Agent uses a JSON configuration file (`trae_config.json`).

**WARNING:**
For Doubao users, use this `base_url`:

```
base_url=https://ark.cn-beijing.volces.com/api/v3/
```

**Configuration Priority:**

1.  Command-line arguments (highest)
2.  Configuration file values
3.  Environment variables
4.  Default values (lowest)

```bash
# Example Usage with OpenRouter
trae-cli run "Write a Python script" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"
trae-cli run "Generate docs" --provider openrouter --model "google/gemini-pro"

# Direct Usage
trae-cli run "Analyze this dataset" --provider google --model gemini-2.5-flash
trae-cli run "Comment this code" --provider ollama --model "qwen3"
```

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

Trae Agent includes tools for file editing, bash execution, structured thinking, and more.

Explore available tools in [docs/tools.md](docs/tools.md).

## Trajectory Recording

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:

*   LLM Interactions
*   Agent Steps
*   Tool Usage
*   Metadata

Learn more in [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## Contributing

Follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) to contribute.

1.  Fork the repository
2.  Setup dev install:
    ```bash
    make install-dev
    ```
3.  Create feature branch (`git checkout -b feature/amazing-feature`)
4.  Make changes
5.  Add tests
6.  Pre-commit check
    ```bash
    make pre-commit
    or:
    make uv-pre-commit
    ```
    If formatting issues:
    ```bash
    make fix-format
    ```
7.  Commit (`git commit -m 'Add amazing feature'`)
8.  Push (`git push origin feature/amazing-feature`)
9.  Open Pull Request

### Development Guidelines

*   PEP 8 style
*   Add tests
*   Update documentation
*   Use type hints
*   Ensure tests pass

## Requirements

*   Python 3.12+
*   API key for your chosen provider (OpenAI, Anthropic, OpenRouter, Google Gemini)

## Troubleshooting

### Common Issues

**Import Errors:**

```bash
# Try setting PYTHONPATH
PYTHONPATH=. trae-cli run "your task"
```

**API Key Issues:**

```bash
# Verify keys
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
echo $OPENROUTER_API_KEY

# Check configuration
trae-cli show-config
```

**Permission Errors:**

```bash
# Ensure permissions
chmod +x /path/to/your/project
```

**Command not found Errors:**

```bash
# Use uv run
uv run trae-cli `xxxxx`
```

## License

MIT License - see [LICENSE](LICENSE).

## Acknowledgments

Thanks to Anthropic for [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts).