# Trae Agent: Your LLM-Powered Software Engineering Assistant

**Empower your software engineering workflow with Trae Agent, a versatile and extensible LLM-based agent designed for a wide range of tasks. [Explore the original repo](https://github.com/bytedance/trae-agent).**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent is an innovative LLM-based agent specifically designed to streamline software engineering tasks. It features a transparent, modular architecture, perfect for researchers and developers interested in AI agent design.

**Project Status:** Actively under development. See [docs/roadmap.md](docs/roadmap.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Key Features

*   **Multi-LLM Support:** Works seamlessly with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   **Rich Tool Ecosystem:** Includes file editing, bash execution, sequential thinking, and more, enabling complex workflows.
*   **Interactive Mode:** Provides a conversational interface for iterative development and experimentation.
*   **Trajectory Recording:** Detailed logging of agent actions for debugging, analysis, and reproducibility.
*   **Flexible Configuration:** JSON-based configuration with environment variable support for easy customization.
*   **Easy Installation:** Simple pip-based installation using `uv` or `make`.
*   **Lakeview:** Provides short and concise summarization for agent steps.

## Quick Start

### Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) to set up the project.

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

Configure Trae Agent via a config file or environment variables.

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

Consider using [python-dotenv](https://pypi.org/project/python-dotenv/) to store `MODEL_API_KEY="My API Key"` in your `.env` file, preventing direct exposure of your API key.

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

The core command is `trae` with subcommands:

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

In interactive mode:

-   Type task descriptions to execute them.
-   Use `status` for agent info.
-   Use `help` for commands.
-   Use `clear` and `exit`/`quit` as expected.

#### `trae show-config` - Configuration Status

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

### Configuration

Trae Agent uses a JSON configuration file (see `trae_config.json`).

**WARNING:** For Doubao users, use `base_url=https://ark.cn-beijing.volces.com/api/v3/`.

**Configuration Priority:**

1.  Command-line arguments
2.  Configuration file values
3.  Environment variables
4.  Default values

**Examples:**

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

Trae Agent is equipped with a robust suite of tools for file editing, bash execution, structured thinking, task completion, and JSON manipulation.  New tools and enhancements are consistently being added.  For a comprehensive list, refer to [docs/tools.md](docs/tools.md).

## Trajectory Recording

Trae Agent automatically records detailed execution trajectories for debugging and analysis.

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:

*   LLM Interactions: Messages, responses, and tool calls
*   Agent Steps: State transitions and decision points
*   Tool Usage: Called tools and their results
*   Metadata: Timestamps, token usage, and execution metrics

See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md) for more details.

## Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

1.  Fork the repository.
2.  Set up a development install:
    ```bash
    make install-dev
    ```
3.  Create a feature branch (`git checkout -b feature/amazing-feature`).
4.  Make changes.
5.  Add tests.
6.  Pre-commit check:
    ```bash
    make pre-commit
    or:
    make uv-pre-commit
    ```
    If formatting errors occur, try:
    ```
    make fix-format
    ```
7.  Commit changes (`git commit -m 'Add amazing feature'`).
8.  Push (`git push origin feature/amazing-feature`).
9.  Open a Pull Request.

### Development Guidelines

*   Follow PEP 8 style.
*   Add tests for new features.
*   Update documentation.
*   Use type hints.
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
# Verify your API keys
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
echo $OPENROUTER_API_KEY

# Check configuration
trae-cli show-config
```

**Permission Errors:**

```bash
# Ensure proper permissions
chmod +x /path/to/your/project
```

**Command not found Errors:**

```bash
# you can try
uv run trae-cli `xxxxx`
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Thanks to Anthropic for [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts), a valuable reference for the tool ecosystem.