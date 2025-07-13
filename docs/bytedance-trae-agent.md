# Trae Agent: Your AI Assistant for Software Engineering

**Unlock the power of AI to automate and accelerate your software engineering tasks with Trae Agent**, a powerful and versatile LLM-based agent.  [Explore the original repo](https://github.com/bytedance/trae-agent) for the latest updates.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent is an AI-powered command-line tool designed to streamline software engineering workflows. It utilizes large language models (LLMs) and a rich set of tools to understand natural language instructions and execute complex tasks.  This project is research-friendly, facilitating AI agent architecture studies and the development of novel capabilities.

## Key Features

*   **Multi-LLM Support:** Seamlessly integrate with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   **Rich Tool Ecosystem:**  File editing, bash execution, sequential reasoning, and more, all at your fingertips.
*   **Interactive Mode:**  Engage in a conversational interface for iterative development and easy experimentation.
*   **Detailed Trajectory Recording:**  Comprehensive logging of agent actions for debugging, analysis, and understanding agent behavior.
*   **Flexible Configuration:** Configure Trae Agent using JSON files and environment variables.
*   **Easy Installation:**  Quick setup with `pip`.
*   **Lakeview:** Concise summarization of agent steps.

## Getting Started

### Installation

We recommend using [uv](https://docs.astral.sh/uv/) for project setup.

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv venv
uv sync --all-extras
```

or using `make`:

```bash
make uv-venv
make uv-sync
```

### Setting Up API Keys

Configure Trae Agent using the config file (preferred).  Alternatively, set API keys as environment variables:

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

Consider using [python-dotenv](https://pypi.org/project/python-dotenv/) to store `MODEL_API_KEY="My API Key"` in a `.env` file to protect your API key.

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## Command Line Interface

### `trae run` - Execute a Task

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

### `trae interactive` - Interactive Mode

```bash
# Start interactive session
trae-cli interactive

# With custom configuration
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

In interactive mode, you can:

*   Type task descriptions.
*   Use `status` to view agent information.
*   Use `help` for available commands.
*   Use `clear` to clear the screen.
*   Use `exit` or `quit` to end the session.

### `trae show-config` - Configuration Status

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

## Configuration

Trae Agent utilizes a JSON configuration file.  See `trae_config.json` in the root directory for details.

**WARNING:**
For Doubao users, use this `base_url`:

```
base_url=https://ark.cn-beijing.volces.com/api/v3/
```

**Configuration Priority:**

1.  Command-line arguments (highest)
2.  Configuration file
3.  Environment variables
4.  Default values (lowest)

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

## Environment Variables

*   `OPENAI_API_KEY` - OpenAI API key
*   `ANTHROPIC_API_KEY` - Anthropic API key
*   `GOOGLE_API_KEY` - Google API key
*   `OPENROUTER_API_KEY` - OpenRouter API key
*   `GOOGLE_API_KEY` - Google Gemini API key
*   `OPENROUTER_SITE_URL` - (Optional) Your site URL for OpenRouter rankings
*   `OPENROUTER_SITE_NAME` - (Optional) Your site name for OpenRouter rankings

## Available Tools

Trae Agent includes a comprehensive suite of tools for:

*   File editing
*   Bash execution
*   Structured thinking
*   Task completion
*   JSON manipulation

Explore [docs/tools.md](docs/tools.md) for detailed information.

## Trajectory Recording

Trae Agent automatically records execution trajectories for debugging and analysis.

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:

*   LLM interactions
*   Agent steps
*   Tool usage
*   Metadata (timestamps, token usage, metrics)

See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md) for more.

## Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1.  Fork the repository
2.  Set up a development install (`make install-dev pre-commit-install`)
3.  Create a feature branch (`git checkout -b feature/amazing-feature`)
4.  Make changes.
5.  Add tests.
6.  Commit (`git commit -m 'Add amazing feature'`)
7.  Push (`git push origin feature/amazing-feature`)
8.  Open a Pull Request

### Development Guidelines

*   Follow PEP 8 style.
*   Include tests for new features.
*   Update documentation as needed.
*   Use type hints.
*   Ensure tests pass.

## Requirements

*   Python 3.12+
*   API keys for your chosen providers (OpenAI, Anthropic, OpenRouter, Google Gemini)

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
echo $GOOGLE_API_KEY

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

Thanks to Anthropic for the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project.