# Trae Agent: Your AI-Powered Software Engineering Assistant

Trae Agent is an advanced LLM-based agent designed to automate and simplify your software engineering tasks.  Explore the original repository [here](https://github.com/bytedance/trae-agent).

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent is a research-friendly AI agent that empowers developers and researchers to streamline software engineering workflows with natural language commands.

**Key Features:**

*   **Multi-LLM Support:** Integrates with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   **Rich Tool Ecosystem:** Offers file editing, bash execution, sequential thinking, and more, for comprehensive task execution.
*   **Interactive Mode:** Engage in conversational development and iterate on your projects with ease.
*   **Trajectory Recording:** Provides detailed logging of agent actions for debugging and analysis.
*   **Flexible Configuration:** Utilize JSON-based configuration with environment variable support for easy setup.
*   **Easy Installation:**  Simple pip-based installation process to get you started quickly.
*   **Lakeview:** Get short and concise summaries of agent steps.

## Getting Started

### Installation

We recommend using [uv](https://docs.astral.sh/uv/) for project setup.

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

### Configuration

1.  **Copy the example configuration:**

    ```bash
    cp trae_config.json.example trae_config.json
    ```

2.  **Edit `trae_config.json`** Replace the placeholder values with your API keys and other credentials.

    **Note:** `trae_config.json` is git-ignored for security.

You can also use environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="your-model-provider-base-url"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export OPENROUTER_SITE_URL="https://your-site.com" # Optional
export OPENROUTER_SITE_NAME="Your App Name" # Optional
export OPENAI_BASE_URL="your-openai-compatible-api-base-url" # Optional
```

Consider using [python-dotenv](https://pypi.org/project/python-dotenv/) for storing API keys in a `.env` file.

### Basic Usage

```bash
# Create a "hello world" Python script
trae-cli run "Create a hello world Python script"

# Use Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Use Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## Command Line Interface

### `trae run` - Execute a Task

```bash
trae-cli run "Create a Python script that calculates fibonacci numbers"
trae-cli run "Fix the bug in main.py" --provider anthropic --model claude-sonnet-4-20250514
trae-cli run "Optimize this code" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Add documentation" --provider openrouter --model "anthropic/claude-3-5-sonnet"
trae-cli run "Implement a data parsing function" --provider google --model gemini-2.5-pro
trae-cli run "Add unit tests for the utils module" --working-dir /path/to/project
trae-cli run "Refactor the database module" --trajectory-file debug_session.json
trae-cli run "Update the API endpoints" --must-patch
```

### `trae interactive` - Interactive Mode

```bash
trae-cli interactive
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

### `trae show-config` - Configuration Status

```bash
trae-cli show-config
trae-cli show-config --config-file my_config.json
```

### Configuration

Trae Agent uses a JSON configuration file for customization. Refer to `trae_config.json` for details.

**Configuration Priority:**

1.  Command-line arguments (highest)
2.  Configuration file
3.  Environment variables
4.  Default values (lowest)

```bash
trae-cli run "Write a Python script" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"
trae-cli run "Generate docs" --provider openrouter --model "google/gemini-pro"
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

Trae Agent includes a wide array of tools for various software engineering tasks, with new tools and enhancements regularly added.  See [docs/tools.md](docs/tools.md) for details.

## Trajectory Recording

Trae Agent automatically records detailed execution trajectories:

```bash
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:

*   LLM Interactions
*   Agent Steps
*   Tool Usage
*   Metadata

For more information, see [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## Contributing

Follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) to contribute.

1.  Fork the repository.
2.  Set up a development install: `make install-dev`
3.  Create a feature branch.
4.  Make your changes and add tests.
5.  Pre-commit check: `make pre-commit` or `make uv-pre-commit`
6.  Commit and push your changes.
7.  Open a Pull Request.

### Development Guidelines

*   PEP 8 style.
*   Add tests for new features.
*   Update documentation.
*   Use type hints.
*   Ensure all tests pass.

## Requirements

*   Python 3.12+
*   API Keys:  OpenAI, Anthropic, OpenRouter, Google Gemini.

## Troubleshooting

### Common Issues

**Import Errors:** `PYTHONPATH=. trae-cli run "your task"`
**API Key Issues:** Verify and check configuration.
**Permission Errors:** Ensure proper file permissions.
**Command not found Errors:**  `uv run trae-cli `xxxxx``

## License

MIT License - see the [LICENSE](LICENSE) file.

## Acknowledgments

Thanks to Anthropic for the [anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts) project.