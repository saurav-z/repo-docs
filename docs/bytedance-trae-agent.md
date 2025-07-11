# Trae Agent: Your AI-Powered Software Engineering Assistant

**Trae Agent is an advanced AI agent designed to simplify and automate software engineering tasks with natural language commands.** ([Original Repo](https://github.com/bytedance/trae-agent))

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![Alpha]( https://img.shields.io/badge/Status-Alpha-red)
 [![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
 [![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)

## Key Features

*   **🤖 Multi-LLM Support:** Works with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs, offering flexibility in model selection.
*   **🛠️ Rich Tool Ecosystem:** Provides file editing, bash execution, sequential thinking, and more, enabling complex software engineering workflows.
*   **🌊 Lakeview Summarization:** Get concise summaries of agent steps for enhanced clarity.
*   **🎯 Interactive Mode:** Offers a conversational interface for iterative development and easy experimentation.
*   **📊 Trajectory Recording:** Detailed logging of all agent actions for debugging, analysis, and understanding agent behavior.
*   **⚙️ Flexible Configuration:** JSON-based configuration with environment variable support for easy customization.
*   **🚀 Easy Installation:** Simple pip-based installation for quick setup and deployment.

## Getting Started

### Installation

We recommend using [uv](https://docs.astral.sh/uv/) for project setup.

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
make install
```

### Setting Up API Keys

Configure Trae Agent using a config file or environment variables for a secure and convenient setup.

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

Consider using [python-dotenv](https://pypi.org/project/python-dotenv/) to store API keys in a `.env` file and avoid exposing them in version control.

### Basic Usage

```bash
# Create a simple Python script
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## Command Line Interface

Trae Agent's main command is `trae`, with subcommands offering various functionalities.

### `trae run` - Execute a Task

Execute software engineering tasks using natural language descriptions.

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

Engage in a conversational mode for iterative development and experimentation.

```bash
# Start interactive session
trae-cli interactive

# With custom configuration
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

In interactive mode, you can:

-   Input task descriptions for execution
-   Use `status` to view agent information
-   Use `help` for command assistance
-   Use `clear` to clean the screen
-   Use `exit` or `quit` to end the session

### `trae show-config` - Configuration Status

Displays the current configuration settings.

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

## Configuration

Trae Agent utilizes a JSON configuration file for settings. Consult the `trae_config.json` file in the root directory for detailed configuration structures.

**WARNING:** For Doubao users, use the following `base_url`.

```
base_url=https://ark.cn-beijing.volces.com/api/v3/
```

**Configuration Priority:**

1.  Command-line arguments
2.  Configuration file values
3.  Environment variables
4.  Default values

**Examples using OpenRouter:**

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

Trae Agent provides a comprehensive suite of tools for file editing, bash execution, structured thinking, and JSON manipulation. For detailed information on all available tools and their capabilities, see [docs/tools.md](docs/tools.md).

## Trajectory Recording

Trae Agent automatically records detailed execution trajectories for debugging and analysis.
```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:

-   **LLM Interactions**: Messages, responses, and tool calls
-   **Agent Steps**: State transitions and decision points
-   **Tool Usage**: Tool calls and their results
-   **Metadata**: Timestamps, token usage, and execution metrics

For more details, see [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

1.  Fork the repository
2.  Set up a development install (`make install-dev pre-commit-install`)
3.  Create a feature branch (`git checkout -b feature/amazing-feature`)
4.  Make your changes
5.  Add tests for new features
6.  Commit your changes (`git commit -m 'Add amazing feature'`)
7.  Push to the branch (`git push origin feature/amazing-feature`)
8.  Open a Pull Request

### Development Guidelines

-   Follow PEP 8 style guidelines
-   Add tests for new features
-   Update documentation as needed
-   Use type hints where appropriate
-   Ensure all tests pass before submitting

## Requirements

-   Python 3.12+
-   API key for your chosen provider:
    -   OpenAI API key (for OpenAI models)
    -   Anthropic API key (for Anthropic models)
    -   OpenRouter API key (for OpenRouter models)
    -   Google API key (for Google Gemini models)

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

We thank Anthropic for building the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project that served as a valuable reference for the tool ecosystem.