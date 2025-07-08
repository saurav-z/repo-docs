# Trae Agent: Your AI-Powered Assistant for Software Engineering

**Trae Agent** is an open-source, LLM-powered agent designed to streamline software engineering tasks. [Explore the original repository here](https://github.com/bytedance/trae-agent).

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Alpha-red)](https://github.com/bytedance/trae-agent)

*Note: This project is in the alpha stage and is actively being developed. Contributions are welcome!*

## Key Features:

*   🌊 **Lakeview Summarization:** Provides concise summaries of agent steps for easy understanding.
*   🤖 **Multi-LLM Support:** Works seamlessly with various LLM providers, including OpenAI, Anthropic, Doubao, Azure, and OpenRouter.
*   🛠️ **Rich Tool Ecosystem:** Offers a versatile toolkit, including file editing, bash execution, and sequential thinking capabilities.
*   🎯 **Interactive Mode:**  Engage in a conversational interface for iterative development and real-time interaction.
*   📊 **Detailed Trajectory Recording:** Captures all agent actions, ensuring in-depth debugging and insightful analysis.
*   ⚙️ **Flexible Configuration:** Configure settings using JSON files, complemented by environment variable support.
*   🚀 **Easy Installation:** Quickly get started with a simple pip-based installation process.

## Why Use Trae Agent?

Trae Agent's modular and transparent architecture makes it ideal for:

*   **Research and Development:**  Study AI agent architectures and build innovative agent capabilities.
*   **Ablation Studies:** Conduct in-depth experiments to understand the impact of specific agent components.
*   **Community Contributions:**  Contribute to and expand the foundational agent framework.

## Getting Started

### Installation

It's recommended to use [UV](https://docs.astral.sh/uv/) for project setup.

```bash
git clone <repository-url>
cd trae-agent
uv sync
```

### Setting Up API Keys

Configure Trae Agent via the configuration file or environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Doubao (and other OpenAI-compatible providers)
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_API_BASE_URL="your-model-provider-base-url"

# OpenRouter
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Optional OpenRouter settings
export OPENROUTER_SITE_URL="https://your-site.com"
export OPENROUTER_SITE_NAME="Your App Name"
```

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6
```

## Command-Line Interface (CLI)

The core functionality of Trae Agent is accessed through the `trae` command and its subcommands.

### `trae run` - Execute Tasks

```bash
# Basic task execution
trae-cli run "Create a Python script that calculates fibonacci numbers"

# Specify provider and model
trae-cli run "Fix the bug in main.py" --provider anthropic --model claude-sonnet-4-20250514

# Use OpenRouter with any supported model
trae-cli run "Optimize this code" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Add documentation" --provider openrouter --model "anthropic/claude-3-5-sonnet"

# Custom working directory
trae-cli run "Add unit tests for the utils module" --working-dir /path/to/project

# Save trajectory for debugging
trae-cli run "Refactor the database module" --trajectory-file debug_session.json

# Force patch generation
trae-cli run "Update the API endpoints" --must-patch
```

### `trae interactive` - Interactive Mode

```bash
# Start interactive session
trae-cli interactive

# Custom configuration example
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

In interactive mode, you can:
*   Enter task descriptions to execute.
*   Use `status` to view agent information.
*   Type `help` for available commands.
*   Use `clear` to clear the screen.
*   Use `exit` or `quit` to end the session.

### `trae show-config` - Configuration Status

```bash
trae-cli show-config

# With a custom config file
trae-cli show-config --config-file my_config.json
```

## Configuration

Trae Agent uses a JSON configuration file (`trae_config.json`) to customize settings.

```json
{
  "default_provider": "anthropic",
  "max_steps": 20,
  "enable_lakeview": true,
  "model_providers": {
    "openai": {
      "api_key": "your_openai_api_key",
      "model": "gpt-4o",
      "max_tokens": 128000,
      "temperature": 0.5,
      "top_p": 1,
      "max_retries": 10
    },
    "anthropic": {
      "api_key": "your_anthropic_api_key",
      "model": "claude-sonnet-4-20250514",
      "max_tokens": 4096,
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 0,
      "max_retries": 10
    },
    "azure": {
      "api_key": "you_azure_api_key",
      "base_url": "your_azure_base_url",
      "api_version": "2024-03-01-preview",
      "model": "model_name",
      "max_tokens": 4096,
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 0,
      "max_retries": 10
    },
    "openrouter": {
      "api_key": "your_openrouter_api_key",
      "model": "openai/gpt-4o",
      "max_tokens": 4096,
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 0,
      "max_retries": 10
    },
    "doubao": {
      "api_key": "you_doubao_api_key",
      "model": "model_name",
      "base_url": "your_doubao_base_url",
      "max_tokens": 8192,
      "temperature": 0.5,
      "top_p": 1,
      "max_retries": 20
    }
  },
  "lakeview_config": {
    "model_provider": "anthropic",
    "model_name": "claude-sonnet-4-20250514"
  }
}
```

**Configuration Priority:**
1.  Command-line arguments (highest)
2.  Configuration file values
3.  Environment variables
4.  Default values (lowest)

**Example Usage (OpenRouter):**

```bash
# Use GPT-4 through OpenRouter
trae-cli run "Write a Python script" --provider openrouter --model "openai/gpt-4o"

# Use Claude through OpenRouter
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"

# Use Gemini through OpenRouter
trae-cli run "Generate docs" --provider openrouter --model "google/gemini-pro"

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
*   `OPENROUTER_API_KEY` - OpenRouter API key
*   `OPENROUTER_SITE_URL` - (Optional) Your site URL for OpenRouter rankings
*   `OPENROUTER_SITE_NAME` - (Optional) Your site name for OpenRouter rankings

## Available Tools

Trae Agent offers several built-in tools to facilitate software engineering tasks:

*   **`str_replace_based_edit_tool`**: Create, edit, view, and manipulate files.
    *   `view` - Display file contents or directory listings
    *   `create` - Create new files
    *   `str_replace` - Replace text in files
    *   `insert` - Insert text at specific lines
*   **`bash`**: Execute shell commands and scripts.
    *   Run commands with persistent state
    *   Handle long-running processes
    *   Capture output and errors
*   **`sequential_thinking`**: Structured problem-solving and analysis.
    *   Break down complex problems
    *   Iterative thinking with revision capabilities
    *   Hypothesis generation and verification
*   **`task_done`**: Signal task completion.
    *   Mark tasks as successfully completed
    *   Provide final results and summaries

## Trajectory Recording

Trae Agent automatically records detailed execution trajectories for debugging and analysis.

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files include:

*   **LLM Interactions**: Messages, responses, and tool calls.
*   **Agent Steps**: State transitions and decision points.
*   **Tool Usage**: Calls made and their results.
*   **Metadata**: Timestamps, token usage, and execution metrics.

For detailed information, see [TRAJECTORY_RECORDING.md](TRAJECTORY_RECORDING.md).

## Contributing

Join the Trae Agent community!

1.  Fork the repository
2.  Set up a development install(`uv sync --all-extras && pre-commit install`)
3.  Create a feature branch (`git checkout -b feature/amazing-feature`)
4.  Implement your changes
5.  Add tests for any new functionality
6.  Commit your changes (`git commit -m 'Add amazing feature'`)
7.  Push to the branch (`git push origin feature/amazing-feature`)
8.  Open a Pull Request

### Development Guidelines

*   Follow PEP 8 style guidelines.
*   Include tests for all new features.
*   Keep the documentation up-to-date.
*   Utilize type hints wherever appropriate.
*   Ensure all tests are passing before submitting your contribution.

## Requirements

*   Python 3.12+
*   API key for your chosen provider:
    *   OpenAI API key (for OpenAI models)
    *   Anthropic API key (for Anthropic models)
    *   OpenRouter API key (for OpenRouter models)

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
echo $OPENROUTER_API_KEY

# Check configuration
trae-cli show-config
```

**Permission Errors:**

```bash
# Ensure proper permissions for file operations
chmod +x /path/to/your/project
```

## License

This project is licensed under the MIT License. For more details, please consult the [LICENSE](LICENSE) file.

## Acknowledgments

We extend our gratitude to Anthropic for creating the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project, which served as a valuable resource for the tool ecosystem.