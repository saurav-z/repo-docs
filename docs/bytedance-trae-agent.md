# Trae Agent: Your AI-Powered Software Engineering Assistant

**Trae Agent** is an open-source, LLM-powered agent designed to automate and streamline software engineering tasks. Explore the original repo [here](https://github.com/bytedance/trae-agent).

**Key Features:**

*   **🤖 Multi-LLM Support:** Works seamlessly with OpenAI, Anthropic, Doubao, Azure, and OpenRouter APIs.
*   **🛠️ Rich Tool Ecosystem:** Includes file editing, bash execution, and sequential thinking tools.
*   **🌊 Lakeview Summarization:** Provides concise summaries of agent steps for easy understanding.
*   **🎯 Interactive Mode:** Engage in a conversational interface for iterative development.
*   **📊 Trajectory Recording:** Detailed logging for debugging and analysis.
*   **⚙️ Flexible Configuration:** JSON-based configuration with environment variable support.
*   **🚀 Easy Installation:** Simple pip-based installation.

## Getting Started

### Installation

We recommend using [uv](https://docs.astral.sh/uv/) for project setup.

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv sync
```

### API Key Setup

Configure Trae Agent using a config file or environment variables:

```bash
# Set API keys for your preferred provider
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_API_BASE_URL="your-model-provider-base-url"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENAI_BASE_URL="your-openai-compatible-api-base-url"

# Recommended: Use .env file for API key security
#  Add MODEL_API_KEY="My API Key" to your .env file
```

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6
```

## Usage Guide

### Command Line Interface

The `trae-cli` command offers these subcommands:

#### `trae run`: Task Execution

```bash
# Create a Python script that calculates fibonacci numbers
trae-cli run "Create a Python script that calculates fibonacci numbers"

# Use a specific provider and model
trae-cli run "Fix the bug in main.py" --provider anthropic --model claude-sonnet-4-20250514

# Use OpenRouter
trae-cli run "Optimize this code" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Add documentation" --provider openrouter --model "anthropic/claude-3-5-sonnet"

# Use custom working directory
trae-cli run "Add unit tests for the utils module" --working-dir /path/to/project

# Save trajectory for debugging
trae-cli run "Refactor the database module" --trajectory-file debug_session.json

# Force to generate patches
trae-cli run "Update the API endpoints" --must-patch
```

#### `trae interactive`: Interactive Mode

```bash
# Start interactive session
trae-cli interactive

# With custom configuration
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

In interactive mode, use commands like `status`, `help`, `clear`, and `exit` or `quit`.

#### `trae show-config`: Configuration Status

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

### Configuration

The `trae_config.json` file allows customization of settings:

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

**Important for Doubao users**: Use the base URL `https://ark.cn-beijing.volces.com/api/v3/`.

**Configuration Priority:** Command-line > Config File > Environment Variables > Default.

### OpenRouter Model Examples

```bash
trae-cli run "Write a Python script" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"
trae-cli run "Generate docs" --provider openrouter --model "google/gemini-pro"
trae-cli run "Comment this code" --provider ollama --model "qwen3"
```

### Environment Variables

*   `OPENAI_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `OPENROUTER_API_KEY`
*   `OPENROUTER_SITE_URL` (Optional)
*   `OPENROUTER_SITE_NAME` (Optional)

## Tools

Trae Agent includes these tools:

*   **str_replace_based_edit_tool**: File creation, editing, and manipulation.
*   **bash**: Execute shell commands.
*   **sequential_thinking**: Structured problem-solving.
*   **task_done**: Signals task completion.

## Trajectory Recording

Execution trajectories are automatically recorded for debugging:

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

See [TRAJECTORY\_RECORDING.md](TRAJECTORY_RECORDING.md) for details.

## Contributing

1.  Fork the repository.
2.  Set up a development install(`uv sync --all-extras && pre-commit install`).
3.  Create a feature branch.
4.  Make your changes.
5.  Add tests.
6.  Commit changes.
7.  Push to the branch.
8.  Open a Pull Request.

### Development Guidelines

*   Follow PEP 8.
*   Add tests for new features.
*   Update documentation.
*   Use type hints.
*   Ensure tests pass.

## Requirements

*   Python 3.12+
*   API keys (OpenAI, Anthropic, OpenRouter).

## Troubleshooting

### Common Issues

**Import Errors:**

```bash
PYTHONPATH=. trae-cli run "your task"
```

**API Key Issues:**

```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $OPENROUTER_API_KEY
trae-cli show-config
```

**Permission Errors:**

```bash
chmod +x /path/to/your/project
```

## License

This project is licensed under the MIT License.

## Acknowledgments

Thanks to Anthropic for the [anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts) project.