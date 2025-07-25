# Trae Agent: Your AI-Powered Software Engineering Assistant

Trae Agent is an advanced, open-source AI agent designed to automate and streamline software engineering tasks using natural language, offering a powerful CLI and research-friendly architecture. **[Explore the Trae Agent on GitHub](https://github.com/bytedance/trae-agent)**.

Key Features:

*   🧠 **Intelligent Task Execution:** Understands natural language instructions for complex engineering workflows.
*   🗣️ **Multi-LLM Support:**  Integrates with OpenAI, Anthropic, Google Gemini, and more.
*   🛠️ **Rich Tool Ecosystem:** Includes file editing, bash execution, sequential reasoning, and more for comprehensive task management.
*   💬 **Interactive Mode:** Conversational interface for iterative development and debugging.
*   💾 **Trajectory Recording:** Detailed logging for comprehensive debugging and analysis.
*   ⚙️ **Flexible Configuration:** Easy setup using JSON configuration and environment variables.
*   🚀 **Simplified Installation:**  Easy installation using `pip` and `uv`.

## Getting Started

### Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) to set up the project:

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

### Configure API Keys

1.  **Copy the example configuration:**

    ```bash
    cp trae_config.json.example trae_config.json
    ```

2.  **Edit `trae_config.json` and replace placeholder values:**

    *   Replace `"your_openai_api_key"` with your actual OpenAI API key.
    *   Replace `"your_anthropic_api_key"` with your actual Anthropic API key.
    *   Replace `"your_google_api_key"` with your actual Google API key.
    *   Replace other placeholder URLs and API keys as needed.

    **Note:**  The `trae_config.json` file is ignored by Git to prevent accidental exposure of your API keys.

You can also set API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"
# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
# Doubao (also works with OpenAI-compatible model providers)
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="your-model-provider-base-url"
# OpenRouter
export OPENROUTER_API_KEY="your-openrouter-api-key"
# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"
# Optional: OpenRouter rankings
export OPENROUTER_SITE_URL="https://your-site.com"
export OPENROUTER_SITE_NAME="Your App Name"
# Optional: OpenAI compatible api provider
export OPENAI_BASE_URL="your-openai-compatible-api-base-url"
```

Consider using [python-dotenv](https://pypi.org/project/python-dotenv/) for managing API keys in a `.env` file to prevent them from being committed to version control.

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## Command-Line Interface (CLI)

The primary entry point is the `trae-cli` command, which offers several subcommands.

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

In interactive mode:

*   Type a task description to execute it.
*   Use `status` to view agent information.
*   Use `help` for a list of commands.
*   Use `clear` to clear the screen.
*   Use `exit` or `quit` to end the session.

### `trae show-config` - Configuration Status

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

## Configuration

Trae Agent settings are managed through a JSON configuration file. See `trae_config.json` in the root directory for structure.

**Configuration Priority:**

1.  Command-line arguments (highest)
2.  Configuration file values
3.  Environment variables
4.  Default values (lowest)

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
*   `GOOGLE_API_KEY` - Google Gemini API key
*   `OPENROUTER_API_KEY` - OpenRouter API key
*   `OPENROUTER_SITE_URL` - (Optional) Your site URL for OpenRouter rankings
*   `OPENROUTER_SITE_NAME` - (Optional) Your site name for OpenRouter rankings

## Available Tools

Trae Agent includes a wide range of tools for software engineering, including file editing, bash execution, structured thinking, and JSON manipulation. For details, see [docs/tools.md](docs/tools.md).

## Trajectory Recording

Trae Agent automatically records all actions for debugging and analysis.

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

See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md) for more.

## Contributing

Contributions are welcome!  Follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

1.  Fork the repository.
2.  Set up a development install:
    ```bash
    make install-dev
    ```
3.  Create a feature branch.
4.  Make changes.
5.  Add tests for new functionality.
6.  Pre-commit check
    ```bash
     make pre-commit
     or:
     make uv-pre-commit
    ```
     if having formatting error,please try:
    ```
     make fix-format
    ```
7.  Commit your changes.
8.  Push to your branch.
9.  Open a Pull Request.

### Development Guidelines

*   Follow PEP 8 style.
*   Add tests.
*   Update documentation as needed.
*   Use type hints.
*   Ensure all tests pass.

## Requirements

*   Python 3.12+
*   API keys for your chosen LLM provider.

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to Anthropic for the [anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts) project, which served as a valuable reference.