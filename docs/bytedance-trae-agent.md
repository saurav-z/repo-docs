<!-- SEO META DATA -->
<!-- Keywords: AI Agent, LLM, Software Engineering, CLI, Automation, OpenAI, Anthropic, Google Gemini, OpenRouter, Python -->

# Trae Agent: Your AI-Powered Software Engineering Assistant

Trae Agent is an innovative LLM-based agent designed to automate and streamline software engineering tasks, offering a powerful CLI for natural language interaction.  [Explore the original repository](https://github.com/bytedance/trae-agent).

**Key Features:**

*   ✨ **Multi-LLM Support:**  Integrates with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini for flexible model selection.
*   🛠️ **Rich Tool Ecosystem:** Offers file editing, bash execution, and sequential thinking capabilities, providing a comprehensive toolkit for software development.
*   🎯 **Interactive Mode:**  Provides a conversational interface for iterative development and debugging.
*   📊 **Trajectory Recording:**  Detailed logging of all agent actions, facilitating debugging, analysis, and auditing of agent behavior.
*   ⚙️ **Flexible Configuration:** Utilizes a JSON-based configuration system with environment variable support for easy setup and customization.
*   🚀 **Easy Installation:** Installs quickly with a simple pip-based process.
*   🌊 **Lakeview:**  Provides concise summarization for agent steps, improving workflow clarity and understanding.

## Getting Started

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

### API Key Setup

1.  **Configure API Keys:** Edit `trae_config.json` to include your API keys for OpenAI, Anthropic, Google, and other supported providers. A sample config file is provided.

    ```bash
    cp trae_config.json.example trae_config.json
    ```
    Replace the placeholder API keys and URLs in `trae_config.json` with your credentials.  This file is intentionally ignored by Git.

2.  **Alternative - Environment Variables:** Set API keys as environment variables for added security:

    ```bash
    # OpenAI
    export OPENAI_API_KEY="your-openai-api-key"

    # Anthropic
    export ANTHROPIC_API_KEY="your-anthropic-api-key"

    # Doubao (and OpenAI compatible models)
    export DOUBAO_API_KEY="your-doubao-api-key"
    export DOUBAO_BASE_URL="your-model-provider-base-url"

    # OpenRouter
    export OPENROUTER_API_KEY="your-openrouter-api-key"

    # Google Gemini
    export GOOGLE_API_KEY="your-google-api-key"

    # Optional: OpenRouter rankings
    export OPENROUTER_SITE_URL="https://your-site.com"
    export OPENROUTER_SITE_NAME="Your App Name"

    # Optional: OpenAI compatible API providers
    export OPENAI_BASE_URL="your-openai-compatible-api-base-url"
    ```

    For enhanced security, consider using a `.env` file with python-dotenv, adding `MODEL_API_KEY="My API Key"`.

### Basic Usage Examples

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## Command-Line Interface (CLI)

The `trae-cli` command is your primary interface for interacting with Trae Agent.

### `trae run` - Execute a Task

Execute tasks using natural language prompts.

```bash
# Basic execution
trae-cli run "Create a Python script that calculates fibonacci numbers"

# Specify provider and model
trae-cli run "Fix the bug in main.py" --provider anthropic --model claude-sonnet-4-20250514

# Utilize OpenRouter
trae-cli run "Optimize this code" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Add documentation" --provider openrouter --model "anthropic/claude-3-5-sonnet"

# Google Gemini example
trae-cli run "Implement a data parsing function" --provider google --model gemini-2.5-pro

# Custom working directory
trae-cli run "Add unit tests for the utils module" --working-dir /path/to/project

# Save trajectory for debugging
trae-cli run "Refactor the database module" --trajectory-file debug_session.json

# Force patch generation
trae-cli run "Update the API endpoints" --must-patch
```

### `trae interactive` - Interactive Mode

Engage in a conversational session with the agent.

```bash
# Start interactive session
trae-cli interactive

# Custom configuration
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

Within interactive mode:

*   Type task descriptions to execute them.
*   Use `status` for agent information.
*   Use `help` for available commands.
*   Use `clear` to clear the screen.
*   Use `exit` or `quit` to end the session.

### `trae show-config` - Configuration Status

View your current configuration settings.

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

## Configuration

Trae Agent uses a JSON configuration file (`trae_config.json`) for settings.  Configuration precedence is: command-line arguments > configuration file > environment variables > default values.

**Important for Doubao users:**

```
base_url=https://ark.cn-beijing.volces.com/api/v3/
```

### Model Provider Examples

```bash
# GPT-4 through OpenRouter
trae-cli run "Write a Python script" --provider openrouter --model "openai/gpt-4o"

# Claude through OpenRouter
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"

# Gemini through OpenRouter
trae-cli run "Generate docs" --provider openrouter --model "google/gemini-pro"

# Gemini directly
trae-cli run "Analyze this dataset" --provider google --model gemini-2.5-flash

# Qwen through Ollama
trae-cli run "Comment this code" --provider ollama --model "qwen3"
```

**Popular OpenRouter Models:**

*   `openai/gpt-4o` - Latest GPT-4 model
*   `anthropic/claude-3-5-sonnet` - Excellent for coding tasks
*   `google/gemini-pro` - Strong reasoning capabilities
*   `meta-llama/llama-3.1-405b` - Open source alternative
*   `openai/gpt-4o-mini` - Fast and cost-effective

## Environment Variables

-   `OPENAI_API_KEY`
-   `ANTHROPIC_API_KEY`
-   `GOOGLE_API_KEY`
-   `OPENROUTER_API_KEY`
-   `OPENROUTER_SITE_URL` (Optional)
-   `OPENROUTER_SITE_NAME` (Optional)

## Available Tools

Trae Agent provides a comprehensive set of tools for software engineering tasks, including file editing, bash execution, and structured thinking, with ongoing development and improvements.  See [docs/tools.md](docs/tools.md) for details.

## Trajectory Recording

Trae Agent automatically records detailed execution trajectories for in-depth analysis and debugging.

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files include:

*   LLM Interactions
*   Agent Steps
*   Tool Usage
*   Metadata

See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md) for more information.

## Contributing

Contributions are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

1.  Fork the repository.
2.  Set up a development install:  `make install-dev`
3.  Create a feature branch (`git checkout -b feature/amazing-feature`).
4.  Make changes and add tests.
5.  Run pre-commit:
    ```bash
    make pre-commit
    # or
    make uv-pre-commit
    ```
    If formatting issues arise:
    ```bash
    make fix-format
    ```
6.  Commit changes (`git commit -m 'Add amazing feature'`).
7.  Push to the branch.
8.  Open a Pull Request.

### Development Guidelines

*   Follow PEP 8 style.
*   Add tests for new features.
*   Update documentation as necessary.
*   Use type hints.
*   Ensure all tests pass before submitting.

## Requirements

*   Python 3.12+
*   API keys for your chosen LLM providers (OpenAI, Anthropic, OpenRouter, Google Gemini).

## Troubleshooting

### Common Issues

**Import Errors:**  Try setting `PYTHONPATH=. trae-cli run "your task"`.

**API Key Issues:**

1.  Verify API keys: `echo $OPENAI_API_KEY`, etc.
2.  Check configuration: `trae-cli show-config`.

**Permission Errors:**  Ensure proper file permissions.

**Command Not Found:**  Try `uv run trae-cli <your command>`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank Anthropic for their [anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts) project, which provided valuable insights for the tool ecosystem.