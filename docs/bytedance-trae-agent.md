# Trae Agent: Your AI-Powered Software Engineering Assistant

Trae Agent is an innovative LLM-based agent designed to streamline software engineering tasks with a powerful CLI interface.  [Explore the original repository](https://github.com/bytedance/trae-agent).

**Key Features:**

*   ✅ **Multi-LLM Support:** Works seamlessly with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   🧰 **Rich Tool Ecosystem:**  Includes file editing, bash execution, sequential thinking, and more, enabling complex workflows.
*   💬 **Interactive Mode:** Engage in conversational development with an intuitive interface.
*   📝 **Trajectory Recording:** Detailed logging of all agent actions for comprehensive debugging and analysis.
*   ⚙️ **Flexible Configuration:** Utilize JSON-based configuration with environment variable support for easy customization.
*   🚀 **Easy Installation:** Simple pip-based installation for quick setup.
*   🔍 **Lakeview Summarization:** Concise summaries of agent steps.
*   💡 **Research-Friendly Design:**  A modular and transparent architecture ideal for studying AI agent architectures and developing novel capabilities.

## Getting Started

### Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) to set up the project.

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

### Setting Up API Keys

Configure Trae Agent using a config file or environment variables for each provider.

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

Consider using [python-dotenv](https://pypi.org/project/python-dotenv/) to securely manage your API keys in a `.env` file.

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## Comprehensive Usage

### Command Line Interface

The primary entry point is the `trae` command with the following subcommands:

#### `trae run` - Task Execution

Execute software engineering tasks using natural language instructions:

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

Start a conversational session for iterative development:

```bash
# Start interactive session
trae-cli interactive

# With custom configuration
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

In interactive mode:
*   Type task descriptions to execute them.
*   Use `status` for agent information.
*   Use `help` for command reference.
*   Use `clear` to clear the screen.
*   Use `exit` or `quit` to end the session.

#### `trae show-config` - Configuration Display

View the current configuration settings:

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

### Configuration

Trae Agent uses a JSON configuration file for settings.  Refer to `trae_config.json` for the detailed structure.

**WARNING:**
For Doubao users, please use the following base_url.

```
base_url=https://ark.cn-beijing.volces.com/api/v3/
```

**Configuration Priority (highest to lowest):**

1.  Command-line arguments
2.  Configuration file values
3.  Environment variables
4.  Default values

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

*   `OPENAI_API_KEY` - Your OpenAI API key.
*   `ANTHROPIC_API_KEY` - Your Anthropic API key.
*   `GOOGLE_API_KEY` - Your Google Gemini API key.
*   `OPENROUTER_API_KEY` - Your OpenRouter API key.
*   `OPENROUTER_SITE_URL` - (Optional) Your site URL for OpenRouter rankings.
*   `OPENROUTER_SITE_NAME` - (Optional) Your site name for OpenRouter rankings.

## Available Tools

Trae Agent includes tools for:

*   File Editing
*   Bash Execution
*   Structured Thinking
*   Task Completion
*   JSON Manipulation

For more details, see [docs/tools.md](docs/tools.md).

## Trajectory Recording

Trae Agent automatically records detailed execution trajectories:

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:
*   LLM Interactions.
*   Agent Steps.
*   Tool Usage.
*   Metadata.

For more information, consult [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Steps:

1.  Fork the repository.
2.  Set up a development install: `make install-dev`.
3.  Create a feature branch (`git checkout -b feature/amazing-feature`).
4.  Make your changes.
5.  Add tests for new functionality.
6.  Pre-commit check:  `make pre-commit` or `make uv-pre-commit`.
    *   If formatting errors, try `make fix-format`.
7.  Commit your changes (`git commit -m 'Add amazing feature'`).
8.  Push to the branch (`git push origin feature/amazing-feature`).
9.  Open a Pull Request.

### Development Guidelines

*   Follow PEP 8.
*   Include tests for new features.
*   Update documentation as required.
*   Use type hints.
*   Ensure all tests pass.

## Requirements

*   Python 3.12+
*   API keys for your chosen providers: OpenAI, Anthropic, OpenRouter, Google Gemini.

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

This project is licensed under the MIT License; see the [LICENSE](LICENSE) file.

## Acknowledgments

We thank Anthropic for the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project.