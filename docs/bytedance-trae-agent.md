# Trae Agent: Your AI-Powered Software Engineering Assistant

**Trae Agent is a powerful CLI tool that leverages LLMs to automate and streamline software engineering tasks, acting as your intelligent coding companion.**  [Check out the original repository](https://github.com/bytedance/trae-agent) for more details.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

**Key Features:**

*   🤖 **Multi-LLM Support:**  Integrates with a wide range of LLM providers including OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini.
*   🛠️ **Rich Tool Ecosystem:** Offers a comprehensive suite of tools for file editing, bash execution, sequential reasoning, and more, simplifying complex software engineering workflows.
*   🎯 **Interactive Mode:** Provides a conversational interface for iterative development and easy interaction with the agent.
*   📊 **Trajectory Recording:** Logs detailed agent actions, LLM interactions, and tool usage for in-depth debugging, analysis, and understanding of agent behavior.
*   ⚙️ **Flexible Configuration:** Enables customization with JSON-based configuration and environment variable support for easy setup and adaptation.
*   🌊 **Lakeview:** Offers short and concise summarisation for agent steps.
*   🚀 **Easy Installation:** Simplifies the setup process with a straightforward pip-based installation.

**Project Status:**  The project is under active development.  See the [docs/roadmap.md](docs/roadmap.md) and [CONTRIBUTING](CONTRIBUTING.md) files for contributing.

**Research-Friendly Design:** Trae Agent's transparent, modular architecture allows researchers and developers to easily modify, extend, and analyze agent capabilities, facilitating research and innovation in AI agent architectures.

## Getting Started

### Installation

We recommend using [uv](https://docs.astral.sh/uv/) to set up the project.

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv venv
uv sync --all-extras
```

Alternatively, use make:

```bash
make uv-venv
make uv-sync
```

### API Key Setup

Set your API keys through the config file or as environment variables.  Example:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
# ... and other providers
```

Using a `.env` file with `python-dotenv` is also recommended for securing your keys:

```bash
# In your .env file:
MODEL_API_KEY="My API Key"
```

### Basic Usage

```bash
trae-cli run "Create a hello world Python script"
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## Usage Details

### Command Line Interface

The `trae-cli` command is the primary interface.  Subcommands include:

#### `trae run` - Execute a Task

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

#### `trae interactive` - Interactive Mode

```bash
trae-cli interactive
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

In interactive mode, type task descriptions and use commands like `status`, `help`, `clear`, `exit`, and `quit`.

#### `trae show-config` - Configuration Status

```bash
trae-cli show-config
trae-cli show-config --config-file my_config.json
```

### Configuration

Trae Agent uses a JSON configuration file (`trae_config.json`) to store settings.  Command-line arguments take the highest priority, followed by configuration file values, environment variables, and then default values.

**WARNING:**
For Doubao users, use this base_url: `https://ark.cn-beijing.volces.com/api/v3/`

**Example Commands:**

```bash
trae-cli run "Write a Python script" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"
trae-cli run "Generate docs" --provider openrouter --model "google/gemini-pro"
trae-cli run "Analyze this dataset" --provider google --model gemini-2.5-flash
trae-cli run "Comment this code" --provider ollama --model "qwen3"
```

**Popular OpenRouter Models:**

*   `openai/gpt-4o` - Latest GPT-4 model
*   `anthropic/claude-3-5-sonnet` - Excellent for coding tasks
*   `google/gemini-pro` - Strong reasoning capabilities
*   `meta-llama/llama-3.1-405b` - Open source alternative
*   `openai/gpt-4o-mini` - Fast and cost-effective

### Environment Variables

*   `OPENAI_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `GOOGLE_API_KEY`
*   `OPENROUTER_API_KEY`
*   `OPENROUTER_SITE_URL` (Optional)
*   `OPENROUTER_SITE_NAME` (Optional)

## Available Tools

Trae Agent includes a comprehensive toolkit for various software engineering tasks.  For details, see [docs/tools.md](docs/tools.md).

## Trajectory Recording

Trae Agent automatically records execution trajectories.

```bash
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain detailed information about:

*   LLM Interactions
*   Agent Steps
*   Tool Usage
*   Metadata

For more details, see [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines. Steps include forking, setting up a development install, creating a feature branch, making changes, adding tests, committing, pushing, and opening a pull request.

### Development Guidelines

*   Follow PEP 8 style.
*   Add tests.
*   Update documentation.
*   Use type hints.
*   Ensure tests pass.

## Requirements

*   Python 3.12+
*   API keys for your chosen providers (OpenAI, Anthropic, OpenRouter, Google Gemini).

## Troubleshooting

### Common Issues

**Import Errors:**  Try `PYTHONPATH=. trae-cli run "your task"`

**API Key Issues:** Verify your keys using `echo` and check your configuration using `trae-cli show-config`.

**Permission Errors:**  Ensure proper file permissions.

**Command not found Errors:** Try `uv run trae-cli `xxxxx``

## License

MIT License - see the [LICENSE](LICENSE) file.

## Acknowledgments

We thank Anthropic for building the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project.