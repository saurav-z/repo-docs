# Trae Agent: Your AI-Powered Software Engineering Assistant

**Trae Agent is an LLM-based agent designed to simplify your software engineering tasks by understanding natural language and executing complex workflows.** Explore the power of AI-driven development and enhance your productivity.  [View the original repository on GitHub](https://github.com/bytedance/trae-agent).

**Key Features:**

*   🤖 **Multi-LLM Support:** Seamlessly integrates with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs, offering flexibility in model selection.
*   🛠️ **Rich Tool Ecosystem:** Provides a comprehensive set of tools, including file editing, bash execution, and sequential reasoning, streamlining various development processes.
*   🎯 **Interactive Mode:** Offers a conversational interface for iterative development, allowing for real-time feedback and refinement of tasks.
*   🌊 **Lakeview:** Offers short and concise summarisation for agent steps
*   📊 **Trajectory Recording:** Detailed logging of all agent actions for debugging and analysis
*   ⚙️ **Flexible Configuration:** JSON-based configuration with environment variable support
*   🚀 **Easy Installation:** Simple pip-based installation

## Getting Started

### Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) to setup the project.

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

1.  **Copy the example configuration file:**

    ```bash
    cp trae_config.json.example trae_config.json
    ```

2.  **Edit `trae_config.json`** with your API keys or set environment variables:

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

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

##  Commands

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

### `trae show-config` - Configuration Status

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

## Configuration

Utilizes a JSON configuration file (`trae_config.json`) for settings. Command-line arguments override config file values, which in turn override environment variables, followed by default values.

**Doubao users should use the following `base_url`:**

```
base_url=https://ark.cn-beijing.volces.com/api/v3/
```

## Available Tools

Trae Agent includes a comprehensive toolkit for various software engineering tasks. For details, see [docs/tools.md](docs/tools.md).

## Trajectory Recording

Trae Agent automatically records execution trajectories for debugging and analysis. Learn more at [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Requirements

*   Python 3.12+
*   API keys: OpenAI, Anthropic, OpenRouter, Google Gemini.

## Troubleshooting

See common issues and solutions for help.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

We thank Anthropic for their [anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts) project, which served as a valuable reference for the tool ecosystem.