# Trae Agent: Your AI-Powered Assistant for Software Engineering

**Trae Agent is an LLM-based agent designed to streamline software engineering tasks with its powerful CLI, offering a versatile and extensible platform for developers and researchers.** ([Original Repository](https://github.com/bytedance/trae-agent))

[![arXiv:2507.23370](https://img.shields.io/badge/TechReport-arXiv%3A2507.23370-b31a1b)](https://arxiv.org/abs/2507.23370)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

## Key Features

*   **🤖 Multi-LLM Support:** Seamlessly integrates with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs, allowing flexibility in choosing your preferred LLM.
*   **🛠️ Rich Tool Ecosystem:** Equipped with essential tools such as file editing, bash execution, and sequential thinking, empowering you to tackle complex software engineering workflows.
*   **🌊 Lakeview Summarization:** Quickly understand each step with concise summaries.
*   **🎯 Interactive Mode:** Engage in iterative development with a conversational interface.
*   **📊 Trajectory Recording:** Detailed logging of all agent actions, perfect for debugging and analysis.
*   **⚙️ Flexible Configuration:** YAML-based configuration with environment variable support for easy customization.
*   **🚀 Easy Installation:** Simple pip-based installation for a quick start.

## Getting Started

### Installation

Follow these steps to install Trae Agent:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/bytedance/trae-agent.git
    cd trae-agent
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    uv venv
    uv sync --all-extras

    # Activate the virtual environment
    source .venv/bin/activate
    ```
    Alternatively, use `make`:

    ```bash
    make uv-venv
    make uv-sync

    # Activate the virtual environment
    source .venv/bin/activate
    ```

### Setup API Keys

1.  **Configure via `trae_config.yaml`:**

    *   Copy the example configuration file:

        ```bash
        cp trae_config.yaml.example trae_config.yaml
        ```

    *   Edit `trae_config.yaml` and replace the placeholder values with your actual API keys for the models you plan to use (e.g., OpenAI, Anthropic, Google).
    *   Configure your preferred models and settings.
    *   *Note:* The `trae_config.yaml` file is ignored by Git to prevent accidental API key commits.

2.  **Legacy JSON Configuration:** Refer to [docs/legacy\_config.md](docs/legacy_config.md) for instructions if you are using the older JSON configuration format.

3.  **Configure environment variables:**

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

*   **Recommendation:** Use [python-dotenv](https://pypi.org/project/python-dotenv/) to store your API keys securely in a `.env` file:  `MODEL_API_KEY="My API Key"`.

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## Usage

### Command Line Interface

The primary entry point is the `trae-cli` command with subcommands:

#### `trae run` - Execute a Task

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

```bash
# Start interactive session
trae-cli interactive

# With custom configuration
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

In interactive mode:

*   Type task descriptions to execute them.
*   Use `status` for agent information.
*   Use `help` for available commands.
*   Use `clear` to clear the screen.
*   Use `exit` or `quit` to end the session.

#### `trae show-config` - Configuration Status

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file trae_config.yaml
```

### Configuration

Trae Agent uses a YAML configuration file (`trae_config.yaml`) for settings.

#### YAML Configuration Structure

The YAML configuration file is structured as follows:

*   **agents:** Configure agent behavior, tools, and models.
*   **lakeview:** Configure the summarization feature.
*   **model_providers:** Define API credentials and settings for different LLM providers.
*   **models:** Define specific model configurations with parameters.

Example YAML configuration:

```yaml
agents:
  trae_agent:
    enable_lakeview: true
    model: trae_agent_model
    max_steps: 200
    tools:
      - bash
      - str_replace_based_edit_tool
      - sequentialthinking
      - task_done

model_providers:
  anthropic:
    api_key: your_anthropic_api_key
    provider: anthropic
  openai:
    api_key: your_openai_api_key
    provider: openai

models:
  trae_agent_model:
    model_provider: anthropic
    model: claude-sonnet-4-20250514
    max_tokens: 4096
    temperature: 0.5
    top_p: 1
    max_retries: 10
    parallel_tool_calls: true
```

**WARNING:**
For Doubao users, please use the following base_url.

```
base_url=https://ark.cn-beijing.volces.com/api/v3/
```

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

### Environment Variables

*   `OPENAI_API_KEY` - OpenAI API key
*   `ANTHROPIC_API_KEY` - Anthropic API key
*   `GOOGLE_API_KEY` - Google Gemini API key
*   `OPENROUTER_API_KEY` - OpenRouter API key
*   `OPENROUTER_SITE_URL` - (Optional) Your site URL for OpenRouter rankings
*   `OPENROUTER_SITE_NAME` - (Optional) Your site name for OpenRouter rankings

## Available Tools

Trae Agent provides a comprehensive toolkit for file editing, bash execution, structured thinking, task completion, and JSON manipulation, with new tools actively being developed and existing ones continuously enhanced.

For detailed information about available tools, see [docs/tools.md](docs/tools.md).

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

*   LLM Interactions
*   Agent Steps
*   Tool Usage
*   Metadata

For more details, see [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Requirements

*   Python 3.12+
*   API key for your chosen provider:
    *   OpenAI API key
    *   Anthropic API key
    *   OpenRouter API key
    *   Google API key

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{traeresearchteam2025traeagent,
      title={Trae Agent: An LLM-based Agent for Software Engineering with Test-time Scaling},
      author={Trae Research Team and Pengfei Gao and Zhao Tian and Xiangxin Meng and Xinchen Wang and Ruida Hu and Yuanan Xiao and Yizhou Liu and Zhao Zhang and Junjie Chen and Cuiyun Gao and Yun Lin and Yingfei Xiong and Chao Peng and Xia Liu},
      year={2025},
      eprint={2507.23370},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2507.23370},
}
```

## Acknowledgments

We thank Anthropic for the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project that served as a valuable reference.