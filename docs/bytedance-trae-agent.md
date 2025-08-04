# Trae Agent: Your AI Assistant for Streamlined Software Engineering

[![arXiv:2507.23370](https://img.shields.io/badge/TechReport-arXiv%3A2507.23370-b31a1b)](https://arxiv.org/abs/2507.23370)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

**Trae Agent is a powerful, LLM-powered agent designed to revolutionize software engineering tasks, enabling you to build, debug, and optimize code with ease.**  Visit the original repository: [https://github.com/bytedance/trae-agent](https://github.com/bytedance/trae-agent).

**Key Features:**

*   **Multi-LLM Support:** Seamlessly integrates with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs, offering flexibility in model selection.
*   **Rich Tool Ecosystem:**  Provides a comprehensive suite of tools, including file editing, bash execution, and sequential thinking, empowering you to tackle diverse engineering challenges.
*   **Interactive Mode:**  Engage in conversational development with an interactive interface for iterative coding and refinement.
*   **Trajectory Recording:**  Detailed logging of agent actions allows for easy debugging, analysis, and optimization of your workflows.
*   **Flexible Configuration:** YAML-based configuration with environment variable support simplifies customization and deployment.
*   **Easy Installation:**  Quickly set up with a straightforward pip-based installation process using `uv` or `make`.
*   **Lakeview summarization**: Provides short and concise summarization for agent steps

**Project Status:** Actively developed; consult [docs/roadmap.md](docs/roadmap.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for contribution opportunities.

## Getting Started

### Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) to setup the project.

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv venv
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate
```

or use make.

```bash
make uv-venv
make uv-sync

# Activate the virtual environment
source .venv/bin/activate
```

### Configuration

Configure `trae_config.yaml` with your API keys.  See the `trae_config.yaml.example` file for reference.  Environment variables are also supported.

1.  **Copy the example configuration file:**

    ```bash
    cp trae_config.yaml.example trae_config.yaml
    ```

2.  **Edit `trae_config.yaml` and replace the placeholder values with your actual credentials:**
    *   Replace `your_anthropic_api_key` with your actual Anthropic API key
    *   Add additional model providers as needed (OpenAI, Google, Azure, etc.)
    *   Configure your preferred models and settings

**Note:** The `trae_config.yaml` file is ignored by git to prevent accidentally committing your API keys.

You can also set your API keys as environment variables:

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

Although you can pass your API key directly using the `api_key` argument, we suggest utilizing [python-dotenv](https://pypi.org/project/python-dotenv/) to add `MODEL_API_KEY="My API Key"` to your `.env` file. This approach helps prevent your API key from being exposed in source control.

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

The `trae` command provides several subcommands:

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

*   Type any task description to execute it.
*   Use `status` to see agent information.
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

*   Uses a YAML configuration file (`trae_config.yaml`).
*   Example configuration structure provided in `trae_config.yaml.example`.
*   Sections: `agents`, `lakeview`, `model_providers`, `models`.

Example configuration:

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

*   `OPENAI_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `GOOGLE_API_KEY`
*   `OPENROUTER_API_KEY`
*   `OPENROUTER_SITE_URL` (Optional)
*   `OPENROUTER_SITE_NAME` (Optional)

## Available Tools

Trae Agent provides a versatile toolkit for file editing, bash execution, and structured thinking. New tools are in development, and existing ones are continuously improved.

For details, refer to [docs/tools.md](docs/tools.md).

## Trajectory Recording

Automatically records execution trajectories for debugging and analysis.

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files include:

*   LLM interactions
*   Agent steps
*   Tool usage
*   Metadata

See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md) for more details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Requirements

*   Python 3.12+
*   API keys for your chosen LLM provider (OpenAI, Anthropic, OpenRouter, Google Gemini).

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

Licensed under the MIT License.  See [LICENSE](LICENSE) for details.

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

Thanks to Anthropic for the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project.