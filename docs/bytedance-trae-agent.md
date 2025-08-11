# Trae Agent: Your AI-Powered Software Engineering Assistant

**Trae Agent empowers developers with an LLM-based agent designed to automate and accelerate software engineering tasks, offering a transparent and extensible architecture.** ([Original Repo](https://github.com/bytedance/trae-agent))

[![arXiv:2507.23370](https://img.shields.io/badge/TechReport-arXiv%3A2507.23370-b31a1b)](https://arxiv.org/abs/2507.23370)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent is a cutting-edge AI agent designed for software engineering, offering a powerful CLI interface and a flexible, research-friendly architecture.  This enables both developers and researchers to easily study, modify, and extend the agent's capabilities.

**Key Features:**

*   **Multi-LLM Support:** Integrates with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   **Rich Tool Ecosystem:** Includes file editing, bash execution, sequential thinking, and a variety of tools to automate complex tasks.
*   **Interactive Mode:** Provides a conversational interface for iterative development and easy task refinement.
*   **Trajectory Recording:** Detailed logging of all agent actions for debugging, analysis, and reproducibility.
*   **Flexible Configuration:** YAML-based configuration with environment variable support for easy setup and customization.
*   **Easy Installation:** Simple installation using `pip` and UV.
*   **Lakeview Summarization:** Concise summaries of agent steps for improved transparency.

## Getting Started

### Installation

1.  **Requirements:** UV (https://docs.astral.sh/uv/) and API keys for your chosen provider (OpenAI, Anthropic, Google Gemini, OpenRouter, etc.).
2.  **Setup:**

    ```bash
    git clone https://github.com/bytedance/trae-agent.git
    cd trae-agent
    uv sync --all-extras
    source .venv/bin/activate
    ```

### Configuration

Trae Agent supports YAML configuration (recommended) and environment variables.

#### YAML Configuration

1.  Copy the example configuration:

    ```bash
    cp trae_config.yaml.example trae_config.yaml
    ```

2.  Edit `trae_config.yaml` with your API credentials and preferences. A basic example is provided below:

    ```yaml
    agents:
      trae_agent:
        enable_lakeview: true
        model: trae_agent_model  # the model configuration name for Trae Agent
        max_steps: 200  # max number of agent steps
        tools:  # tools used with Trae Agent
          - bash
          - str_replace_based_edit_tool
          - sequentialthinking
          - task_done

    model_providers:  # model providers configuration
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
    ```

    **Note:** `trae_config.yaml` is ignored by Git to protect your API keys.

#### Environment Variables

Alternatively, set API keys using environment variables in your `.env` file:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3/"
```

#### Configuration Priority

The priority of configuration settings is: Command-line arguments > Configuration file > Environment variables > Default values.

## Usage

### Basic Commands

```bash
# Run a task
trae-cli run "Create a hello world Python script"

# Check configuration
trae-cli show-config

# Interactive mode
trae-cli interactive
```

### Provider-Specific Examples

```bash
# OpenAI
trae-cli run "Fix the bug in main.py" --provider openai --model gpt-4o

# Anthropic
trae-cli run "Add unit tests" --provider anthropic --model claude-sonnet-4-20250514

# Google Gemini
trae-cli run "Optimize this algorithm" --provider google --model gemini-2.5-flash

# OpenRouter (access to multiple providers)
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"
trae-cli run "Generate documentation" --provider openrouter --model "openai/gpt-4o"

# Doubao
trae-cli run "Refactor the database module" --provider doubao --model doubao-seed-1.6

# Ollama (local models)
trae-cli run "Comment this code" --provider ollama --model qwen3
```

### Advanced Options

```bash
# Custom working directory
trae-cli run "Add tests for utils module" --working-dir /path/to/project

# Save execution trajectory
trae-cli run "Debug authentication" --trajectory-file debug_session.json

# Force patch generation
trae-cli run "Update API endpoints" --must-patch

# Interactive mode with custom settings
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

### Interactive Mode Commands

In interactive mode, use:
-   Type any task description to execute it
-   `status` - Show agent information
-   `help` - Show available commands
-   `clear` - Clear the screen
-   `exit` or `quit` - End the session

## Advanced Features

### Available Tools

Trae Agent provides a comprehensive toolkit for software engineering, including file editing, bash execution, and more.  Refer to [docs/tools.md](docs/tools.md) for a detailed tool list.

### Trajectory Recording

Trae Agent automatically records execution trajectories:

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_YYYYMMDD_HHMMSS.json

# Custom trajectory file
trae-cli run "Optimize database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain LLM interactions, agent steps, tool usage, and execution metadata. For more details, see [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## Development

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

### Troubleshooting

*   **Import Errors:**  `PYTHONPATH=. trae-cli run "your task"`
*   **API Key Issues:**  `echo $OPENAI_API_KEY` and `trae-cli show-config` to verify.
*   **Command Not Found:** `uv run trae-cli run "your task"`
*   **Permission Errors:** `chmod +x /path/to/your/project`

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

We thank Anthropic for their [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project, which provided valuable reference for the tool ecosystem.