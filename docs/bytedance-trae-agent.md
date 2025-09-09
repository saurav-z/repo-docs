# Trae Agent: Your AI-Powered Software Engineering Assistant

[![arXiv:2507.23370](https://img.shields.io/badge/TechReport-arXiv%3A2507.23370-b31a1b)](https://arxiv.org/abs/2507.23370)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

**Trae Agent** is an advanced, LLM-powered CLI agent designed to streamline software engineering tasks with natural language commands.  [Explore the original repository](https://github.com/bytedance/trae-agent).

**Key Features:**

*   **Multi-LLM Support:** Seamlessly integrates with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini.
*   **Rich Tool Ecosystem:** Offers file editing, bash execution, sequential thinking, and other essential tools for efficient software development.
*   **Interactive Mode:** Provides a conversational interface for iterative development and debugging.
*   **Transparent & Modular Architecture:** Easily modifiable and extendable, ideal for research and experimentation.
*   **Trajectory Recording:** Detailed logging of agent actions for thorough analysis and debugging.
*   **Flexible Configuration:** Supports YAML configuration with environment variable override for easy customization.
*   **Docker Integration**: Run tasks in isolated environments for increased security and reproducibility.
*   **Lakeview Summarization**: Efficiently summarize agent steps.

## 🚀 Getting Started

### Installation

1.  **Requirements:**
    *   UV (https://docs.astral.sh/uv/)
    *   API key for your chosen provider (e.g., OpenAI, Anthropic)

2.  **Setup:**

    ```bash
    git clone https://github.com/bytedance/trae-agent.git
    cd trae-agent
    uv sync --all-extras
    source .venv/bin/activate
    ```

## ⚙️ Configuration

### YAML Configuration (Recommended)

1.  **Copy the example configuration:**

    ```bash
    cp trae_config.yaml.example trae_config.yaml
    ```

2.  **Edit `trae_config.yaml`** to include your API credentials and preferences:

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
    ```

    **Note:** The `trae_config.yaml` file is excluded from Git.

### Using Base URL
In some cases, you need to use a custom URL for the api. Just add the `base_url` field after `provider`, take the following config as an example:

```
openai:
    api_key: your_openrouter_api_key
    provider: openai
    base_url: https://openrouter.ai/api/v1
```
**Note:** For field formatting, use spaces only. Tabs (\t) are not allowed.

### Environment Variables (Alternative)

Configure API keys via environment variables stored in a `.env` file:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="your-openai-base-url"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_BASE_URL="your-anthropic-base-url"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_BASE_URL="your-google-base-url"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3/"
```

### MCP Services (Optional)

To enable Model Context Protocol (MCP) services, add an `mcp_servers` section to your configuration:

```yaml
mcp_servers:
  playwright:
    command: npx
    args:
      - "@playwright/mcp@0.0.27"
```

**Configuration Priority:** Command-line arguments > Configuration file > Environment variables > Default values

## 📖 Usage

### Basic Commands

```bash
# Execute a task
trae-cli run "Create a hello world Python script"

# Show your configuration
trae-cli show-config

# Enter interactive mode
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
# Use a custom working directory
trae-cli run "Add tests for utils module" --working-dir /path/to/project

# Save your execution trajectory
trae-cli run "Debug authentication" --trajectory-file debug_session.json

# Force patch generation
trae-cli run "Update API endpoints" --must-patch

# Interactive mode with specific settings
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

## Docker Mode Commands
### Preparation
**Important**: You need to make sure Docker is configured in your environment.

### Usage
```bash
# Specify a Docker image to run the task in a new container
trae-cli run "Add tests for utils module" --docker-image python:3.11

# Specify a Docker image to run the task in a new container and mount the directory
trae-cli run "write a script to print helloworld" --docker-image python:3.12 --working-dir test_workdir/

# Attach to an existing Docker container by ID (`--working-dir` is invalid with `--docker-container-id`)
trae-cli run "Update API endpoints" --docker-container-id 91998a56056c

# Specify an absolute path to a Dockerfile to build an environment
trae-cli run "Debug authentication" --dockerfile-path test_workspace/Dockerfile

# Specify a path to a local Docker image file (tar archive) to load
trae-cli run "Fix the bug in main.py" --docker-image-file test_workspace/trae_agent_custom.tar

# Remove the Docker container after finishing the task (keep default)
trae-cli run "Add tests for utils module" --docker-image python:3.11 --docker-keep false
```

### Interactive Mode Commands

In interactive mode, you can use:
- Type any task description to execute it
- `status` - Show agent information
- `help` - Show available commands
- `clear` - Clear the screen
- `exit` or `quit` - End the session

## 🛠️ Advanced Features

### Available Tools

Trae Agent offers a rich set of tools for software engineering tasks, including file editing, bash execution, structured thinking, and task completion. For a comprehensive list of tools and their functionalities, see [docs/tools.md](docs/tools.md).

### Trajectory Recording

Trae Agent automatically saves detailed execution trajectories for thorough debugging and analysis:

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_YYYYMMDD_HHMMSS.json

# Specify a custom trajectory file
trae-cli run "Optimize database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain LLM interactions, agent steps, tool usage, and execution metadata. See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md) for more details.

## 🔧 Development

### Contributing

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

### Troubleshooting

**Import Errors:**

```bash
PYTHONPATH=. trae-cli run "your task"
```

**API Key Issues:**

```bash
# Verify your API keys
echo $OPENAI_API_KEY
trae-cli show-config
```

**Command Not Found:**

```bash
uv run trae-cli run "your task"
```

**Permission Errors:**

```bash
chmod +x /path/to/your/project
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✍️ Citation

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

## 🙏 Acknowledgments

We thank Anthropic for building the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project that served as a valuable reference for the tool ecosystem.