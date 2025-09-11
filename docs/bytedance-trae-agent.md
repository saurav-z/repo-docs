# Trae Agent: Your AI-Powered Software Engineering Assistant

**Trae Agent** is an innovative LLM-based agent designed to automate and simplify software engineering tasks, offering a flexible and powerful CLI interface. [Check out the original repository](https://github.com/bytedance/trae-agent) and explore its capabilities!

[![arXiv:2507.23370](https://img.shields.io/badge/TechReport-arXiv%3A2507.23370-b31a1b)](https://arxiv.org/abs/2507.23370)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent empowers developers and researchers to automate complex software engineering workflows using natural language instructions and a variety of LLM providers and tools. With its **research-friendly design**, Trae Agent is a valuable platform for studying AI agent architectures, conducting ablation studies, and developing novel agent capabilities.

**Project Status:** Actively developed; see [docs/roadmap.md](docs/roadmap.md) and [CONTRIBUTING](CONTRIBUTING.md) for contributing.

## Key Features

*   **✨ Comprehensive Toolset:** File editing, bash execution, sequential thinking, and more.
*   **🤖 Multi-LLM Support:** Compatible with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   **🎯 Interactive Mode:** Conversational interface for iterative development.
*   **📊 Detailed Trajectory Recording:** Detailed logging of all agent actions for debugging and analysis.
*   **⚙️ Flexible Configuration:** YAML-based configuration with environment variable support.
*   **🚀 Easy Installation:** Simple pip-based installation using `uv`.
*   **🌊 Lakeview**: Provides short and concise summarization for agent steps.

## 🚀 Installation

### Prerequisites

*   UV (https://docs.astral.sh/uv/)
*   API keys for your chosen LLM providers (OpenAI, Anthropic, Google Gemini, etc.)

### Setup

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv sync --all-extras
source .venv/bin/activate
```

## ⚙️ Configuration

### YAML Configuration (Recommended)

1.  Copy the example configuration:

    ```bash
    cp trae_config.yaml.example trae_config.yaml
    ```

2.  Edit `trae_config.yaml` with your API credentials and preferences.  *Remember to keep API keys secure!*

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

### Using Base URL

In some cases, we need to use a custom URL for the api. Just add the `base_url` field after `provider`, take the following config as an example:

```
openai:
    api_key: your_openrouter_api_key
    provider: openai
    base_url: https://openrouter.ai/api/v1
```

### Environment Variables (Alternative)

Configure API keys using environment variables in your `.env` file:

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

Enable Model Context Protocol (MCP) by adding an `mcp_servers` section:

```yaml
mcp_servers:
  playwright:
    command: npx
    args:
      - "@playwright/mcp@0.0.27"
```

**Configuration Priority:** Command-line arguments > Configuration file > Environment variables > Default values

**Legacy JSON Configuration:** See [docs/legacy_config.md](docs/legacy_config.md) for legacy JSON format.  YAML is recommended.

## 📖 Usage

### Basic Commands

```bash
# Simple task execution
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

In interactive mode:
- Type any task description to execute it
- `status` - Show agent information
- `help` - Show available commands
- `clear` - Clear the screen
- `exit` or `quit` - End the session

## 🛠️ Advanced Features

### Available Tools

Trae Agent includes file editing, bash execution, structured thinking, and task completion tools; see [docs/tools.md](docs/tools.md) for details.

### Trajectory Recording

Trae Agent automatically records detailed execution trajectories:

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_YYYYMMDD_HHMMSS.json

# Custom trajectory file
trae-cli run "Optimize database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain LLM interactions, agent steps, tool usage, and metadata; see [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## 🔧 Development

### Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Troubleshooting

**Import Errors:**
```bash
PYTHONPATH=. trae-cli run "your task"
```

**API Key Issues:**
```bash
# Verify API keys
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

Licensed under the MIT License; see [LICENSE](LICENSE).

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

Thanks to Anthropic for their [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project.