# Trae Agent: Your AI-Powered Software Engineering Assistant

**Trae Agent is an LLM-powered agent that streamlines software engineering tasks, making development faster and more efficient.** ([Original Repo](https://github.com/bytedance/trae-agent))

[![arXiv:2507.23370](https://img.shields.io/badge/TechReport-arXiv%3A2507.23370-b31a1b)](https://arxiv.org/abs/2507.23370)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent leverages the power of large language models (LLMs) to understand natural language instructions and automate complex software engineering workflows.  Its modular and extensible architecture makes it ideal for research, experimentation, and building innovative AI agent capabilities.

## Key Features

*   **🤖 Multi-LLM Support:** Works with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs, offering flexibility in model selection.
*   **🛠️ Rich Tool Ecosystem:** Includes file editing, bash execution, sequential thinking, and more to handle diverse tasks.
*   **🌊 Lakeview Summarization:** Provides concise summaries of agent steps for easy monitoring.
*   **🎯 Interactive Mode:** Engage in conversational development sessions for iterative refinement.
*   **📊 Trajectory Recording:** Detailed logging of all agent actions for comprehensive debugging and analysis.
*   **⚙️ Flexible Configuration:** Supports both YAML configuration files and environment variables for easy setup.
*   **🚀 Easy Installation:** Simple pip-based installation process to get you up and running quickly.

## Getting Started

### Requirements

*   UV ([https://docs.astral.sh/uv/](https://docs.astral.sh/uv/))
*   API keys for your chosen LLM providers (OpenAI, Anthropic, Google Gemini, etc.)

### Installation

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv sync --all-extras
source .venv/bin/activate
```

## Configuration

Choose from YAML configuration files (recommended) or environment variables to configure Trae Agent.

### YAML Configuration

1.  Copy the example configuration:
    ```bash
    cp trae_config.yaml.example trae_config.yaml
    ```
2.  Edit `trae_config.yaml` with your API keys and preferences.

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

### Environment Variables

Alternatively, set API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3/"
```

**Configuration Priority:** Command-line arguments > Configuration file > Environment variables > Default values

## Usage

### Basic Commands

```bash
# Run a task
trae-cli run "Create a hello world Python script"

# View configuration
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
# Set working directory
trae-cli run "Add tests for utils module" --working-dir /path/to/project

# Save execution trajectory
trae-cli run "Debug authentication" --trajectory-file debug_session.json

# Force patch generation
trae-cli run "Update API endpoints" --must-patch

# Interactive mode with custom settings
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

### Interactive Mode Commands

In interactive mode, use these commands:

*   Enter a task description to execute
*   `status` - View agent information
*   `help` - Display available commands
*   `clear` - Clear the screen
*   `exit` or `quit` - End the session

## Advanced Features

### Available Tools

Trae Agent provides a comprehensive toolkit for software engineering. Refer to [docs/tools.md](docs/tools.md) for detailed information.

### Trajectory Recording

Trae Agent automatically records execution trajectories for debugging and analysis.

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_YYYYMMDD_HHMMSS.json

# Custom trajectory file
trae-cli run "Optimize database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain LLM interactions, agent steps, tool usage, and execution metadata. See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md) for details.

## Development

### Contributing

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

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

We thank Anthropic for building the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project that served as a valuable reference for the tool ecosystem.