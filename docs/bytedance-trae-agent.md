# Trae Agent: Your AI Companion for Software Engineering Tasks

**Trae Agent** is an open-source, LLM-powered agent designed to streamline software engineering workflows. ([See the original repo](https://github.com/bytedance/trae-agent))

[![arXiv:2507.23370](https://img.shields.io/badge/TechReport-arXiv%3A2507.23370-b31a1b)](https://arxiv.org/abs/2507.23370)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

Trae Agent simplifies complex software engineering tasks by understanding natural language instructions and executing them using a suite of powerful tools, making development more efficient and accessible.

For in-depth technical details, please refer to our [technical report](https://arxiv.org/abs/2507.23370).

**Project Status:** Actively being developed; refer to [docs/roadmap.md](docs/roadmap.md) and [CONTRIBUTING](CONTRIBUTING.md) for participation.

**Key Differentiator:** Trae Agent’s **_research-friendly design_** with a transparent, modular architecture encourages modification and extension, facilitating AI agent architecture studies, ablation studies, and the development of new agent capabilities.

## Key Features

*   ✅ **Multi-LLM Support:** Works with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   🛠️ **Rich Tool Ecosystem:** Includes file editing, bash execution, sequential reasoning, and task completion tools.
*   💬 **Interactive Mode:** Provides a conversational interface for iterative development and experimentation.
*   📊 **Trajectory Recording:** Detailed logging for debugging, analysis, and understanding agent behavior.
*   ⚙️ **Flexible Configuration:** YAML-based configuration with environment variable support for ease of customization.
*   🚀 **Easy Installation:** Simple pip-based installation for quick setup.
*   🌊 **Lakeview:** Provides short and concise summarisation for agent steps

## Getting Started

### Requirements

*   [UV](https://docs.astral.sh/uv/)
*   API key for your chosen provider (e.g., OpenAI, Anthropic, Google Gemini, OpenRouter)

### Installation

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv sync --all-extras
source .venv/bin/activate
```

## Configuration

### YAML Configuration (Recommended)

1.  Copy the example configuration:

```bash
cp trae_config.yaml.example trae_config.yaml
```

2.  Edit `trae_config.yaml` with your API credentials and preferences:

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

**Note:** The `trae_config.yaml` file is ignored by git to protect your API keys.

### Environment Variables (Alternative)

Configure API keys using environment variables in a `.env` file:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3/"
```

### MCP Services (Optional)

To enable Model Context Protocol (MCP) services:

```yaml
mcp_servers:
  playwright:
    command: npx
    args:
      - "@playwright/mcp@0.0.27"
```

**Configuration Priority:** Command-line arguments > Configuration file > Environment variables > Default values

**Legacy JSON Configuration:** See [docs/legacy_config.md](docs/legacy_config.md) for JSON format usage; YAML is now recommended.

## Usage

### Basic Commands

```bash
# Run a task
trae-cli run "Create a hello world Python script"

# Show configuration
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

*   Type a task description to execute it
*   `status` - Show agent information
*   `help` - Show available commands
*   `clear` - Clear the screen
*   `exit` or `quit` - End the session

## Advanced Features

### Available Tools

Trae Agent offers file editing, bash execution, structured thinking, and task completion tools. For details, see [docs/tools.md](docs/tools.md).

### Trajectory Recording

Trae Agent records detailed execution trajectories:

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_YYYYMMDD_HHMMSS.json

# Custom trajectory file
trae-cli run "Optimize database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain LLM interactions, agent steps, tool usage, and metadata. For more details, see [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## Development

### Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

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

This project is licensed under the MIT License; see the [LICENSE](LICENSE) file.

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

We thank Anthropic for the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project, a helpful reference for the tool ecosystem.