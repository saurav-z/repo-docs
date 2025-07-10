# Trae Agent: Your AI-Powered Software Engineering Assistant

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Alpha](https://img.shields.io/badge/Status-Alpha-red)](https://github.com/bytedance/trae-agent)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)

**Trae Agent** is an advanced, LLM-based agent that helps you automate and streamline software engineering tasks with natural language instructions.  [Explore the original repository](https://github.com/bytedance/trae-agent) for detailed information and contributions.

**Key Features:**

*   🤖 **Multi-LLM Support:**  Works seamlessly with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   🛠️ **Rich Tool Ecosystem:**  Includes file editing, bash execution, sequential thinking, and more to handle diverse engineering tasks.
*   🌊 **Lakeview Summarization:** Provides concise summaries of agent steps for clear task progression.
*   🎯 **Interactive Mode:**  Offers a conversational interface for iterative development and real-time feedback.
*   📊 **Trajectory Recording:** Detailed logging of all agent actions for debugging, analysis, and understanding behavior.
*   ⚙️ **Flexible Configuration:**  Uses JSON-based configuration with environment variable support for easy customization.
*   🚀 **Easy Installation:** Simple installation via pip.

**Project Status:** Actively under development. See [docs/roadmap.md](docs/roadmap.md) and [CONTRIBUTING](CONTRIBUTING.md) for contributing.

**Research-Friendly Design:** Trae Agent's modular architecture is designed to be easily modified, extended, and analyzed by researchers and developers, making it ideal for studying AI agent architectures and developing new capabilities.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
make install
```

It's highly recommended to use [uv](https://docs.astral.sh/uv/) for project setup.

### Setup API Keys

Configure Trae Agent using the config file (recommended).

Alternatively, set API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Doubao (also works with other OpenAI-compatible model providers)
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_API_BASE_URL="your-model-provider-base-url"

# OpenRouter
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# Optional: OpenRouter site ranking
export OPENROUTER_SITE_URL="https://your-site.com"
export OPENROUTER_SITE_NAME="Your App Name"

# Optional: OpenAI compatible API provider
export OPENAI_BASE_URL="your-openai-compatible-api-base-url"
```

Use [python-dotenv](https://pypi.org/project/python-dotenv/) for secure key management: add `MODEL_API_KEY="My API Key"` to your `.env` file.

### Basic Usage

```bash
# Create a hello world Python script
trae-cli run "Create a hello world Python script"

# Use Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Use Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## 📖 Usage

### Command Line Interface

*   **`trae run` - Execute a Task:**

    ```bash
    # Basic task execution
    trae-cli run "Create a Python script that calculates fibonacci numbers"

    # Specify provider and model
    trae-cli run "Fix the bug in main.py" --provider anthropic --model claude-sonnet-4-20250514

    # Use OpenRouter
    trae-cli run "Optimize this code" --provider openrouter --model "openai/gpt-4o"
    trae-cli run "Add documentation" --provider openrouter --model "anthropic/claude-3-5-sonnet"

    # Use Google Gemini
    trae-cli run "Implement a data parsing function" --provider google --model gemini-2.5-pro

    # Set working directory
    trae-cli run "Add unit tests for the utils module" --working-dir /path/to/project

    # Save trajectory
    trae-cli run "Refactor the database module" --trajectory-file debug_session.json

    # Force patch generation
    trae-cli run "Update the API endpoints" --must-patch
    ```

*   **`trae interactive` - Interactive Mode:**

    ```bash
    # Start interactive session
    trae-cli interactive

    # With custom configuration
    trae-cli interactive --provider openai --model gpt-4o --max-steps 30
    ```

    Interactive mode commands: `status`, `help`, `clear`, `exit` or `quit`.

*   **`trae show-config` - Configuration Status:**

    ```bash
    trae-cli show-config

    # With custom config file
    trae-cli show-config --config-file my_config.json
    ```

### Configuration

Trae Agent uses a JSON configuration file (`trae_config.json`):

```json
{
  "default_provider": "anthropic",
  "max_steps": 20,
  "enable_lakeview": true,
  "model_providers": {
    "openai": {
      "api_key": "your_openai_api_key",
      "model": "gpt-4o",
      "max_tokens": 128000,
      "temperature": 0.5,
      "top_p": 1,
      "max_retries": 10
    },
    "anthropic": {
      "api_key": "your_anthropic_api_key",
      "model": "claude-sonnet-4-20250514",
      "max_tokens": 4096,
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 0,
      "max_retries": 10
    },
    "google": {
      "api_key": "your_google_api_key",
      "model": "gemini-2.5-pro",
      "max_tokens": 128000,
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 0,
      "max_retries": 10
    },
    "azure": {
      "api_key": "you_azure_api_key",
      "base_url": "your_azure_base_url",
      "api_version": "2024-03-01-preview",
      "model": "model_name",
      "max_tokens": 4096,
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 0,
      "max_retries": 10
    },
    "ollama": {
      "api_key": "ollama",
      "base_url": "http://localhost:11434",
      "model": "model_name",
      "max_tokens": 4096,
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 0,
      "max_retries": 10
    },
    "openrouter": {
      "api_key": "your_openrouter_api_key",
      "model": "openai/gpt-4o",
      "max_tokens": 4096,
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 0,
      "max_retries": 10
    },
    "doubao": {
      "api_key": "you_doubao_api_key",
      "model": "model_name",
      "base_url": "your_doubao_base_url",
      "max_tokens": 8192,
      "temperature": 0.5,
      "top_p": 1,
      "max_retries": 20
    }
  },
  "lakeview_config": {
    "model_provider": "anthropic",
    "model_name": "claude-sonnet-4-20250514"
  }
}
```

**WARNING:** For Doubao users, please use this base_url:

```
base_url=https://ark.cn-beijing.volces.com/api/v3/
```

**Configuration Priority:**

1.  Command-line arguments
2.  Configuration file values
3.  Environment variables
4.  Default values

**Example Usage**

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

*   `openai/gpt-4o`
*   `anthropic/claude-3-5-sonnet`
*   `google/gemini-pro`
*   `meta-llama/llama-3.1-405b`
*   `openai/gpt-4o-mini`

### Environment Variables

*   `OPENAI_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `GOOGLE_API_KEY`
*   `OPENROUTER_API_KEY`
*   `GOOGLE_API_KEY`
*   `OPENROUTER_SITE_URL` (Optional)
*   `OPENROUTER_SITE_NAME` (Optional)

## 🛠️ Available Tools

Trae Agent includes tools for file editing, bash execution, structured thinking, task completion, and JSON manipulation. See [docs/tools.md](docs/tools.md) for details.

## 📊 Trajectory Recording

Trae Agent automatically saves execution trajectories for debugging and analysis:

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:

*   LLM Interactions
*   Agent Steps
*   Tool Usage
*   Metadata

See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md) for more.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1.  Fork the repository
2.  Set up a development install (`make install-dev pre-commit-install`)
3.  Create a feature branch (`git checkout -b feature/amazing-feature`)
4.  Make your changes
5.  Add tests
6.  Commit (`git commit -m 'Add amazing feature'`)
7.  Push (`git push origin feature/amazing-feature`)
8.  Open a Pull Request

### Development Guidelines

*   Follow PEP 8
*   Add tests
*   Update documentation
*   Use type hints
*   Ensure tests pass

## 📋 Requirements

*   Python 3.12+
*   API keys for your chosen providers.

## 🔧 Troubleshooting

### Common Issues

**Import Errors:**

```bash
# Set PYTHONPATH
PYTHONPATH=. trae-cli run "your task"
```

**API Key Issues:**

```bash
# Verify keys
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
echo $OPENROUTER_API_KEY
echo $GOOGLE_API_KEY

# Check configuration
trae-cli show-config
```

**Permission Errors:**

```bash
# Ensure permissions
chmod +x /path/to/your/project
```

## 📄 License

MIT License - see the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Thanks to Anthropic for their [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project.