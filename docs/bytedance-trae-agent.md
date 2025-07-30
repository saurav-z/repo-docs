# Trae Agent: Your AI-Powered Software Engineering Assistant

Trae Agent is a powerful, LLM-driven CLI tool designed to automate and streamline software engineering tasks using various tools and LLM providers. [Explore the original repository here](https://github.com/bytedance/trae-agent).

**Key Features:**

*   **🤖 Multi-LLM Support:** Integrates with OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   **🛠️ Rich Tool Ecosystem:** Includes file editing, bash execution, and sequential thinking capabilities.
*   **🎯 Interactive Mode:** Offers a conversational interface for iterative development and debugging.
*   **📊 Detailed Trajectory Recording:** Logs all agent actions for thorough debugging and analysis.
*   **⚙️ Flexible Configuration:** Uses JSON-based configuration with environment variable support.
*   **🚀 Easy Installation:** Simplifies setup with pip-based installation.
*   **🌊 Lakeview:** Provides short and concise summarisation for agent steps

Trae Agent's modular and transparent architecture makes it ideal for research and development, enabling users to easily modify, extend, and analyze AI agent architectures.

## 🚀 Quick Start

### Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) to set up the project.

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

### Setup API Keys

**Configuration Setup:**

1.  **Copy the example configuration file:**

    ```bash
    cp trae_config.json.example trae_config.json
    ```

2.  **Edit `trae_config.json`:** Replace placeholder values with your API keys for OpenAI, Anthropic, Google, and other providers.  The `trae_config.json` file is ignored by git.

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

### Basic Usage

```bash
# Run a simple task
trae-cli run "Create a hello world Python script"

# Run with Doubao
trae-cli run "Create a hello world Python script" --provider doubao --model doubao-seed-1.6

# Run with Google Gemini
trae-cli run "Create a hello world Python script" --provider google --model gemini-2.5-flash
```

## 📖 Usage

### Command Line Interface

*   **`trae run`**:  Execute a task, specifying the provider and model.  Use `--trajectory-file` to save debugging information.
*   **`trae interactive`**:  Engage in a conversational mode for iterative task refinement.
*   **`trae show-config`**: View the current configuration settings.

### Configuration

The agent uses a JSON configuration file (`trae_config.json`) with command-line arguments taking precedence, followed by the configuration file, environment variables, and default values.

**Configuration Priority:**

1.  Command-line arguments (highest)
2.  Configuration file values
3.  Environment variables
4.  Default values (lowest)

**Popular OpenRouter Models:**

-   `openai/gpt-4o` - Latest GPT-4 model
-   `anthropic/claude-3-5-sonnet` - Excellent for coding tasks
-   `google/gemini-pro` - Strong reasoning capabilities
-   `meta-llama/llama-3.1-405b` - Open source alternative
-   `openai/gpt-4o-mini` - Fast and cost-effective

### Environment Variables

*   `OPENAI_API_KEY`
*   `ANTHROPIC_API_KEY`
*   `GOOGLE_API_KEY`
*   `OPENROUTER_API_KEY`
*   `OPENROUTER_SITE_URL` (optional)
*   `OPENROUTER_SITE_NAME` (optional)

## 🛠️ Available Tools

Trae Agent features a diverse set of tools, including file editing and bash execution, and the toolset is constantly expanding.  See [docs/tools.md](docs/tools.md) for detailed information.

## 📊 Trajectory Recording

Trae Agent automatically records detailed execution trajectories for debugging and analysis:

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:

-   **LLM Interactions**: All messages, responses, and tool calls
-   **Agent Steps**: State transitions and decision points
-   **Tool Usage**: Which tools were called and their results
-   **Metadata**: Timestamps, token usage, and execution metrics

For more details, see [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## 🤝 Contributing

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines, including setting up a development environment, style guidelines, and creating pull requests.

## 📋 Requirements

*   Python 3.12+
*   API Keys for your chosen LLM providers (OpenAI, Anthropic, OpenRouter, Google Gemini, etc.)

## 🔧 Troubleshooting

*   **Import Errors**:  Try setting `PYTHONPATH=.` before running `trae-cli run`.
*   **API Key Issues**: Verify API keys with `echo $YOUR_API_KEY` and `trae-cli show-config`.
*   **Permission Errors**: Ensure you have appropriate file permissions.
*   **Command not found Errors:** Use `uv run trae-cli xxxxx`

## 📄 License

This project is licensed under the MIT License (see [LICENSE](LICENSE)).

## 🙏 Acknowledgments

Thanks to Anthropic for the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project, which served as a valuable reference.