<!--
  SPDX-License-Identifier: MIT
-->
# Trae Agent: Your AI-Powered Software Engineering Assistant

Trae Agent is an advanced LLM-based agent designed to streamline software engineering tasks with natural language commands.  Explore the power of AI for coding! [Explore the original repository on GitHub](https://github.com/bytedance/trae-agent).

**Key Features:**

*   🚀 **Multi-LLM Support:** Integrates with leading LLM providers including OpenAI, Anthropic, Doubao, Azure, OpenRouter, Ollama, and Google Gemini APIs.
*   🛠️ **Rich Tool Ecosystem:** Equipped with a comprehensive suite of tools for file editing, bash execution, sequential reasoning, and more, enabling complex workflows.
*   🎯 **Interactive Mode:** Engage in iterative development with a conversational interface, simplifying experimentation and refinement.
*   📊 **Trajectory Recording:** Detailed logging of all agent actions for robust debugging, analysis, and performance evaluation.
*   ⚙️ **Flexible Configuration:** Utilize JSON-based configuration with environment variable support for easy customization and management.
*   🌊 **Lakeview Summarization**: Provides clear, concise summaries of agent steps for easy monitoring.
*   🚀 **Easy Installation:**  Simple pip-based installation makes setup straightforward.
*   🧪 **Research-Friendly Design:** Designed for research and development, allowing easy modification and extension of the agent's architecture.

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

Configure Trae Agent with the config file.

**Configuration Setup:**

1.  **Copy the example configuration file:**

    ```bash
    cp trae_config.json.example trae_config.json
    ```

2.  **Edit `trae_config.json` and replace the placeholder values with your actual credentials:**
    *   Replace `"your_openai_api_key"` with your actual OpenAI API key
    *   Replace `"your_anthropic_api_key"` with your actual Anthropic API key
    *   Replace `"your_google_api_key"` with your actual Google API key
    *   Replace `"your_azure_base_url"` with your actual Azure base URL
    *   Replace other placeholder URLs and API keys as needed

    **Note:** The `trae_config.json` file is ignored by git.

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

Use [python-dotenv](https://pypi.org/project/python-dotenv/) to add `MODEL_API_KEY="My API Key"` to your `.env` file.

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

The `trae-cli` command has multiple subcommands.

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

-   Type a task description.
-   Use `status` for agent info.
-   Use `help` for commands.
-   Use `clear` to clear the screen.
-   Use `exit` or `quit` to end the session.

#### `trae show-config` - Configuration Status

```bash
trae-cli show-config

# With custom config file
trae-cli show-config --config-file my_config.json
```

### Configuration

Trae Agent uses a JSON configuration file.  See `trae_config.json` for structure.

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

*   `OPENAI_API_KEY` - OpenAI API key
*   `ANTHROPIC_API_KEY` - Anthropic API key
*   `GOOGLE_API_KEY` - Google Gemini API key
*   `OPENROUTER_API_KEY` - OpenRouter API key
*   `OPENROUTER_SITE_URL` - (Optional) Your site URL for OpenRouter rankings
*   `OPENROUTER_SITE_NAME` - (Optional) Your site name for OpenRouter rankings

## 🛠️ Available Tools

Trae Agent includes tools for file editing, bash execution, structured thinking, task completion, and JSON manipulation. Find details in [docs/tools.md](docs/tools.md).

## 📊 Trajectory Recording

Trae Agent logs execution trajectories.

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_20250612_220546.json

# Custom trajectory file
trae-cli run "Optimize the database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain:

-   LLM Interactions
-   Agent Steps
-   Tool Usage
-   Metadata

See [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

1.  Fork the repo.
2.  Set up a dev install:

    ```bash
    make install-dev
    ```

3.  Create a branch.
4.  Make changes.
5.  Add tests.
6.  Pre-commit check.

    ```bash
     make pre-commit
     or:
     make uv-pre-commit
    ```

     If having formatting error, please try:

    ```
     make fix-format
    ```

7.  Commit.
8.  Push.
9.  Open a Pull Request.

### Development Guidelines

*   Follow PEP 8.
*   Add tests.
*   Update documentation.
*   Use type hints.
*   Ensure tests pass.

## 📋 Requirements

*   Python 3.12+
*   API key:  OpenAI, Anthropic, OpenRouter, Google Gemini.

## 🔧 Troubleshooting

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

## 📄 License

MIT License - see [LICENSE](LICENSE).

## 🙏 Acknowledgments

Thanks to Anthropic for the [anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts) project.