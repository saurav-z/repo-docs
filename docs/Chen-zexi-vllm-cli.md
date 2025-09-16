<!-- SEO-optimized README.md -->
# vLLM CLI: Command-Line Interface for Serving Large Language Models

**Easily serve and manage your Large Language Models (LLMs) with vLLM, offering both interactive and command-line modes.** ([View on GitHub](https://github.com/Chen-zexi/vllm-cli))

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Key Features

*   **Interactive Mode:** Navigate and manage your LLMs with a user-friendly terminal interface.
*   **Command-Line Mode:** Automate and script your LLM serving with direct CLI commands.
*   **Model Management:** Seamlessly discover and load models from Hugging Face Hub and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles or create custom settings for optimal performance.
*   **Server Monitoring:** Monitor the status of your vLLM servers in real-time.
*   **System Information:** Check GPU, memory, and CUDA compatibility to ensure a smooth experience.
*   **Advanced Configuration:** Fine-tune vLLM parameters for full control over your LLM deployments.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint for efficient resource utilization and dynamic management.

## What's New

### Multi-Model Proxy Server (Experimental)
Serve multiple LLMs through a single unified API endpoint with live model management.

### Hardware-Optimized Profiles
Built-in profiles optimized for GPT-OSS models on NVIDIA GPUs:

*   `gpt_oss_ampere`
*   `gpt_oss_hopper`
*   `gpt_oss_blackwell`

### Shortcuts System
Save and quickly launch your favorite model + profile combinations.

### Full Ollama Integration
Automatic discovery of Ollama models, GGUF format support (experimental).

## Quick Start

### Installation

Choose one of the following installation methods:

#### Option 1: Install vLLM Separately (Recommended)
```bash
# Install vLLM -- Skip this step if you have vllm installed in your environment
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
# Or specify a backend: uv pip install vllm --torch-backend=cu128

# Install vLLM CLI
uv pip install --upgrade vllm-cli
uv run vllm-cli

# If you are using conda:
# Activate the environment you have vllm installed in
pip install vllm-cli
vllm-cli
```

#### Option 2: Install vLLM CLI + vLLM
```bash
pip install vllm-cli[vllm]
vllm-cli
```

#### Option 3: Build from Source (Requires vLLM Installation)
```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

#### Option 4: Isolated Installation (pipx/system packages)
```bash
# If you do not want to use virtual environment and want to install vLLM along with vLLM CLI
pipx install "vllm-cli[vllm]"

# If you want to install pre-release version
pipx install --pip-args="--pre" "vllm-cli[vllm]"
```

### Prerequisites

*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed (see installation instructions)

### Basic Usage

```bash
# Start Interactive mode
vllm-cli

# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

For detailed usage and troubleshooting, consult the [Usage Guide](docs/usage-guide.md).

## Configuration

### Built-in Profiles

Pre-configured profiles optimized for various use cases:

**General Purpose:**
*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

**Hardware-Specific (GPT-OSS):**
*   `gpt_oss_ampere`
*   `gpt_oss_hopper`
*   `gpt_oss_blackwell`

See the [Profiles Guide](docs/profiles.md) for detailed information.

### Configuration Files
*   **Main Config**: `~/.config/vllm-cli/config.yaml`
*   **User Profiles**: `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts**: `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [Usage Guide](docs/usage-guide.md)
*   [Multi-Model Proxy](docs/multi-model-proxy.md)
*   [Profiles Guide](docs/profiles.md)
*   [Troubleshooting](docs/troubleshooting.md)
*   [Screenshots](docs/screenshots.md)
*   [Model Discovery](docs/MODEL_DISCOVERY_QUICK_REF.md)
*   [Ollama Integration](docs/ollama-integration.md)
*   [Custom Models](docs/custom-model-serving.md)
*   [Roadmap](docs/roadmap.md)

## Integration with hf-model-tool

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery and management.

## Development

### Project Structure
```
src/vllm_cli/
├── cli/           # CLI command handling
├── config/        # Configuration management
├── models/        # Model management
├── server/        # Server lifecycle
├── ui/            # Terminal interface
└── schemas/       # JSON schemas
```

### Contributing

Contributions are welcome!  Please see the [GitHub repository](https://github.com/Chen-zexi/vllm-cli) for details on how to contribute.

## License

MIT License - see [LICENSE](LICENSE) file for details.
```
Key improvements and optimization techniques:

*   **Strong SEO Keywords:** Included phrases like "Large Language Models," "LLMs," "Command-Line Interface," "vLLM," "Model Management," and "Server Monitoring" throughout the text.
*   **Concise and Engaging Hook:**  The first sentence immediately grabs attention and highlights the core functionality.
*   **Clear Headings and Structure:**  Organized content with clear headings and subheadings for readability and scannability.
*   **Bulleted Key Features:**  Emphasized key selling points in an easily digestible bulleted list.
*   **Action-Oriented Language:** Used verbs like "serve," "manage," "discover," and "optimize" to encourage engagement.
*   **Quick Start Section:**  Focused on ease of use with straightforward installation and basic usage examples.
*   **Internal and External Linking:**  Incorporated links to relevant documentation and the original GitHub repository.
*   **Emphasis on Benefits:** Highlighted the advantages of using the tool (e.g., efficient resource utilization).
*   **Updated for Recent Changes:** The README reflects the latest features and improvements.
*   **Removed Redundancy:** Condensed information where possible to enhance clarity.
*   **Added more details:** Incorporated detailed explanations to highlight the functionality.