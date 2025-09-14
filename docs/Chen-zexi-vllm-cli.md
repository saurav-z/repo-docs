# vLLM CLI: The Ultimate Command-Line Interface for Serving Large Language Models

**Supercharge your LLM serving with vLLM CLI, a powerful and user-friendly command-line tool, providing interactive and automated control for efficient model deployment.** ([See the original repository](https://github.com/Chen-zexi/vllm-cli))

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI simplifies the process of serving and managing Large Language Models (LLMs) using the high-performance vLLM framework. It offers both interactive terminal interfaces and command-line options, along with robust features like configuration profiles, model management, and server monitoring.

## Key Features

*   **Interactive Mode:** Navigate and manage your LLM servers with an intuitive, menu-driven terminal interface.
*   **Command-Line Mode:** Automate and script your LLM deployments with direct CLI commands.
*   **Model Management:** Seamlessly discover and manage local LLMs from Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles or customize server settings for various use cases.
*   **Server Monitoring:** Real-time monitoring of your active vLLM servers, including GPU utilization.
*   **System Information:** Check GPU, memory, and CUDA compatibility to ensure optimal performance.
*   **Advanced Configuration:** Fine-tune vLLM parameters with extensive control and validation.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint.

## What's New

*   **Multi-Model Proxy Server (Experimental):** A new feature enabling serving multiple LLMs through a single API endpoint for easier management and efficient GPU resource allocation. See the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md) for more information.
*   **Hardware-Optimized Profiles:** Built-in profiles optimized for GPT-OSS models on various NVIDIA GPU architectures (Ampere, Hopper, Blackwell).
*   **Shortcuts System:** Quickly launch your favorite model and profile combinations with convenient shortcuts.
*   **Full Ollama Integration:** Includes automatic Ollama model discovery and experimental GGUF format support.
*   **Enhanced Configuration:** Environment variables and enhanced GPU selection features.

## Quick Start

### Installation

Choose an installation method based on your needs:

#### Option 1: Recommended: Install vLLM Separately and then vLLM CLI

```bash
# Install vLLM (if you don't have it already)
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

#### Option 3: Build from Source (Requires vLLM installation)

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
*   CUDA-compatible GPU (Recommended)
*   vLLM package installed (See installation instructions above)

### Basic Usage

```bash
# Interactive mode
vllm-cli

# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

For more detailed instructions, see the [📘 Usage Guide](docs/usage-guide.md).

## Configuration

### Built-in Profiles

vLLM CLI comes with several optimized profiles for different use cases:

**General Purpose:**

*   `standard`: Default profile with smart defaults.
*   `high_throughput`: Maximizes performance.
*   `low_memory`: Suitable for memory-constrained environments.
*   `moe_optimized`: Optimized for Mixture of Experts models.

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`: Optimized for NVIDIA A100 GPUs.
*   `gpt_oss_hopper`: Optimized for NVIDIA H100/H200 GPUs.
*   `gpt_oss_blackwell`: Optimized for NVIDIA Blackwell GPUs.

See the [**📋 Profiles Guide**](docs/profiles.md) for a complete list.

### Configuration Files

*   **Main Config:** `~/.config/vllm-cli/config.yaml`
*   **User Profiles:** `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts:** `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [**📘 Usage Guide**](docs/usage-guide.md) - Complete usage instructions
*   [**🌐 Multi-Model Proxy**](docs/multi-model-proxy.md) - Serve multiple models simultaneously
*   [**📋 Profiles Guide**](docs/profiles.md) - Built-in profiles details
*   [**❓ Troubleshooting**](docs/troubleshooting.md) - Common issues and solutions
*   [**📸 Screenshots**](docs/screenshots.md) - Visual feature overview
*   [**🔍 Model Discovery**](docs/MODEL_DISCOVERY_QUICK_REF.md) - Model management guide
*   [**🦙 Ollama Integration**](docs/ollama-integration.md) - Using Ollama models
*   [**⚙️ Custom Models**](docs/custom-model-serving.md) - Serving custom models
*   [**🗺️ Roadmap**](docs/roadmap.md) - Future development plans

## Integration with hf-model-tool

vLLM CLI integrates with [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for robust model discovery and management, including Ollama support.

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

Contributions are welcomed! Feel free to open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) file for details.