# vLLM CLI: Your Command-Line Interface for Powerful LLM Serving

**Easily serve and manage Large Language Models (LLMs) with vLLM, offering an interactive terminal interface and robust command-line features.** ([Original Repository](https://github.com/Chen-zexi/vllm-cli))

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI provides a streamlined experience for serving LLMs, offering both interactive and command-line modes, configuration profiles, model management, and server monitoring.

**Key Features:**

*   **Interactive Mode:** A rich, menu-driven terminal interface for easy navigation and control.
*   **Command-Line Mode:** Automate tasks and integrate with scripts using direct CLI commands.
*   **Model Management:** Seamlessly discover and manage local models from Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles or customize your own server settings for different use cases.
*   **Server Monitoring:** Real-time monitoring of active vLLM server status and resource usage.
*   **System Information:** Comprehensive system checks including GPU, memory, and CUDA compatibility.
*   **Advanced Configuration:** Fine-tune vLLM parameters with validation for optimal performance.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint (under active development).
*   **Hardware-Optimized Profiles:**  Built-in profiles optimized for specific GPU architectures (NVIDIA A100, H100/H200, Blackwell).
*   **Shortcuts System:** Save and launch favorite model and profile combinations with shortcuts.
*   **Full Ollama Integration:** Automatic Ollama model discovery and GGUF format support (experimental).

**Quick Links:** [📖 Documentation](#documentation) | [🚀 Quick Start](#quick-start) | [📸 Screenshots](docs/screenshots.md) | [📘 Usage Guide](docs/usage-guide.md)

## What's New

### Multi-Model Proxy Server (Experimental)

Serve multiple models through a single API endpoint.  Features live model management and dynamic GPU allocation.

**Note:**  This feature is experimental.  Provide feedback via [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).  See the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

### Hardware-Optimized Profiles

*   Optimized profiles for GPT-OSS models on NVIDIA GPUs (Ampere, Hopper, Blackwell).
*   Shortcuts System
*   Full Ollama integration

## Quick Start

### Important: vLLM Installation Notes

⚠️ **Binary Compatibility Warning**: Ensure your vLLM installation matches your PyTorch version exactly to avoid errors.

vLLM-CLI will not install vLLM or Pytorch by default.

### Installation

#### Option 1: Install vLLM separately, then install vLLM CLI (Recommended)

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
# Install vLLM CLI + vLLM
pip install vllm-cli[vllm]
vllm-cli
```

#### Option 3: Build from source (requires separate vLLM install)

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

#### Option 4: Isolated Installation (pipx/system packages)

⚠️ **Compatibility Note:** pipx creates isolated environments which may have compatibility issues with vLLM's CUDA dependencies. Consider using uv or conda (see above) for better PyTorch/CUDA compatibility.

```bash
# If you do not want to use virtual environment and want to install vLLM along with vLLM CLI
pipx install "vllm-cli[vllm]"

# If you want to install pre-release version
pipx install --pip-args="--pre" "vllm-cli[vllm]"
```

### Prerequisites

*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed
*   For dependency issues, see [Troubleshooting Guide](docs/troubleshooting.md#dependency-conflicts)

### Basic Usage

```bash
# Interactive mode - menu-driven interface
vllm-cl
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

## Configuration

### Built-in Profiles

Pre-configured profiles for various use cases:

**General Purpose:**

*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`
*   `gpt_oss_hopper`
*   `gpt_oss_blackwell`

See the [**📋 Profiles Guide**](docs/profiles.md) for details.

### Configuration Files

*   **Main Config**: `~/.config/vllm-cli/config.yaml`
*   **User Profiles**: `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts**: `~/.config/vllm-cli/shortcuts.json`

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

vLLM CLI utilizes [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery, providing comprehensive scanning and Ollama support.

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

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) file for details.