# vLLM CLI: The Command-Line Tool to Supercharge Your LLM Serving 🚀

**Easily serve and manage Large Language Models with vLLM using an interactive terminal interface and powerful command-line tools.**  Check out the [original repository](https://github.com/Chen-zexi/vllm-cli) for more information.

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI simplifies serving Large Language Models (LLMs) with vLLM, offering both interactive and command-line modes. It streamlines model management, server monitoring, and configuration, allowing you to get your LLMs up and running quickly and efficiently.

**Key Features:**

*   **Interactive Mode:** Navigate a rich terminal interface with menus and real-time GPU and system information.
*   **Command-Line Mode:** Automate tasks and script your LLM deployments with direct CLI commands.
*   **Model Management:** Seamlessly discover and manage local models from Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles or customize server settings to suit your needs.
*   **Server Monitoring:** Monitor active vLLM servers in real time.
*   **System Information:** Quickly check GPU, memory, and CUDA compatibility.
*   **Advanced Configuration:** Fine-tune vLLM parameters for optimal performance.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint with dynamic GPU management.

## What's New

### Multi-Model Proxy Server (Experimental)

*   **Single Endpoint:** Access multiple models through one API.
*   **Live Management:** Add or remove models without service interruption.
*   **Dynamic GPU Management:** Efficiently allocate GPU resources using vLLM's sleep/wake functionality.
*   **Interactive Setup:** Get guided through setup with a user-friendly wizard.

### Hardware-Optimized Profiles for GPT-OSS Models

*   Pre-configured profiles for NVIDIA A100, H100/H200, and Blackwell GPUs.

### Shortcuts System

*   Save and quickly launch your favorite model and profile combinations.

### Full Ollama Integration

*   Automatic Ollama model discovery and support.
*   GGUF format support (experimental).

### Enhanced Configuration

*   Environment variable management.
*   GPU selection and more.

For more details, see the [CHANGELOG.md](CHANGELOG.md).

## Quick Start

### Important: vLLM Installation Notes

vLLM-CLI **does not** install vLLM or PyTorch by default. You must install them separately.

### Installation

#### Option 1: Install vLLM Separately (Recommended)

```bash
# Install vLLM (if you haven't already)
# uv venv --python 3.12 --seed
# source .venv/bin/activate
# uv pip install vllm --torch-backend=auto

# Install vLLM CLI
uv pip install --upgrade vllm-cli
uv run vllm-cli
```

#### Option 2: Install vLLM CLI + vLLM

```bash
pip install vllm-cli[vllm]
vllm-cli
```

#### Option 3: Build from Source

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

#### Option 4: For Isolated Installation (pipx/system packages)

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

vLLM CLI provides several optimized profiles:

**General Purpose:** `standard`, `high_throughput`, `low_memory`, `moe_optimized`
**Hardware-Specific (GPT-OSS):** `gpt_oss_ampere`, `gpt_oss_hopper`, `gpt_oss_blackwell`

See [**📋 Profiles Guide**](docs/profiles.md) for more info.

### Configuration Files

*   **Main Config**: `~/.config/vllm-cli/config.yaml`
*   **User Profiles**: `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts**: `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [**📘 Usage Guide**](docs/usage-guide.md)
*   [**🌐 Multi-Model Proxy**](docs/multi-model-proxy.md)
*   [**📋 Profiles Guide**](docs/profiles.md)
*   [**❓ Troubleshooting**](docs/troubleshooting.md)
*   [**📸 Screenshots**](docs/screenshots.md)
*   [**🔍 Model Discovery**](docs/MODEL_DISCOVERY_QUICK_REF.md)
*   [**🦙 Ollama Integration**](docs/ollama-integration.md)
*   [**⚙️ Custom Models**](docs/custom-model-serving.md)
*   [**🗺️ Roadmap**](docs/roadmap.md)

## Integration with hf-model-tool

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for comprehensive model management.

## Development

### Project Structure

```
src/vllm_cli/
├── cli/
├── config/
├── models/
├── server/
├── ui/
└── schemas/
```

### Contributing

Contributions are welcome!