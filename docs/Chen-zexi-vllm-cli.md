# vLLM CLI: Supercharge Your Large Language Model Serving with Ease

**Quickly deploy and manage your Large Language Models (LLMs) with `vllm-cli`, a user-friendly command-line interface built on the powerful vLLM library.**  ([Original Repo](https://github.com/Chen-zexi/vllm-cli))

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

## Key Features

*   **Interactive Mode:**  Navigate and manage your LLMs with a rich, menu-driven terminal interface.
*   **Command-Line Mode:** Automate tasks and integrate with scripts using direct CLI commands.
*   **Model Management:** Seamlessly discover and load models from Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured or custom server profiles tailored for different use cases.
*   **Server Monitoring:** Monitor active vLLM servers in real-time for optimal performance.
*   **System Information:**  Check GPU, memory, and CUDA compatibility to ensure smooth operation.
*   **Advanced Configuration:** Fine-tune vLLM parameters with validation for maximum control.
*   **Multi-Model Proxy (Experimental):** Serve multiple models through a single API endpoint for efficient resource utilization (under active development).
*   **Hardware-Optimized Profiles:**  Built-in profiles for optimal performance with GPT-OSS models on various NVIDIA GPUs.
*   **Shortcuts System:**  Save and quickly launch your preferred model and profile combinations.

## What's New

### Multi-Model Proxy Server (Experimental)

*   **Single Endpoint:** Access all your models through a unified API.
*   **Live Management:** Add or remove models without stopping the service.
*   **Dynamic GPU Management:** Efficient resource allocation using vLLM's sleep/wake functionality.
*   **Interactive Setup:** User-friendly wizard to guide you through the configuration.

See the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md) for more details and contribute your feedback on [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).

### Recent Updates

*   **Hardware-Optimized Profiles for GPT-OSS Models**: Pre-configured profiles for NVIDIA A100, H100/H200, and Blackwell GPUs based on official vLLM GPT recipes.
*   **Shortcuts System**: Easily save and launch your favorite model and profile combinations.
*   **Full Ollama Integration**: Including GGUF format support (experimental) and system/user directory scanning.
*   **Enhanced Configuration**: Environment variables, GPU selection, and enhanced system information.

Explore the [CHANGELOG.md](CHANGELOG.md) for a complete list of changes.

## Quick Start

### Installation

#### Option 1: Install vLLM separately and then install vLLM CLI (Recommended)

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

#### Option 3: Build from source (You still need to install vLLM separately)

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
# Interactive mode
vllm-cli
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

Refer to the [📘 Usage Guide](docs/usage-guide.md) and [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md) for comprehensive instructions.

## Configuration

### Built-in Profiles

`vllm-cli` includes several pre-configured profiles:

**General Purpose:**

*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`
*   `gpt_oss_hopper`
*   `gpt_oss_blackwell`

See the [**📋 Profiles Guide**](docs/profiles.md) for profile details.

### Configuration Files

*   **Main Config:** `~/.config/vllm-cli/config.yaml`
*   **User Profiles:** `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts:** `~/.config/vllm-cli/shortcuts.json`

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

`vllm-cli` leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for efficient model discovery and configuration.

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

Contributions are welcome!  Please open an issue or submit a pull request.