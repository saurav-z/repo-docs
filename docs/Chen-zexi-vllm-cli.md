# vLLM CLI: Supercharge Your LLM Serving with Ease

**Quickly deploy and manage Large Language Models (LLMs) with vLLM, featuring a user-friendly CLI and advanced server management.** ([Original Repository](https://github.com/Chen-zexi/vllm-cli))

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI is a powerful command-line tool designed to simplify the process of serving LLMs using vLLM, offering both interactive and command-line modes for efficient model management and server monitoring.

## Key Features:

*   **Interactive Mode:** Navigate a rich terminal interface with menu-driven options for easy server management.
*   **Command-Line Mode:** Automate tasks and integrate seamlessly into scripts with direct CLI commands.
*   **Model Management:** Easily discover and manage local models, including support for Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured and customizable profiles tailored to various use cases and hardware.
*   **Server Monitoring:** Monitor active vLLM servers in real-time for optimal performance.
*   **System Information:** Check GPU, memory, and CUDA compatibility to ensure smooth operation.
*   **Advanced Configuration:** Fine-tune vLLM parameters with comprehensive control and validation.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint for efficient resource utilization.

## What's New

### Multi-Model Proxy Server (Experimental)

*   **Single Endpoint:** Access all your models via one API.
*   **Live Management:** Dynamically add/remove models without service interruptions.
*   **Dynamic GPU Management:** Efficient GPU resource distribution using vLLM's sleep/wake.
*   **Interactive Setup:** A user-friendly wizard guides you through configuration.

See the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md) for detailed information.

### Hardware-Optimized Profiles for GPT-OSS Models (v0.2.4)

*   Built-in profiles optimized for GPT-OSS models on various NVIDIA GPU architectures, based on official vLLM GPT recipes:
    *   `gpt_oss_ampere` (NVIDIA A100)
    *   `gpt_oss_hopper` (NVIDIA H100/H200)
    *   `gpt_oss_blackwell` (NVIDIA Blackwell)

### Shortcuts System (v0.2.4)

*   Save and launch favorite model and profile combinations with shortcuts.

### Full Ollama Integration (v0.2.4)

*   Automatic Ollama model discovery and management.
*   GGUF format support (experimental).
*   System and user directory scanning.

## Quick Start

### Installation

**Option 1: Install vLLM Separately (Recommended)**

```bash
# Install vLLM (Skip if already installed in your environment)
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto  # Or specify a backend (cu128, etc.)

# Install vLLM CLI
uv pip install --upgrade vllm-cli
uv run vllm-cli
```

**Option 2: Install vLLM CLI + vLLM**

```bash
pip install vllm-cli[vllm]
vllm-cli
```

**Option 3: Build from Source**

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .  # Requires vLLM installed separately
```

**Option 4: For Isolated Installation (pipx)**

```bash
# Recommended: Use uv or conda for better CUDA compatibility
pipx install "vllm-cli[vllm]"
```

### Prerequisites

*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed (or installed through the CLI)
*   See [Troubleshooting Guide](docs/troubleshooting.md#dependency-conflicts) for dependency issues.

### Basic Usage

```bash
# Interactive mode
vllm-cli
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b
# Use a shortcut
vllm-cli serve --shortcut my-model
```

## Configuration

### Built-in Profiles

Choose from pre-configured profiles for optimized performance:

**General Purpose:**

*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`
*   `gpt_oss_hopper`
*   `gpt_oss_blackwell`

See the [📋 Profiles Guide](docs/profiles.md) for details.

### Configuration Files

*   **Main Config:** `~/.config/vllm-cli/config.yaml`
*   **User Profiles:** `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts:** `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [📘 Usage Guide](docs/usage-guide.md)
*   [🌐 Multi-Model Proxy](docs/multi-model-proxy.md)
*   [📋 Profiles Guide](docs/profiles.md)
*   [❓ Troubleshooting](docs/troubleshooting.md)
*   [📸 Screenshots](docs/screenshots.md)
*   [🔍 Model Discovery](docs/MODEL_DISCOVERY_QUICK_REF.md)
*   [🦙 Ollama Integration](docs/ollama-integration.md)
*   [⚙️ Custom Models](docs/custom-model-serving.md)
*   [🗺️ Roadmap](docs/roadmap.md)

## Integration with hf-model-tool

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for comprehensive model scanning and Ollama support.

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

Contributions are welcome!  Please submit issues or pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.