# vLLM CLI: Command-Line Interface for Serving Large Language Models

**Effortlessly deploy and manage your LLMs with vLLM CLI, a powerful command-line tool for efficient model serving.**  [View the original repo](https://github.com/Chen-zexi/vllm-cli)

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI provides a user-friendly interface for serving Large Language Models using vLLM, offering both interactive and command-line modes, configuration profiles, model management, and server monitoring. This tool simplifies the process of deploying and managing LLMs, making it accessible to developers of all levels.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Key Features

*   **Interactive Mode:** Navigate and manage your LLMs with a rich, menu-driven terminal interface.
*   **Command-Line Mode:** Automate and script your LLM operations with direct CLI commands.
*   **Model Management:** Automatically discover and manage local models, including Hugging Face and Ollama models.
*   **Configuration Profiles:** Utilize pre-configured and customizable server profiles tailored for various use cases.
*   **Server Monitoring:** Monitor active vLLM servers in real-time for optimal performance.
*   **System Information:** Check GPU, memory, and CUDA compatibility to ensure smooth operation.
*   **Advanced Configuration:** Gain full control over vLLM parameters with comprehensive validation.
*   **Hardware-Optimized Profiles:** Built-in profiles for GPT-OSS models, optimized for different NVIDIA GPUs.
*   **Shortcuts System:** Save and quickly launch frequently used model and profile combinations.
*   **Ollama Integration:**  Seamlessly integrate with Ollama models, supporting GGUF format (experimental).

**Quick Links:** [Documentation](#documentation) | [Quick Start](#quick-start) | [Screenshots](docs/screenshots.md) | [Usage Guide](docs/usage-guide.md) | [Troubleshooting](docs/troubleshooting.md) | [Roadmap](docs/roadmap.md)

## What's New

### Hardware-Optimized Profiles for GPT-OSS Models

New built-in profiles specifically optimized for serving GPT-OSS models on different GPU architectures:
- **`gpt_oss_ampere`** - Optimized for NVIDIA A100 GPUs
- **`gpt_oss_hopper`** - Optimized for NVIDIA H100/H200 GPUs
- **`gpt_oss_blackwell`** - Optimized for NVIDIA Blackwell GPUs

### Shortcuts System

Save and quickly launch your favorite model + profile combinations:
```bash
vllm-cli serve --shortcut my-gpt-server
```

### Full Ollama Integration

- Automatic discovery of Ollama models
- GGUF format support (experimental)
- System and user directory scanning

### Enhanced Configuration

-   **Environment Variables**: Universal and profile-specific environment variable management
-   **GPU Selection**: Choose specific GPUs for model serving (`--device 0,1`)
-   **Enhanced System Info**: vLLM feature detection with attention backend availability

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Pre-Release Features (v0.2.5)

### Multi-Model Proxy Server (Experimental)

The Multi-Model Proxy is a new experimental feature that enables serving multiple LLMs through a single unified API endpoint. This feature is currently under active development and available for testing.

**What It Does:**

*   **Single Endpoint:** Access all your models through a single API.
*   **Live Management:** Add or remove models without service interruption.
*   **Dynamic GPU Management:** Efficient GPU resource distribution with vLLM's sleep/wake functionality.
*   **Interactive Setup:** User-friendly wizard for easy configuration.

**Install the pre-release version:**

```bash
pip install --pre --upgrade vllm-cli
```

**Note:** This is an experimental feature. Provide feedback via [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).

For complete documentation, see the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Quick Start

### Installation

#### Option 1: Install vLLM Separately (Recommended)
```bash
# Install vLLM
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm-cli[vllm] --torch-backend=auto
# Or specify a backend: uv pip install vllm-cli[vllm] --torch-backend=cu129

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

#### Option 4: Isolated Installation (pipx/system packages)

```bash
pipx install "vllm-cli[vllm]"
```

### Prerequisites

*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed (see installation instructions above)

### Basic Usage

```bash
# Interactive mode
vllm-cl
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b
# Use a shortcut
vllm-cli serve --shortcut my-model
```

For detailed instructions, see the [Usage Guide](docs/usage-guide.md) and [Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI offers optimized profiles:

**General Purpose:**

*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`
*   `gpt_oss_hopper`
*   `gpt_oss_blackwell`

See [Profiles Guide](docs/profiles.md) for details.

### Configuration Files

*   **Main Config:** `~/.config/vllm-cli/config.yaml`
*   **User Profiles:** `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts:** `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [Usage Guide](docs/usage-guide.md) - Complete usage instructions
*   [Multi-Model Proxy](docs/multi-model-proxy.md) - Serve multiple models simultaneously
*   [Profiles Guide](docs/profiles.md) - Built-in profiles details
*   [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions
*   [Screenshots](docs/screenshots.md) - Visual feature overview
*   [Model Discovery](docs/MODEL_DISCOVERY_QUICK_REF.md) - Model management guide
*   [Ollama Integration](docs/ollama-integration.md) - Using Ollama models
*   [Custom Models](docs/custom-model-serving.md) - Serving custom models
*   [Roadmap](docs/roadmap.md) - Future development plans

## Integration with hf-model-tool

vLLM CLI utilizes [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery:

*   Comprehensive model scanning
*   Ollama model support
*   Shared configuration

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

## License

MIT License - see [LICENSE](LICENSE) for details.