# vLLM CLI: Command-Line Interface for Serving Large Language Models

**Effortlessly deploy and manage Large Language Models with vLLM using an intuitive command-line interface and advanced features.**  [Explore the original repository on GitHub](https://github.com/Chen-zexi/vllm-cli).

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI is a powerful command-line tool designed to simplify the deployment and management of Large Language Models (LLMs) using vLLM. It provides a user-friendly interface for both interactive and command-line interactions, with a focus on ease of use, performance, and comprehensive features.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Key Features

*   **Interactive Mode:** Rich, menu-driven terminal interface for easy navigation and management.
*   **Command-Line Mode:** Direct CLI commands for automation, scripting, and integration.
*   **Model Management:** Seamless discovery and management of local models, with Hugging Face and Ollama support.
*   **Configuration Profiles:** Pre-configured and customizable server profiles for diverse use cases.
*   **Server Monitoring:** Real-time monitoring of active vLLM servers, providing valuable insights.
*   **System Information:** Comprehensive GPU, memory, and CUDA compatibility checks.
*   **Advanced Configuration:** Fine-grained control over vLLM parameters with built-in validation.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint.

**Quick Links:** [Documentation](#documentation) | [Quick Start](#quick-start) | [Screenshots](docs/screenshots.md) | [Usage Guide](docs/usage-guide.md) | [Troubleshooting](docs/troubleshooting.md) | [Roadmap](docs/roadmap.md)

## What's New

### Multi-Model Proxy Server (Experimental)

The Multi-Model Proxy is an experimental feature that enables serving multiple LLMs through a single, unified API endpoint. This allows for efficient resource utilization and streamlined model serving.

**Key Benefits:**

*   **Single API Endpoint:** Access all your models through one convenient API.
*   **Dynamic Management:** Add or remove models without interrupting service.
*   **Efficient GPU Management:** Leverage vLLM's sleep/wake functionality for optimal resource allocation.
*   **User-Friendly Setup:** Interactive wizard guides you through configuration.

**Note:** This feature is under active development. Your feedback is invaluable – please share your experiences via [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).

### Hardware-Optimized Profiles for GPT-OSS Models (v0.2.4)

New, built-in profiles optimized for serving GPT-OSS models on different GPU architectures:

*   `gpt_oss_ampere` - Optimized for NVIDIA A100 GPUs
*   `gpt_oss_hopper` - Optimized for NVIDIA H100/H200 GPUs
*   `gpt_oss_blackwell` - Optimized for NVIDIA Blackwell GPUs

These profiles are based on the official [vLLM GPT recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) for optimal performance.

### Shortcuts System (v0.2.4)

Save and quickly launch your favorite model + profile combinations with shortcuts:

```bash
vllm-cli serve --shortcut my-gpt-server
```

### Full Ollama Integration (v0.2.4)

*   Automatic discovery of Ollama models
*   GGUF format support (experimental)
*   System and user directory scanning

### Enhanced Configuration (v0.2.4)

*   **Environment Variables:** Universal and profile-specific environment variable management.
*   **GPU Selection:** Choose specific GPUs for model serving (`--device 0,1`).
*   **Enhanced System Info:** vLLM feature detection with attention backend availability.

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Quick Start

### Important: vLLM Installation Notes

⚠️ **Binary Compatibility Warning**:  vLLM contains pre-compiled CUDA kernels that *must* match your PyTorch version exactly.  Ensure you install compatible versions to avoid errors.

vLLM-CLI does not install vLLM or PyTorch by default.

### Installation

#### Option 1: Install vLLM Separately and then Install vLLM CLI (Recommended)

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

#### Option 3: Build from Source (You still need to install vLLM separately)

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

#### Option 4: For Isolated Installation (pipx/system packages)

⚠️ **Compatibility Note:** pipx creates isolated environments, which may have compatibility issues with vLLM's CUDA dependencies.  Consider using uv or conda (see above) for better PyTorch/CUDA compatibility.

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

For detailed instructions, consult the [Usage Guide](docs/usage-guide.md) and [Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI includes 7 optimized profiles, tailored for various use cases:

**General Purpose:**

*   `standard`: Minimal configuration with intelligent defaults.
*   `high_throughput`: Designed for maximum performance.
*   `low_memory`: Optimized for memory-constrained environments.
*   `moe_optimized`: Optimized for Mixture of Experts models.

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`: For NVIDIA A100 GPUs.
*   `gpt_oss_hopper`: For NVIDIA H100/H200 GPUs.
*   `gpt_oss_blackwell`: For NVIDIA Blackwell GPUs.

See the [Profiles Guide](docs/profiles.md) for detailed information.

### Configuration Files

*   **Main Config:** `~/.config/vllm-cli/config.yaml`
*   **User Profiles:** `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts:** `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [**Usage Guide**](docs/usage-guide.md) - Comprehensive usage instructions.
*   [**Multi-Model Proxy**](docs/multi-model-proxy.md) -  Serve multiple models simultaneously.
*   [**Profiles Guide**](docs/profiles.md) -  Details on built-in profiles.
*   [**Troubleshooting**](docs/troubleshooting.md) - Solutions to common issues.
*   [**Screenshots**](docs/screenshots.md) - Visual feature overview.
*   [**Model Discovery**](docs/MODEL_DISCOVERY_QUICK_REF.md) - Model management guide.
*   [**Ollama Integration**](docs/ollama-integration.md) - Using Ollama models.
*   [**Custom Models**](docs/custom-model-serving.md) - Serving custom models.
*   [**Roadmap**](docs/roadmap.md) - Future development plans.

## Integration with hf-model-tool

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery, offering:

*   Comprehensive model scanning.
*   Ollama model support.
*   Shared configuration.

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

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) file for details.