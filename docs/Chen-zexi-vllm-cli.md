# vLLM CLI: Supercharge Your LLM Serving with a Powerful Command-Line Interface

**Quickly and easily serve Large Language Models (LLMs) using vLLM with the intuitive vLLM CLI, offering interactive and command-line control, plus advanced features for model management and server optimization.** [(See Original Repo)](https://github.com/Chen-zexi/vllm-cli)

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

## Key Features:

*   **Interactive Terminal Interface:** Navigate menus, monitor GPU status, and manage your LLM servers with a user-friendly interface.
*   **Command-Line Automation:** Automate tasks and integrate with scripts using direct CLI commands.
*   **Effortless Model Management:** Discover and manage local models, including HuggingFace and Ollama support.
*   **Customizable Configuration Profiles:** Define and save profiles for different use cases, including optimized hardware-specific profiles for GPT-OSS models.
*   **Real-time Server Monitoring:** Keep tabs on your vLLM server performance with built-in monitoring tools.
*   **System Information & Compatibility Checks:** Verify GPU, memory, and CUDA compatibility to ensure optimal performance.
*   **Advanced Configuration Options:** Fine-tune vLLM parameters with comprehensive control and validation.
*   **Experimental Multi-Model Proxy:** (New!) Serve multiple models through a single unified API endpoint.

## What's New

### Multi-Model Proxy Server (Experimental)

The Multi-Model Proxy allows you to serve multiple LLMs through a single unified API endpoint. This new feature is under active development and available for testing.

**What It Does:**
*   **Single Endpoint** - All your models accessible through one API
*   **Live Management** - Add or remove models without stopping the service
*   **Dynamic GPU Management** - Efficient GPU resource distribution through vLLM's sleep/wake functionality
*   **Interactive Setup** - User-friendly wizard guides you through configuration

**Note:** This is an experimental feature under active development. Please share your experience through [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).

For complete documentation, see the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Hardware-Optimized Profiles for GPT-OSS Models

New built-in profiles specifically optimized for serving GPT-OSS models on different GPU architectures:
*   `gpt_oss_ampere` - Optimized for NVIDIA A100 GPUs
*   `gpt_oss_hopper` - Optimized for NVIDIA H100/H200 GPUs
*   `gpt_oss_blackwell` - Optimized for NVIDIA Blackwell GPUs

Based on official [vLLM GPT recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) for maximum performance.

## Quick Start

### Installation:

**Important: vLLM Installation Notes** ⚠️ **Binary Compatibility Warning**: vLLM contains pre-compiled CUDA kernels that must match your PyTorch version exactly. Installing mismatched versions will cause errors.

vLLM-CLI will not install vLLM or Pytorch by default.

#### Option 1: Install vLLM seperately and then install vLLM CLI (Recommended)

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

#### Option 3: Build from source (You still need to install vLLM seperately)
```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

#### Option 4: For Isolated Installation (pipx/system packages)

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
*   vLLM package installed (separately or via the install options above)

### Basic Usage:

```bash
# Interactive mode - menu-driven interface
vllm-cli
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

For detailed usage instructions, see the [📘 Usage Guide](docs/usage-guide.md).

## Configuration

### Built-in Profiles:

vLLM CLI includes pre-configured profiles for optimal performance:

**General Purpose:**

*   `standard` - Minimal configuration with smart defaults
*   `high_throughput` - Maximum performance configuration
*   `low_memory` - Memory-constrained environments
*   `moe_optimized` - Optimized for Mixture of Experts models

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere` - NVIDIA A100 GPUs
*   `gpt_oss_hopper` - NVIDIA H100/H200 GPUs
*   `gpt_oss_blackwell` - NVIDIA Blackwell GPUs

See [**📋 Profiles Guide**](docs/profiles.md) for detailed information.

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

vLLM CLI uses [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery:
* Comprehensive model scanning
* Ollama model support
* Shared configuration

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