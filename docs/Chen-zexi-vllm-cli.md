# vLLM CLI: Command-Line Interface for Serving Large Language Models

**Effortlessly deploy and manage your LLMs with vLLM-CLI, a powerful and user-friendly command-line tool.**

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI provides a streamlined way to serve your Large Language Models using the high-performance vLLM library. Offering both interactive and command-line modes, it simplifies model management, configuration, and server monitoring.

[**Explore the vLLM CLI Repo**](https://github.com/Chen-zexi/vllm-cli)

## Key Features

*   **🚀 Interactive Mode:** Intuitive terminal interface with menu-driven navigation for easy model management and server control.
*   **⚡ Command-Line Mode:** Direct CLI commands for automation, scripting, and seamless integration into your workflows.
*   **🤖 Model Management:** Automatic discovery of local models from Hugging Face and Ollama, simplifying model selection.
*   **🔧 Configuration Profiles:** Pre-configured and custom server profiles tailored to various use cases and hardware configurations.
*   **📊 Server Monitoring:** Real-time monitoring of active vLLM servers, providing insights into performance and resource usage.
*   **🖥️ System Information:** Comprehensive system checks for GPU, memory, and CUDA compatibility.
*   **📝 Advanced Configuration:** Fine-grained control over vLLM parameters with validation.
*   **🌐 Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint, enabling efficient GPU resource management.

## What's New

### Multi-Model Proxy Server (Experimental)

The Multi-Model Proxy allows you to serve multiple LLMs through a unified API endpoint, with dynamic GPU management and a user-friendly setup wizard.

*   **Single Endpoint:** Access all your models through a single API.
*   **Live Management:** Add or remove models without restarting the service.
*   **Dynamic GPU Management:** Efficiently distribute GPU resources using vLLM's sleep/wake functionality.
*   **Interactive Setup:** Guides you through the configuration process.

**Note:** This is an experimental feature. Please share your feedback through [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).

For complete documentation, see the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

### Hardware-Optimized Profiles for GPT-OSS Models

New built-in profiles specifically optimized for serving GPT-OSS models:
- `gpt_oss_ampere` - Optimized for NVIDIA A100 GPUs
- `gpt_oss_hopper` - Optimized for NVIDIA H100/H200 GPUs
- `gpt_oss_blackwell` - Optimized for NVIDIA Blackwell GPUs

### ⚡ Shortcuts System
Save and quickly launch your favorite model + profile combinations:
```bash
vllm-cli serve --shortcut my-gpt-server
```

### 🦙 Full Ollama Integration
- Automatic discovery of Ollama models
- GGUF format support (experimental)
- System and user directory scanning

### 🔧 Enhanced Configuration
- **Environment Variables** - Universal and profile-specific environment variable management
- **GPU Selection** - Choose specific GPUs for model serving (`--device 0,1`)
- **Enhanced System Info** - vLLM feature detection with attention backend availability

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Quick Start

### Important: vLLM Installation Notes

⚠️ **Binary Compatibility Warning**: vLLM contains pre-compiled CUDA kernels that must match your PyTorch version exactly. Installing mismatched versions will cause errors.

vLLM-CLI will not install vLLM or Pytorch by default.

### Installation

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

For detailed usage instructions, see the [📘 Usage Guide](docs/usage-guide.md) and [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI includes 7 optimized profiles for different use cases:

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

vLLM CLI utilizes [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for robust model discovery, including Ollama model support and shared configuration.

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