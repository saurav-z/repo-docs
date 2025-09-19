# vLLM CLI: Command-Line Interface for Serving Large Language Models 🚀

**Effortlessly deploy and manage your LLMs with vLLM CLI, offering a streamlined interface and advanced features for optimal performance.**  [Check out the original repository!](https://github.com/Chen-zexi/vllm-cli)

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI provides a powerful and user-friendly command-line interface for serving and managing Large Language Models (LLMs) using the efficient vLLM library. It offers both interactive and command-line modes, along with features for model management, configuration profiles, and real-time server monitoring.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Key Features

*   **Interactive Mode:** Navigate and control your LLMs with a rich, menu-driven terminal interface.
*   **Command-Line Mode:** Automate tasks and integrate with scripts using direct CLI commands.
*   **Model Management:**  Effortlessly discover and manage local models with support for Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles or create custom profiles for optimized performance.
*   **Server Monitoring:** Real-time monitoring of active vLLM servers to track performance and resource usage.
*   **System Information:**  Provides critical GPU, memory, and CUDA compatibility checks.
*   **Advanced Configuration:** Fine-tune vLLM parameters with comprehensive control and validation.
*   **Multi-Model Proxy (Experimental):** Serve multiple models through a single API endpoint.

## What's New

### Multi-Model Proxy Server (Experimental)

This experimental feature allows you to serve multiple LLMs through a single unified API endpoint, offering benefits like:

*   **Single Endpoint:** Access all models via one API.
*   **Live Management:** Add or remove models without server downtime.
*   **Dynamic GPU Management:** Efficient GPU resource allocation with vLLM's sleep/wake functionality.
*   **Interactive Setup:** User-friendly wizard for easy configuration.

**Note:** This is an experimental feature, your feedback is welcome!

For complete documentation, see the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

### Hardware-Optimized Profiles for GPT-OSS Models & More

Optimized profiles specifically tailored for GPT-OSS models on different GPU architectures, and full Ollama integration.

*   **`gpt_oss_ampere`**: NVIDIA A100 GPUs
*   **`gpt_oss_hopper`**: NVIDIA H100/H200 GPUs
*   **`gpt_oss_blackwell`**: NVIDIA Blackwell GPUs
*   **Shortcuts System:** Save and quickly launch your favorite model + profile combinations:
   ```bash
   vllm-cli serve --shortcut my-gpt-server
   ```
*   **🦙 Full Ollama Integration**
*   **Environment Variables** - Universal and profile-specific environment variable management

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

vLLM CLI provides 7 optimized profiles, including:

**General Purpose:**

*   `standard`: Minimal configuration with smart defaults.
*   `high_throughput`: Configuration for maximum performance.
*   `low_memory`: Optimized for memory-constrained environments.
*   `moe_optimized`: Optimized for Mixture of Experts models.

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`: NVIDIA A100 GPUs
*   `gpt_oss_hopper`: NVIDIA H100/H200 GPUs
*   `gpt_oss_blackwell`: NVIDIA Blackwell GPUs

See [**📋 Profiles Guide**](docs/profiles.md) for detailed information.

### Configuration Files

*   **Main Config**: `~/.config/vllm-cli/config.yaml`
*   **User Profiles**: `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts**: `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [**📘 Usage Guide**](docs/usage-guide.md): Comprehensive usage instructions
*   [**🌐 Multi-Model Proxy**](docs/multi-model-proxy.md): Guide to serving multiple models
*   [**📋 Profiles Guide**](docs/profiles.md): Details on built-in profiles
*   [**❓ Troubleshooting**](docs/troubleshooting.md): Solutions to common issues
*   [**📸 Screenshots**](docs/screenshots.md): Visual feature overview
*   [**🔍 Model Discovery**](docs/MODEL_DISCOVERY_QUICK_REF.md): Guide to model management
*   [**🦙 Ollama Integration**](docs/ollama-integration.md): Using Ollama models
*   [**⚙️ Custom Models**](docs/custom-model-serving.md): Serving custom models
*   [**🗺️ Roadmap**](docs/roadmap.md): Future development plans

## Integration with hf-model-tool

vLLM CLI integrates with [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery:
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