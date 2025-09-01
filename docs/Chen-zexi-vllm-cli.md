# vLLM CLI: Effortlessly Serve Large Language Models (LLMs)

**Supercharge your LLM serving with vLLM CLI, a powerful command-line tool designed for easy deployment, management, and monitoring of your models.** ([Original Repo](https://github.com/Chen-zexi/vllm-cli))

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI simplifies the process of serving LLMs, offering both interactive and command-line modes for flexibility. It provides robust features like model management, server monitoring, and advanced configuration options to optimize your LLM deployment.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*

## Key Features

*   **Interactive Mode:** Navigate with a rich terminal interface, providing a user-friendly experience.
*   **Command-Line Mode:** Automate tasks and integrate with scripts using direct CLI commands.
*   **Model Management:**  Seamlessly discover and manage local models with Hugging Face and Ollama support.
*   **Configuration Profiles:**  Utilize pre-configured or customize server profiles to fit your specific needs.
*   **Server Monitoring:** Monitor your vLLM servers in real-time for optimal performance.
*   **System Information:**  Check GPU, memory, and CUDA compatibility to ensure smooth operation.
*   **Advanced Configuration:** Fine-tune vLLM parameters with complete control and validation.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint for efficient resource utilization.
*   **Hardware Optimized Profiles**: Pre-built profiles for NVIDIA A100, H100/H200, and Blackwell GPUs.

## What's New

### Multi-Model Proxy Server (Experimental)

This feature allows serving multiple LLMs through a single, unified API endpoint.

*   **Single Endpoint:** Access all your models through one API.
*   **Live Management:** Add or remove models dynamically.
*   **Dynamic GPU Management:** Efficient GPU resource distribution using vLLM's sleep/wake.
*   **Interactive Setup:** An easy-to-use wizard guides configuration.

For details, see the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

### Hardware-Optimized Profiles for GPT-OSS Models (v0.2.4)

Built-in profiles optimized for GPT-OSS models:

*   `gpt_oss_ampere` - NVIDIA A100 GPUs
*   `gpt_oss_hopper` - NVIDIA H100/H200 GPUs
*   `gpt_oss_blackwell` - NVIDIA Blackwell GPUs

Leveraging official [vLLM GPT recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html).

### Shortcuts System (v0.2.4)

Save and quickly launch your favorite model + profile combinations:
```bash
vllm-cli serve --shortcut my-gpt-server
```

### Full Ollama Integration (v0.2.4)

*   Automatic discovery of Ollama models
*   GGUF format support (experimental)
*   System and user directory scanning

### Enhanced Configuration (v0.2.4)

*   **Environment Variables:** Universal and profile-specific environment variable management
*   **GPU Selection:** Choose specific GPUs for model serving (`--device 0,1`)
*   **Enhanced System Info:** vLLM feature detection with attention backend availability

See [CHANGELOG.md](CHANGELOG.md) for full release notes.

## Quick Start

### Important: vLLM Installation Notes

⚠️ **Binary Compatibility Warning**: vLLM requires that the pre-compiled CUDA kernels match your PyTorch version exactly. Installing mismatched versions will cause errors.

vLLM-CLI does not install vLLM or PyTorch by default.

### Installation

#### Option 1: Install vLLM Separately and then install vLLM CLI (Recommended)

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

For more detailed instructions, see the [📘 Usage Guide](docs/usage-guide.md) and [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI offers 7 optimized profiles:

**General Purpose:**

*   `standard`: Minimal configuration with smart defaults
*   `high_throughput`: For maximum performance
*   `low_memory`: Suitable for memory-constrained environments
*   `moe_optimized`: Optimized for Mixture of Experts models

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`: For NVIDIA A100 GPUs
*   `gpt_oss_hopper`: For NVIDIA H100/H200 GPUs
*   `gpt_oss_blackwell`: For NVIDIA Blackwell GPUs

See the [**📋 Profiles Guide**](docs/profiles.md) for more details.

### Configuration Files

*   **Main Config:** `~/.config/vllm-cli/config.yaml`
*   **User Profiles:** `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts:** `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [**📘 Usage Guide**](docs/usage-guide.md) - Comprehensive usage instructions
*   [**🌐 Multi-Model Proxy**](docs/multi-model-proxy.md) - Serving multiple models
*   [**📋 Profiles Guide**](docs/profiles.md) - Built-in profile details
*   [**❓ Troubleshooting**](docs/troubleshooting.md) - Solutions to common issues
*   [**📸 Screenshots**](docs/screenshots.md) - Visual feature overview
*   [**🔍 Model Discovery**](docs/MODEL_DISCOVERY_QUICK_REF.md) - Model management guide
*   [**🦙 Ollama Integration**](docs/ollama-integration.md) - Using Ollama models
*   [**⚙️ Custom Models**](docs/custom-model-serving.md) - Serving custom models
*   [**🗺️ Roadmap**](docs/roadmap.md) - Future development plans

## Integration with hf-model-tool

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery:

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

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.