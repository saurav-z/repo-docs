# vLLM CLI: Supercharge Your LLMs with a Powerful Command-Line Interface

**Effortlessly serve and manage Large Language Models (LLMs) with vLLM using the intuitive and feature-rich vLLM CLI.** ([Original Repository](https://github.com/Chen-zexi/vllm-cli))

vLLM CLI empowers you to quickly deploy and monitor LLMs with ease, offering both interactive and command-line modes, model management, and advanced configuration options.

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)
<br>
![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Key Features

*   **Interactive Mode:** Navigate and manage your LLMs with a user-friendly terminal interface.
*   **Command-Line Mode:** Automate tasks and integrate with scripts using direct CLI commands.
*   **Model Management:** Seamlessly discover and manage local models with Hugging Face and Ollama support.
*   **Configuration Profiles:** Utilize pre-configured profiles and create custom server profiles tailored to your needs.
*   **Server Monitoring:** Get real-time insights into your active vLLM servers.
*   **System Information:** Quickly check GPU, memory, and CUDA compatibility.
*   **Advanced Configuration:** Fine-tune vLLM parameters for optimal performance.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint with dynamic GPU management.
*   **Hardware-Optimized Profiles:** Pre-built profiles for specific GPT-OSS models (A100, H100/H200, and Blackwell).
*   **Shortcuts System:** Save and launch your favorite model + profile combinations.
*   **Full Ollama Integration:** Supports Ollama models, including GGUF format (experimental).

## What's New

### Multi-Model Proxy Server (Experimental)

*   **Single Endpoint:** Access all your models through a unified API.
*   **Live Management:** Add or remove models without service interruption.
*   **Dynamic GPU Management:** Efficient GPU resource distribution via vLLM's sleep/wake.
*   **Interactive Setup:** User-friendly wizard for streamlined configuration.

**Note:** This is an experimental feature under active development. Share feedback through [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues). For complete documentation, see the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

### Hardware-Optimized Profiles for GPT-OSS Models
New built-in profiles specifically optimized for serving GPT-OSS models on different GPU architectures:
-   `gpt_oss_ampere` - Optimized for NVIDIA A100 GPUs
-   `gpt_oss_hopper` - Optimized for NVIDIA H100/H200 GPUs
-   `gpt_oss_blackwell` - Optimized for NVIDIA Blackwell GPUs

Based on official [vLLM GPT recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) for maximum performance.

### Shortcuts System
Save and quickly launch your favorite model + profile combinations:
```bash
vllm-cli serve --shortcut my-gpt-server
```

### Full Ollama Integration
-   Automatic discovery of Ollama models
-   GGUF format support (experimental)
-   System and user directory scanning

### Enhanced Configuration
-   **Environment Variables** - Universal and profile-specific environment variable management
-   **GPU Selection** - Choose specific GPUs for model serving (`--device 0,1`)
-   **Enhanced System Info** - vLLM feature detection with attention backend availability

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Quick Start

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
*   vLLM package installed (See Installation instructions)
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

vLLM CLI uses [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery:

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

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) file for details.