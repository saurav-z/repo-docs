<!-- SEO-optimized README for vLLM CLI -->

# vLLM CLI: Effortlessly Serve Large Language Models (LLMs) from the Command Line

**Quickly deploy and manage powerful LLMs with a user-friendly command-line interface, built on the blazing-fast vLLM engine.** ([Original Repository](https://github.com/Chen-zexi/vllm-cli))

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI simplifies serving LLMs, offering both interactive and command-line modes for efficient model management, server monitoring, and configuration. Get started quickly and unlock the power of LLMs with ease.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Key Features

*   **🚀 Blazing-Fast Performance:** Powered by vLLM for optimal inference speed and efficiency.
*   **💻 Command-Line & Interactive Modes:** Choose your preferred interface for model deployment and management.
*   **🤖 Intelligent Model Discovery:** Seamlessly integrates with Hugging Face and Ollama for effortless model loading.
*   **🔧 Customizable Configuration Profiles:** Tailor server settings for diverse use cases and hardware.
*   **📊 Real-time Server Monitoring:** Keep tabs on your LLM servers with built-in monitoring tools.
*   **🖥️ System Information & Compatibility Checks:** Ensure your environment is ready for LLM deployment.
*   **🌐 Experimental Multi-Model Proxy Server:** Serve multiple models through a single endpoint.
*   **⚡ Hardware-Optimized Profiles:** Pre-configured settings for optimal performance on various GPU architectures (GPT-OSS).

## What's New

### Multi-Model Proxy Server (Experimental)

Serve multiple LLMs through a single API endpoint. Includes live management, dynamic GPU allocation, and an interactive setup wizard. Explore the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md) for more details and provide feedback via [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).

### Recent Updates
*   Hardware Optimized Profiles for GPT-OSS Models
*   Shortcuts System
*   Full Ollama Integration
*   Enhanced Configuration

See [CHANGELOG.md](CHANGELOG.md) for a complete overview of the recent releases.

## Getting Started

### Installation

Choose your preferred installation method:

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
*   CUDA-compatible GPU (recommended for optimal performance)
*   vLLM package installed (installation handled by vllm-cli[vllm])
*   For dependency issues, see [Troubleshooting Guide](docs/troubleshooting.md#dependency-conflicts)

### Basic Usage

```bash
# Start the interactive mode - a menu-driven interface
vllm-cli

# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

For detailed usage, refer to the [📘 Usage Guide](docs/usage-guide.md).

## Configuration & Profiles

### Built-in Profiles

vLLM CLI offers optimized profiles for various use cases:

**General Purpose:**

*   `standard` - Smart defaults
*   `high_throughput` - Maximum performance
*   `low_memory` - Memory-constrained
*   `moe_optimized` - Optimized for Mixture of Experts models

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere` - NVIDIA A100 GPUs
*   `gpt_oss_hopper` - NVIDIA H100/H200 GPUs
*   `gpt_oss_blackwell` - NVIDIA Blackwell GPUs

Learn more in the [**📋 Profiles Guide**](docs/profiles.md).

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

vLLM CLI utilizes [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery, offering:

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

Contributions are welcome! Please submit issues and pull requests to help improve vLLM CLI.

## License

MIT License - see the [LICENSE](LICENSE) file for details.