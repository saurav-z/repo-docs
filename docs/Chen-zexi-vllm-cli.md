# vLLM CLI: Supercharge Your Large Language Models with Ease

**Effortlessly serve and manage your LLMs with the vLLM CLI, featuring an interactive terminal, command-line automation, and powerful model management.** ([Original Repository](https://github.com/Chen-zexi/vllm-cli))

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI is a powerful command-line interface designed for serving and managing Large Language Models (LLMs) using the vLLM framework. It provides both interactive and command-line modes, offering a user-friendly experience for both beginners and experienced users.  Configure profiles, manage models, monitor servers, and customize every aspect of your LLM deployment.

**Key Features:**

*   **Interactive Mode:** Rich terminal interface with menu-driven navigation for easy model selection and server control.
*   **Command-Line Mode:** Automate tasks and integrate seamlessly with your existing scripts using direct CLI commands.
*   **Model Management:**  Automatic discovery of local models from Hugging Face Hub and Ollama.
*   **Configuration Profiles:**  Use pre-configured profiles or create custom server profiles tailored to your specific needs.
*   **Server Monitoring:** Real-time monitoring of your active vLLM servers, providing insights into performance.
*   **System Information:**  Quickly check GPU, memory, and CUDA compatibility to ensure optimal performance.
*   **Advanced Configuration:**  Fine-tune vLLM parameters with validation, giving you complete control over your LLM deployments.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint for efficient resource utilization.

**Quick Links:** [📖 Docs](#documentation) | [🚀 Quick Start](#quick-start) | [📸 Screenshots](docs/screenshots.md) | [📘 Usage Guide](docs/usage-guide.md) | [❓ Troubleshooting](docs/troubleshooting.md) | [🗺️ Roadmap](docs/roadmap.md)

## What's New

**Recent Updates:**

*   **Multi-Model Proxy Server (Experimental):**  Serve multiple models through a unified API endpoint.
*   **Hardware-Optimized Profiles:** Built-in profiles optimized for GPT-OSS models on various NVIDIA GPUs (Ampere, Hopper, and Blackwell).
*   **Shortcuts System:** Save and quickly launch your favorite model + profile combinations.
*   **Full Ollama Integration:** Automatic discovery of Ollama models and GGUF format support.
*   **Enhanced Configuration:** Universal and profile-specific environment variable management, GPU selection, and enhanced system info.

For detailed release notes, see the [CHANGELOG.md](CHANGELOG.md).

## Quick Start

### Important: vLLM Installation Notes

⚠️ **Binary Compatibility Warning**: vLLM contains pre-compiled CUDA kernels that must match your PyTorch version exactly. Installing mismatched versions will cause errors. vLLM-CLI will not install vLLM or Pytorch by default.

### Installation

**Option 1: Install vLLM separately (Recommended)**

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

**Option 2: Install vLLM CLI + vLLM**

```bash
pip install vllm-cli[vllm]
vllm-cli
```

**Option 3: Build from Source (Requires vLLM installation)**

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

**Option 4: Isolated Installation (pipx/system packages)**

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
vllm-cli
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

For detailed usage instructions, see the [📘 Usage Guide](docs/usage-guide.md) and [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI offers pre-configured profiles for different use cases.

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

vLLM CLI integrates with [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for:

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

Contributions are welcome!  Feel free to open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) file for details.
```
Key improvements and SEO considerations:

*   **Headline Optimization:**  Uses keywords like "vLLM CLI," "Large Language Models," and action verbs like "Supercharge" and "Effortlessly serve" to attract search traffic.
*   **Concise Hook:** Starts with a compelling one-sentence summary that grabs attention.
*   **Keyword Density:**  Strategically includes relevant keywords throughout the README.
*   **Bulleted Lists:**  Clearly presents key features and benefits, making them easy to scan.
*   **Clear Headings and Structure:** Organizes information with clear headings and subheadings for better readability and SEO.
*   **Internal Linking:** Links to other sections within the README.
*   **External Linking:** Maintains links to the original repository and relevant documentation.
*   **Emphasis on Benefits:** Highlights the advantages of using vLLM CLI (ease of use, automation, control).
*   **Up-to-date Content:**  Includes the latest features and updates from the original README.
*   **Call to Action:** Encourages contribution.