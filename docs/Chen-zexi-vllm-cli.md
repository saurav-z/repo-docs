# vLLM CLI: Command-Line Interface for Serving Large Language Models

**Effortlessly serve and manage your Large Language Models (LLMs) with vLLM using a user-friendly command-line interface.**  [Explore the original repo](https://github.com/Chen-zexi/vllm-cli).

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI provides a robust command-line interface and interactive terminal for serving Large Language Models with vLLM, offering features for model management, server monitoring, and advanced configuration.  Whether you're scripting or interacting, vLLM-CLI streamlines your LLM deployment workflow.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Key Features

*   **Interactive Mode:**  A rich, menu-driven terminal interface for easy model management and server control.
*   **Command-Line Mode:**  Automate tasks and integrate vLLM serving into your scripts with direct CLI commands.
*   **Model Management:** Seamlessly discover and manage local models from Hugging Face and Ollama.
*   **Configuration Profiles:**  Use pre-configured profiles or customize server settings to suit your specific needs.
*   **Server Monitoring:**  Get real-time insights into the performance of your vLLM servers.
*   **System Information:**  Quickly check GPU status, memory usage, and CUDA compatibility.
*   **Advanced Configuration:**  Fine-tune vLLM parameters for optimal performance and control.
*   **Multi-Model Proxy (Experimental):** Serve multiple models through a single API endpoint (under active development).
*   **Hardware-Optimized Profiles**: Optimized performance for various GPU architectures.

**Quick Links:** [📖 Docs](#documentation) | [🚀 Quick Start](#quick-start) | [📸 Screenshots](docs/screenshots.md) | [📘 Usage Guide](docs/usage-guide.md) | [❓ Troubleshooting](docs/troubleshooting.md) | [🗺️ Roadmap](docs/roadmap.md)

## What's New

### Multi-Model Proxy Server (Experimental)

The Multi-Model Proxy is a new experimental feature that enables serving multiple LLMs through a single unified API endpoint.

**Key Benefits:**
*   **Unified API Endpoint:** Access all your models through one single API.
*   **Live Management:** Easily add or remove models without interrupting service.
*   **Dynamic GPU Management:** Efficiently manage GPU resources using vLLM's sleep/wake functionality.
*   **User-Friendly Setup:** A step-by-step wizard guides you through configuration.

**Note:** This is an experimental feature.  Provide feedback via [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).

For complete documentation, see the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

### Enhanced Features in v0.2.4

*   **Hardware-Optimized Profiles:** Dedicated profiles for optimal performance with GPT-OSS models on NVIDIA GPUs:  `gpt_oss_ampere`, `gpt_oss_hopper`, and `gpt_oss_blackwell`.
*   **Shortcuts System:** Save and launch your favorite model + profile combinations for quick access.
*   **Full Ollama Integration:** Automatic discovery of Ollama models, GGUF format support (experimental), and system/user directory scanning.
*   **Enhanced Configuration:** Environment variables and specific GPU selection for customized setups.

See [CHANGELOG.md](CHANGELOG.md) for full release notes.

## Quick Start Guide

### Important: vLLM Installation Notes
⚠️ **Binary Compatibility Warning**: vLLM contains pre-compiled CUDA kernels that must match your PyTorch version exactly. Installing mismatched versions will cause errors.

vLLM-CLI will not install vLLM or Pytorch by default.

### Installation

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

For detailed usage instructions, refer to the [📘 Usage Guide](docs/usage-guide.md) and [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI provides several pre-configured profiles to optimize performance:

**General Purpose:**

*   `standard`:  Smart default configuration.
*   `high_throughput`: Optimized for maximum performance.
*   `low_memory`:  For memory-constrained environments.
*   `moe_optimized`: Optimized for Mixture of Experts models.

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`: NVIDIA A100 GPUs
*   `gpt_oss_hopper`: NVIDIA H100/H200 GPUs
*   `gpt_oss_blackwell`: NVIDIA Blackwell GPUs

See [**📋 Profiles Guide**](docs/profiles.md) for profile details.

### Configuration Files

*   **Main Config**: `~/.config/vllm-cli/config.yaml`
*   **User Profiles**: `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts**: `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [**📘 Usage Guide**](docs/usage-guide.md) - Comprehensive usage instructions.
*   [**🌐 Multi-Model Proxy**](docs/multi-model-proxy.md) - Serve multiple models simultaneously.
*   [**📋 Profiles Guide**](docs/profiles.md) - Detailed profile information.
*   [**❓ Troubleshooting**](docs/troubleshooting.md) - Solutions to common issues.
*   [**📸 Screenshots**](docs/screenshots.md) - Visual feature overview.
*   [**🔍 Model Discovery**](docs/MODEL_DISCOVERY_QUICK_REF.md) - Model management guide.
*   [**🦙 Ollama Integration**](docs/ollama-integration.md) - Using Ollama models.
*   [**⚙️ Custom Models**](docs/custom-model-serving.md) - Serving custom models.
*   [**🗺️ Roadmap**](docs/roadmap.md) - Future development plans.

## Integration with hf-model-tool

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for:

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

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.
```
Key improvements and optimization for SEO:

*   **Strong Hook:** Added a concise, benefit-driven first sentence.
*   **Keyword Optimization:** Used relevant keywords like "vLLM," "LLMs," "command-line interface," "model management," "server monitoring," etc. throughout the document.
*   **Clear Headings and Structure:** Organized content with clear, descriptive headings.
*   **Bulleted Lists:** Used bullet points to highlight key features and benefits, making the information easily scannable.
*   **Concise Language:**  Removed unnecessary words and kept the language direct and to the point.
*   **Emphasis on Benefits:** Highlighted the advantages of using the tool.
*   **Internal Linking:**  Provided links to relevant sections.
*   **SEO-Friendly Formatting:** Used Markdown for proper heading structure and formatting for readability.
*   **Call to Action (Implicit):** Encourages readers to explore the features and documentation.
*   **Clear Installation Instructions:**  Improved quick start and installation steps.
*   **Concise Summaries:**  Summarized complex information into easy-to-understand bullet points.
*   **Actionable advice:** added troubleshooting and vLLM installation notes