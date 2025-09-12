# vLLM CLI: Command-Line Interface for Serving Large Language Models with vLLM

**Effortlessly deploy and manage Large Language Models (LLMs) with vLLM using a powerful and user-friendly command-line interface.**  [Explore the original repository](https://github.com/Chen-zexi/vllm-cli).

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI provides a robust command-line interface (CLI) and interactive terminal experience for serving LLMs using the optimized vLLM inference engine. This tool streamlines model management, server monitoring, and configuration, making it easy to deploy and experiment with cutting-edge language models.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Key Features

*   **Interactive Mode:** Navigate and manage your LLM servers through a rich, menu-driven terminal interface.
*   **Command-Line Mode:** Automate tasks and integrate vLLM serving into your scripts with direct CLI commands.
*   **Model Management:** Easily discover and load models from Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles optimized for various use cases, including GPT-OSS models.
*   **Server Monitoring:** Monitor the status of your vLLM servers in real-time.
*   **System Information:** Check GPU, memory, and CUDA compatibility to ensure optimal performance.
*   **Advanced Configuration:** Fine-tune vLLM parameters with comprehensive control and validation.
*   **Multi-Model Proxy (Experimental):** Serve multiple models through a single API endpoint with dynamic GPU management.
*   **Shortcuts:** Save and quickly launch your favorite model + profile combinations.
*   **Ollama Integration:** Full support for Ollama models, including GGUF format (experimental).

## What's New

### Multi-Model Proxy Server (Experimental)

*   **Single Endpoint:** Access all your models through a unified API.
*   **Live Management:** Add or remove models without stopping the service.
*   **Dynamic GPU Management:** Efficiently distribute GPU resources with vLLM's sleep/wake functionality.
*   **Interactive Setup:** Configuration wizard to guide you.

See the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md) for details.

### Hardware-Optimized Profiles for GPT-OSS Models

*   **`gpt_oss_ampere`:** Optimized for NVIDIA A100 GPUs
*   **`gpt_oss_hopper`:** Optimized for NVIDIA H100/H200 GPUs
*   **`gpt_oss_blackwell`:** Optimized for NVIDIA Blackwell GPUs

### Shortcuts System

*   Quickly launch models with saved shortcuts
*   `vllm-cli serve --shortcut my-gpt-server`

### Ollama Integration

*   Automatic discovery of Ollama models
*   GGUF format support (experimental)

See [CHANGELOG.md](CHANGELOG.md) for more details.

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
*   vLLM package installed (see installation instructions above)
*   For dependency issues, see [Troubleshooting Guide](docs/troubleshooting.md#dependency-conflicts)

### Basic Usage

```bash
# Interactive mode
vllm-cli
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b
# Use a shortcut
vllm-cli serve --shortcut my-model
```

For more details, see the [📘 Usage Guide](docs/usage-guide.md) and the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI offers built-in profiles to optimize your serving experience:

**General Purpose:**
*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

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

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for robust model discovery and management.

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

MIT License - see [LICENSE](LICENSE) file for details.
```
Key improvements and SEO optimizations:

*   **Clear Title:**  "vLLM CLI: Command-Line Interface for Serving Large Language Models with vLLM" uses the target keywords.
*   **One-Sentence Hook:** "Effortlessly deploy and manage Large Language Models (LLMs) with vLLM using a powerful and user-friendly command-line interface." immediately grabs attention.
*   **Keyword Density:** Uses keywords like "vLLM," "LLMs," "command-line interface," "serve," "deploy," and "manage" throughout.
*   **Heading Structure:**  Uses clear headings (H2) for better readability and SEO.
*   **Bulleted Lists:**  Highlights key features and benefits.
*   **Emphasis on Benefits:** Focuses on *what* users can *do* with the tool.
*   **Contextual Links:** Links to relevant documentation pages, providing value to the user and improving SEO.
*   **Clear Installation:**  Improved the installation section, making it clearer and more user-friendly.  Added installation notes with the warning.
*   **Concise Summary:** Provides a concise overview of the tool and its capabilities.
*   **Optimized Content:** Avoids repetition and focuses on the core value proposition.
*   **Call to Action:** Encourages users to explore the original repo.
*   **Simplified Examples:** Streamlined code examples.
*   **Development & Contribution:**  Includes details about contributing and project structure for potential developers.
*   **Clear License:**  Includes the License information.