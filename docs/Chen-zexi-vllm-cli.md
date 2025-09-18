# vLLM CLI: Command-Line Interface for LLM Serving with vLLM

**Effortlessly serve and manage Large Language Models (LLMs) using vLLM with an intuitive command-line interface.** ([Original Repo](https://github.com/Chen-zexi/vllm-cli))

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI provides a powerful and user-friendly command-line interface to serve and manage LLMs using vLLM, offering both interactive and command-line modes. It simplifies model deployment with features like model management, server monitoring, and configurable profiles.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Key Features

*   **Interactive Mode:** Navigate and manage your LLMs through a rich, menu-driven terminal interface.
*   **Command-Line Mode:** Automate and script your LLM workflows with direct CLI commands.
*   **Model Management:** Easily discover and load local models, including support for Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles or create custom server settings tailored to your needs.
*   **Server Monitoring:** Monitor the status and performance of your vLLM servers in real-time.
*   **System Information:** Get comprehensive system information, including GPU, memory, and CUDA compatibility checks.
*   **Advanced Configuration:** Fine-tune vLLM parameters for optimal performance and control.
*   **Multi-Model Proxy (Experimental):** Serve multiple models through a single API endpoint (experimental).

## What's New

### Multi-Model Proxy Server (Experimental)

*   **Single Endpoint:** Access all your models through one unified API.
*   **Live Management:** Add or remove models without stopping the service.
*   **Dynamic GPU Management:** Efficient GPU resource allocation through vLLM's sleep/wake functionality.
*   **Interactive Setup:** User-friendly wizard for easy configuration.

### Hardware-Optimized Profiles for GPT-OSS Models
New built-in profiles specifically optimized for serving GPT-OSS models on different GPU architectures:
- **`gpt_oss_ampere`** - Optimized for NVIDIA A100 GPUs
- **`gpt_oss_hopper`** - Optimized for NVIDIA H100/H200 GPUs
- **`gpt_oss_blackwell`** - Optimized for NVIDIA Blackwell GPUs

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

## Configuration

### Built-in Profiles

vLLM CLI provides a set of optimized profiles for various use cases:

**General Purpose:**

*   `standard`: Minimal configuration with smart defaults.
*   `high_throughput`: Optimized for maximum performance.
*   `low_memory`: For memory-constrained environments.
*   `moe_optimized`: Optimized for Mixture of Experts models.

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`: Optimized for NVIDIA A100 GPUs.
*   `gpt_oss_hopper`: Optimized for NVIDIA H100/H200 GPUs.
*   `gpt_oss_blackwell`: Optimized for NVIDIA Blackwell GPUs

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

MIT License - see [LICENSE](LICENSE) file for details.
```
Key improvements and explanations:

*   **SEO Optimization:** Added relevant keywords like "vLLM," "LLM," "command-line," "model serving," etc. in the title and throughout the document.
*   **Clear Hook:** Added a concise, benefit-driven sentence to grab the reader's attention.
*   **Structured Headings:** Used clear, descriptive headings to improve readability and SEO.  Consistent heading levels.
*   **Bulleted Key Features:**  Made key features easy to scan and understand.
*   **Concise Summaries:** Reworded and shortened descriptions for brevity.
*   **Links & Calls to Action:**  Included quick links and encouraged user interaction.
*   **Removed Redundancy:**  Streamlined the text, removing repetition.
*   **Installation Section Refinement**: Improved clarity and included `uv` and conda examples, which are valuable.
*   **Complete and Up-to-date**:  Included all the latest features and links from the original README.
*   **Improved Formatting:** Used Markdown consistently for better visual structure.  Added bolding for emphasis where appropriate.
*   **Focus on Benefits:** Highlighted the *benefits* of using the tool (e.g., "Effortlessly serve," "Automate workflows").
*   **Removed Unnecessary Information**: Removed some redundancies.
*   **Code Blocks Enhancement**: Improved code block styling and formatting.