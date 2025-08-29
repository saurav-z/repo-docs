# vLLM CLI: Command-Line Interface for Serving Large Language Models

**Easily serve and manage your LLMs with vLLM using this powerful command-line interface.** [See the original repo](https://github.com/Chen-zexi/vllm-cli).

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI provides a user-friendly interface for serving LLMs. You can use it to create and manage profiles, and monitor server performance.

## Key Features

*   **Interactive Terminal Interface:** Navigate menus, view GPU status, and manage your LLM serving environment.
*   **Command-Line Mode:** Automate tasks and integrate LLM serving into your scripts with direct CLI commands.
*   **Model Management:** Automatically discovers local models from Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles or customize server settings for different use cases and hardware.
*   **Server Monitoring:** Get real-time insights into active vLLM server performance.
*   **System Information:** Checks GPU, memory, and CUDA compatibility.
*   **Advanced Configuration:**  Full control over vLLM parameters.

## What's New - Highlights

*   **Multi-Model Proxy Server (Experimental):** Serve multiple LLMs through a single API endpoint, with live management and dynamic GPU allocation.
*   **Hardware-Optimized Profiles:** Pre-built profiles for optimal performance with GPT-OSS models on various NVIDIA GPUs (A100, H100/H200, Blackwell).
*   **Shortcuts System:**  Save and quickly launch your favorite model + profile combinations.
*   **Full Ollama Integration:** Automatic discovery, GGUF format support (experimental), and system/user directory scanning for Ollama models.
*   **Enhanced Configuration:** Manage environment variables, choose specific GPUs, and get enhanced system info.

## Quick Start

### Installation

**Important:** Ensure you have vLLM and PyTorch installed. vLLM CLI will not automatically install vLLM or PyTorch. Choose one of the following installation methods:

#### Option 1: (Recommended) Install vLLM and then vLLM CLI

```bash
# Install vLLM
uv venv --python 3.12 --seed  # or your preferred method
source .venv/bin/activate #or conda activate your_env
uv pip install vllm --torch-backend=auto
# Or specify a backend: uv pip install vllm --torch-backend=cu128

# Install vLLM CLI
uv pip install --upgrade vllm-cli
uv run vllm-cli
```

#### Option 2: Install vLLM CLI and vLLM together

```bash
pip install vllm-cli[vllm]
vllm-cli
```

#### Option 3: Build from Source

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e . # This installs vllm-cli, but you still need vllm
```

#### Option 4: Isolated Installation (pipx)

```bash
# If you do not want to use virtual environment and want to install vLLM along with vLLM CLI
pipx install "vllm-cli[vllm]"

# If you want to install pre-release version
pipx install --pip-args="--pre" "vllm-cli[vllm]"
```

### Basic Usage

```bash
# Interactive Mode
vllm-cli

# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

For detailed instructions, see the [Usage Guide](docs/usage-guide.md).

## Configuration

### Built-in Profiles

vLLM CLI includes pre-configured profiles for different use cases:

**General Purpose:**

*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere` (NVIDIA A100)
*   `gpt_oss_hopper` (NVIDIA H100/H200)
*   `gpt_oss_blackwell` (NVIDIA Blackwell)

See the [Profiles Guide](docs/profiles.md) for details.

### Configuration Files

*   **Main Config:** `~/.config/vllm-cli/config.yaml`
*   **User Profiles:** `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts:** `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [Usage Guide](docs/usage-guide.md)
*   [Multi-Model Proxy](docs/multi-model-proxy.md)
*   [Profiles Guide](docs/profiles.md)
*   [Troubleshooting](docs/troubleshooting.md)
*   [Screenshots](docs/screenshots.md)
*   [Model Discovery](docs/MODEL_DISCOVERY_QUICK_REF.md)
*   [Ollama Integration](docs/ollama-integration.md)
*   [Custom Models](docs/custom-model-serving.md)
*   [Roadmap](docs/roadmap.md)

## Integration with hf-model-tool

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for comprehensive model discovery and management.

## Development

See the [project structure](src/vllm_cli/) and [CONTRIBUTING](CONTRIBUTING.md) to get started.

## License

MIT License. See [LICENSE](LICENSE) for details.