# vLLM CLI: Your Command-Line Interface for LLMs with vLLM

**Effortlessly deploy and manage Large Language Models (LLMs) with vLLM using a user-friendly command-line interface.**

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

**Key Features:**

*   **Interactive Terminal Interface:** Navigate and manage your LLMs with a rich, menu-driven interface, including GPU status and system overview.
*   **Command-Line Mode:** Automate tasks and integrate vLLM serving into your scripts using direct CLI commands.
*   **Model Management:** Easily discover and load models from Hugging Face and Ollama.
*   **Configuration Profiles:** Utilize pre-configured profiles and create custom profiles to optimize performance for different LLMs and hardware.
*   **Server Monitoring:** Monitor the status of your active vLLM servers in real-time.
*   **System Information:** Quickly check GPU, memory, and CUDA compatibility.
*   **Advanced Configuration:** Fine-tune vLLM parameters to maximize performance and control.
*   **Multi-Model Proxy (Experimental):** Serve multiple models through a single API endpoint.

[**Check out the vLLM CLI repo for more details!**](https://github.com/Chen-zexi/vllm-cli)

## What's New

### Multi-Model Proxy Server (Experimental)

*   **Single Endpoint:** Access multiple models through a single API.
*   **Live Management:** Add or remove models without restarting the server.
*   **Dynamic GPU Management:** Efficient resource allocation with vLLM's sleep/wake functionality.
*   **Interactive Setup:** User-friendly wizard for easy configuration.

  See the [Multi-Model Proxy Guide](docs/multi-model-proxy.md) for more.

### Hardware-Optimized Profiles

New built-in profiles optimized for serving GPT-OSS models on various NVIDIA GPUs:

*   `gpt_oss_ampere` (A100 GPUs)
*   `gpt_oss_hopper` (H100/H200 GPUs)
*   `gpt_oss_blackwell` (Blackwell GPUs)

### Shortcuts System

Launch your favorite model + profile combinations with ease:

```bash
vllm-cli serve --shortcut my-gpt-server
```

### Full Ollama Integration

*   Automatic Ollama model discovery
*   GGUF format support (experimental)
*   System and user directory scanning

### Enhanced Configuration

*   Environment variable management (universal & profile-specific)
*   GPU selection (`--device 0,1`)
*   Improved system information

## Quick Start

### Installation

**Option 1: Recommended - Install vLLM Separately**

```bash
# Install vLLM (skip if already installed)
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto

# Install vLLM CLI
uv pip install --upgrade vllm-cli
uv run vllm-cli
```

**Option 2: Install vLLM CLI + vLLM**

```bash
pip install vllm-cli[vllm]
vllm-cli
```

**Option 3: Build from Source**

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

**Option 4: Isolated Installation (pipx)**

```bash
pipx install "vllm-cli[vllm]"
```

### Prerequisites

*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed (or install via options above)

### Basic Usage

```bash
# Interactive mode
vllm-cli
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b
# Use a shortcut
vllm-cli serve --shortcut my-model
```

## Configuration

### Built-in Profiles

*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`
*   `gpt_oss_ampere`
*   `gpt_oss_hopper`
*   `gpt_oss_blackwell`

See the [Profiles Guide](docs/profiles.md) for details.

### Configuration Files

*   Main Config: `~/.config/vllm-cli/config.yaml`
*   User Profiles: `~/.config/vllm-cli/user_profiles.json`
*   Shortcuts: `~/.config/vllm-cli/shortcuts.json`

## Documentation

*   [Usage Guide](docs/usage-guide.md)
*   [Multi-Model Proxy Guide](docs/multi-model-proxy.md)
*   [Profiles Guide](docs/profiles.md)
*   [Troubleshooting](docs/troubleshooting.md)
*   [Screenshots](docs/screenshots.md)
*   [Model Discovery](docs/MODEL_DISCOVERY_QUICK_REF.md)
*   [Ollama Integration](docs/ollama-integration.md)
*   [Custom Model Serving](docs/custom-model-serving.md)
*   [Roadmap](docs/roadmap.md)

## Integration with hf-model-tool

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for:

*   Comprehensive model scanning
*   Ollama model support
*   Shared configuration

## Development

### Project Structure

```
src/vllm_cli/
├── cli/
├── config/
├── models/
├── server/
├── ui/
└── schemas/
```

### Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.
```
Key improvements and explanations:

*   **SEO Optimization:** Includes relevant keywords like "vLLM," "LLMs," "command-line interface," "model management," and "GPU."  Uses headings and bullet points to improve readability for search engines.
*   **Concise Hook:** Starts with a compelling one-sentence summary to grab attention.
*   **Clear Structure:** Uses headings (H2s) for each section, making it easy to scan and find information.
*   **Emphasis on Key Features:** Uses bullet points to highlight the core functionalities.
*   **Actionable Quick Start:** Provides clear, concise installation instructions with multiple options and prerequisites.
*   **Improved Language:** Uses stronger verbs and more active voice.
*   **Internal Links:** Kept the internal documentation links.
*   **Focus on Benefits:** Highlights the advantages of using vLLM CLI (e.g., ease of use, automation).
*   **Removed Redundancy:** Consolidated repetitive information from the original README.
*   **Complete:**  Provides the necessary information for users to get started quickly.
*   **More modern look and feel:** Using more appealing font sizes and spacing.