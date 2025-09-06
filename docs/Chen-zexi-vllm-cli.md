# vLLM CLI: The Ultimate Command-Line Tool for Serving Large Language Models

**Quickly and easily serve your favorite LLMs with vLLM using a powerful CLI, featuring an interactive interface and robust model management.** [Explore the original repo](https://github.com/Chen-zexi/vllm-cli).

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

vLLM CLI provides a user-friendly command-line interface for deploying and managing Large Language Models with vLLM, enabling both interactive and command-line modes for maximum flexibility. It simplifies the process with features like configuration profiles, model management, and real-time server monitoring.

## Key Features

*   **Interactive Mode:** A feature-rich terminal interface provides menu-driven navigation for easy model management.
*   **Command-Line Mode:** Automate tasks and integrate seamlessly into scripts with direct CLI commands.
*   **Model Management:** Effortlessly discover and load local models, with Hugging Face and Ollama support.
*   **Configuration Profiles:** Utilize pre-configured and custom server profiles tailored for different use cases, including optimized profiles for GPT-OSS models.
*   **Server Monitoring:** Monitor your vLLM servers in real-time, gaining insights into performance and resource utilization.
*   **System Information:** Easily check your GPU, memory, and CUDA compatibility to ensure optimal performance.
*   **Advanced Configuration:** Fine-tune vLLM parameters to get the most out of your LLMs.
*   **Multi-Model Proxy (Experimental):** Serve multiple models through a single API endpoint for efficient GPU resource distribution.

## What's New

### Multi-Model Proxy Server (Experimental)

This experimental feature lets you serve multiple LLMs through a single, unified API endpoint for efficient resource management.

**Key Benefits:**

*   **Single API Endpoint:** Access all models through one convenient API.
*   **Dynamic Management:** Easily add or remove models without server downtime.
*   **Efficient Resource Allocation:** Leverage vLLM's sleep/wake functionality for dynamic GPU resource distribution.
*   **User-Friendly Setup:** A helpful wizard guides you through configuration.

*   **Feedback:** Share your experiences via [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).
*   **Complete Guide:** See the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

### Hardware-Optimized Profiles & Shortcuts

*   **Optimized GPT-OSS Profiles:** New built-in profiles for NVIDIA A100, H100/H200, and Blackwell GPUs based on official vLLM recipes for optimal performance.
*   **Shortcuts System:** Save and launch model + profile combinations using shortcuts.

    ```bash
    vllm-cli serve --shortcut my-gpt-server
    ```

### Ollama Integration

*   **Full Ollama Integration:** Automatic discovery of Ollama models, including GGUF format support (experimental).

### Enhanced Configuration

*   **Environment Variables:** Simplified universal and profile-specific environment variable management.
*   **GPU Selection:** Select specific GPUs for model serving (`--device 0,1`).
*   **Enhanced System Info:** vLLM feature detection and attention backend availability.

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Quick Start

### Installation

#### Option 1: Install vLLM Separately (Recommended)
   ```bash
    uv venv --python 3.12 --seed  # or your preferred python version
    source .venv/bin/activate
    uv pip install vllm --torch-backend=auto  # Or specify a backend: cu128
    uv pip install --upgrade vllm-cli
    uv run vllm-cli
   ```

#### Option 2: Install vLLM CLI + vLLM

```bash
pip install vllm-cli[vllm]
vllm-cli
```

#### Option 3: Build from Source

```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

#### Option 4: Isolated Installation (pipx)

```bash
pipx install "vllm-cli[vllm]"
```

### Prerequisites

*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed

### Basic Usage

```bash
# Interactive mode
vllm-cli
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b
# Use a shortcut
vllm-cli serve --shortcut my-model
```

For comprehensive usage, see the [📘 Usage Guide](docs/usage-guide.md) and [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI offers 7 optimized profiles to streamline LLM serving.

**General Purpose:**

*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere` - NVIDIA A100
*   `gpt_oss_hopper` - NVIDIA H100/H200
*   `gpt_oss_blackwell` - NVIDIA Blackwell

See the [**📋 Profiles Guide**](docs/profiles.md) for detailed information.

### Configuration Files

*   **Main Config:** `~/.config/vllm-cli/config.yaml`
*   **User Profiles:** `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts:** `~/.config/vllm-cli/shortcuts.json`

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

vLLM CLI integrates with [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for enhanced model discovery and configuration.

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