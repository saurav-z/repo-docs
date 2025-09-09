# vLLM CLI: Your Command-Line Interface for LLMs

**Quickly deploy and manage Large Language Models with vLLM using a user-friendly command-line interface.**  [Explore the vLLM CLI on GitHub](https://github.com/Chen-zexi/vllm-cli).

vLLM CLI provides a robust and flexible way to interact with and serve LLMs, offering both interactive and command-line modes, extensive model management, and advanced configuration options.

**Key Features:**

*   **Interactive Mode:** Navigate and control your LLMs with a rich, terminal-based interface.
*   **Command-Line Mode:** Automate and script your LLM operations with direct CLI commands.
*   **Model Management:**
    *   Automatic discovery of local models from Hugging Face and Ollama.
    *   Effortlessly manage and load different LLM models.
*   **Configuration Profiles:**
    *   Utilize pre-configured server profiles for various use cases.
    *   Customize profiles for optimal performance.
*   **Server Monitoring:** Real-time insights into active vLLM server performance.
*   **System Information:** GPU, memory, and CUDA compatibility checks for optimal setup.
*   **Advanced Configuration:** Fine-tune vLLM parameters for advanced control and validation.
*   **Multi-Model Proxy (Experimental):**
    *   Serve multiple LLMs through a single unified API endpoint.
    *   Dynamically manage GPU resources and easily add or remove models.

## What's New

### Multi-Model Proxy Server (Experimental)
Enable serving multiple LLMs through a single unified API endpoint.

### Hardware-Optimized Profiles for GPT-OSS Models
New built-in profiles specifically optimized for serving GPT-OSS models.

### ⚡ Shortcuts System
Save and quickly launch your favorite model + profile combinations.

### 🦙 Full Ollama Integration
- Automatic discovery of Ollama models
- GGUF format support (experimental)
- System and user directory scanning

### 🔧 Enhanced Configuration
- **Environment Variables** - Universal and profile-specific environment variable management
- **GPU Selection** - Choose specific GPUs for model serving (`--device 0,1`)
- **Enhanced System Info** - vLLM feature detection with attention backend availability

## Installation

### Prerequisites
*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed
*   For dependency issues, see [Troubleshooting Guide](docs/troubleshooting.md#dependency-conflicts)

### Recommended Installation (with vLLM installed separately):

```bash
# Install vLLM -- Skip this step if you have vllm installed in your environment
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
# Or specify a backend: uv pip install vllm --torch-backend=cu128

# Install vLLM CLI
uv pip install --upgrade vllm-cli
uv run vllm-cli
```

Alternatively, install both vLLM and vLLM CLI together:

```bash
pip install vllm-cli[vllm]
vllm-cli
```

## Quick Start

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