<!-- SEO-optimized README for vLLM CLI -->
# vLLM CLI: Supercharge Your LLMs with a Powerful Command-Line Interface

**Easily serve, manage, and monitor Large Language Models (LLMs) with vLLM using an intuitive command-line interface.**  ([View on GitHub](https://github.com/Chen-zexi/vllm-cli))

## Key Features

*   **Interactive Mode:** Navigate a rich, menu-driven terminal interface for effortless LLM management.
*   **Command-Line Mode:** Automate LLM tasks with direct CLI commands for scripting and integration.
*   **Model Management:** Seamlessly discover and manage local models, including support for Hugging Face and Ollama.
*   **Configuration Profiles:** Leverage pre-configured and custom server profiles to optimize performance for diverse use cases.
*   **Server Monitoring:** Real-time monitoring of active vLLM servers, providing valuable insights.
*   **System Information:** Comprehensive system checks for GPU, memory, and CUDA compatibility, ensuring a smooth setup.
*   **Advanced Configuration:** Fine-tune vLLM parameters with full control and validation, enabling advanced customization.
*   **Multi-Model Proxy (Experimental):** Serve multiple LLMs through a single API endpoint for efficient resource utilization.

## What's New

**v0.2.5**

*   **Multi-Model Proxy Server (Experimental):** This experimental feature allows serving multiple LLMs through a single, unified API endpoint. Benefits include a single endpoint for all models, live model management, dynamic GPU management, and an interactive setup wizard.  [Multi-Model Proxy Guide](docs/multi-model-proxy.md)

**v0.2.4**

*   **Hardware-Optimized Profiles:** Built-in profiles optimized for GPT-OSS models across various NVIDIA GPU architectures (Ampere, Hopper, Blackwell), based on official vLLM GPT recipes.
*   **Shortcuts System:** Save and launch your favorite model + profile combinations with a simple command (e.g., `vllm-cli serve --shortcut my-gpt-server`).
*   **Full Ollama Integration:** Automatic discovery of Ollama models, GGUF format support (experimental), and system/user directory scanning.
*   **Enhanced Configuration:** Environment variables and GPU selection features provide enhanced configuration options.

## Quick Start

### Installation

**Important:** This CLI leverages vLLM.  It's recommended to install vLLM and PyTorch separately to ensure proper CUDA compatibility.

#### Option 1: Recommended: Install vLLM Separately Then Install vLLM CLI

```bash
# Create and activate a virtual environment:
uv venv --python 3.12 --seed
source .venv/bin/activate
# Install vLLM (specify CUDA backend if needed)
uv pip install vllm --torch-backend=auto  # Or specify e.g., --torch-backend=cu128

# Install vLLM CLI
uv pip install --upgrade vllm-cli
uv run vllm-cli  # or vllm-cli
```

#### Option 2: Install vLLM CLI with vLLM

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
pipx install "vllm-cli[vllm]"  # Or install pre-release versions: pipx install --pip-args="--pre" "vllm-cli[vllm]"
```

### Prerequisites

*   Python 3.9+
*   CUDA-compatible GPU (recommended)
*   vLLM package installed
*   For dependency issues, see the [Troubleshooting Guide](docs/troubleshooting.md#dependency-conflicts)

### Basic Usage

```bash
# Start in interactive mode
vllm-cli

# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut (if defined)
vllm-cli serve --shortcut my-model
```

For detailed instructions, see the [Usage Guide](docs/usage-guide.md) and the [Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI offers pre-configured profiles for diverse needs:

**General Purpose:**

*   `standard`
*   `high_throughput`
*   `low_memory`
*   `moe_optimized`

**Hardware-Specific (GPT-OSS):**

*   `gpt_oss_ampere`
*   `gpt_oss_hopper`
*   `gpt_oss_blackwell`

See the [Profiles Guide](docs/profiles.md) for detailed configuration information.

### Configuration Files

*   **Main Config:** `~/.config/vllm-cli/config.yaml`
*   **User Profiles:** `~/.config/vllm-cli/user_profiles.json`
*   **Shortcuts:** `~/.config/vllm-cli/shortcuts.json`

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

vLLM CLI leverages [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for efficient model discovery, with features like comprehensive model scanning and Ollama model support.

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

Contributions are highly encouraged!  Please submit issues or pull requests.

## License

MIT License - See the [LICENSE](LICENSE) file for details.
```
Key improvements and SEO optimizations:

*   **Clear, Concise Title:** The title is now more descriptive and uses keywords.
*   **One-Sentence Hook:**  A strong opening sentence immediately explains the tool's purpose.
*   **Keywords:** Strategic use of keywords like "vLLM," "LLM," "command-line," "serve," "manage," "monitor," and "GPT-OSS" throughout the document.
*   **Bulleted Key Features:**  Easily scannable bullet points highlight core functionality.
*   **Clear Section Headings:**  Well-defined headings for easy navigation.
*   **Focus on Benefits:**  Each feature description emphasizes the user benefits.
*   **Simplified Quick Start:** Installation instructions are clearer and prioritize the recommended approach.  Includes a clear warning about CUDA compatibility.
*   **Internal Links:** Consistent use of internal links to other sections, helping with SEO and navigation.  Also includes the original docs links.
*   **Up-to-Date Content:**  The "What's New" section incorporates information from the original README.
*   **Concise Language:** Improved readability and conciseness throughout.
*   **Call to Action:** Explicitly encourages contributions.
*   **GitHub Link:** The repository link is prominently displayed.
*   **Reorganized:** The content has been reorganized for better flow and SEO.
*   **Profile and Feature Focus:** Highlights the critical features (profiles, shortcuts, Ollama integration, hardware optimization).
*   **Updated Examples:** Keeps the code examples and quick start instructions concise and actionable.
*   **Detailed Configuration:** Expands on profiles and configs.
*   **Model Discovery Tool Highlight:** Gives more emphasis to the model discovery integration.